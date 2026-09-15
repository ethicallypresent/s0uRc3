"""Live think / answer splitter. Ports WPF RenderStreamBuffer + StripJsonFence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamView:
    thinking: str
    answer: str
    is_thinking: bool
    thinking_label: str


def _find_ci(hay: str, needle: str, start: int = 0) -> int:
    return hay.lower().find(needle.lower(), start)


def _try_parse_json_object(buf: str) -> dict | None:
    stripped = (buf or "").strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


_LIVE_THINK_RE = re.compile(r'"think"\s*:\s*"((?:[^"\\]|\\.)*)')


def _live_think_preview(buf: str) -> str:
    """Best-effort live preview of a still-streaming core/action_schema.py
    response's `think` field, before the full object is parseable — lets
    the TUI show something while a structured completion is in flight
    instead of going silent until the whole JSON object lands."""
    m = _LIVE_THINK_RE.search(buf or "")
    if not m:
        return ""
    try:
        return json.loads(f'"{m.group(1)}"')
    except (json.JSONDecodeError, ValueError):
        return m.group(1)


def _structured_answer_text(obj: dict) -> str:
    """Short human-facing line for a parsed core/action_schema.py action —
    mirrors what strip_json_fence/_finish_summary already surface for the
    legacy free-text bare-JSON case."""
    if str(obj.get("action") or "") == "finish":
        finish = obj.get("finish")
        if isinstance(finish, dict):
            summary = finish.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
    rationale = obj.get("rationale")
    return rationale.strip() if isinstance(rationale, str) else ""


def find_think_open(buf: str) -> tuple[int, int]:
    a = _find_ci(buf, "<think>")
    if a >= 0:
        return a, 7
    b = _find_ci(buf, "<thinking>")
    if b >= 0:
        return b, 10
    return -1, 0


def find_think_close(buf: str, start: int) -> int:
    a = _find_ci(buf, "</think>", start)
    b = _find_ci(buf, "</thinking>", start)
    if a < 0:
        return b
    if b < 0:
        return a
    return min(a, b)


def _looks_like_leaked_packet(text: str) -> bool:
    """The model sometimes ignores "never write the packet below into your
    output" (core/loop.py) and echoes its own input packet back instead of
    a real answer — confirmed live in the TUI, where the raw packet dump
    (`{"user_goal":"...","mode":"action","perception":{...`) was shown to
    the user verbatim. core/loop.py always builds that payload with
    "user_goal" as its first key, which never appears in a legitimate
    <think> block or json action — a reliable, cheap signature to catch.
    """
    stripped = (text or "").strip()
    return stripped.startswith('{"user_goal"') or stripped.startswith("{'user_goal'")


def strip_json_fence(text: str) -> str:
    """Prefer finish.summary when the model emitted an action JSON blob."""
    if not text:
        return ""
    if _looks_like_leaked_packet(text):
        return ""
    low = text.lower()
    fence = low.find("```json")
    if fence < 0:
        return _strip_bare_action_json(text)
    visible = text[:fence].strip()
    json_part = text[fence + 7 :]
    end = json_part.find("```")
    if end >= 0:
        json_part = json_part[:end]
    summary = _finish_summary(json_part.strip())
    if summary:
        return summary if not visible else visible + "\n\n" + summary
    return visible if visible else _strip_bare_action_json(text)


def _finish_summary(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    if not isinstance(obj, dict):
        return ""
    finish = obj.get("finish") or {}
    if isinstance(finish, dict):
        summary = finish.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    return ""


def _strip_bare_action_json(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("{") or '"action"' not in stripped:
        return text
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return text
    summary = _finish_summary(stripped[start : end + 1])
    if summary:
        prefix = stripped[:start].strip()
        return summary if not prefix else prefix + "\n\n" + summary
    # Hide internal action JSON with no user-facing summary.
    prefix = stripped[:start].strip()
    return prefix


class StreamSplitter:
    """Incremental <think>…</think> then answer. Blank-line fallback for tiny
    models. Also recognizes a bare JSON object (core/action_schema.py's
    structured-output mode) and pulls thinking/answer from its `think` and
    `rationale`/`finish.summary` fields instead of tag-splitting."""

    def __init__(self) -> None:
        self.buf = ""
        self._think_closed = False
        self._seen_open = False
        self._thinking = ""
        self._answer = ""
        self._is_thinking = True
        self._label = "Thinking"
        self._structured = False

    def reset(self) -> None:
        self.buf = ""
        self._think_closed = False
        self._seen_open = False
        self._thinking = ""
        self._answer = ""
        self._is_thinking = True
        self._label = "Thinking"
        self._structured = False

    def feed(self, token: str) -> StreamView:
        if token:
            self.buf += token
        return self.view()

    def view(self) -> StreamView:
        self._render()
        return StreamView(
            thinking=self._thinking,
            answer=self._answer,
            is_thinking=self._is_thinking,
            thinking_label=self._label,
        )

    def finalize(self) -> StreamView:
        self._render()
        self._is_thinking = False
        if self._seen_open or self._thinking:
            self._label = "Thought"
        if self._structured:
            # think/answer already came straight out of the parsed object's
            # own fields (_render) — no tag to strip, and no blank-line
            # split to run: a multi-paragraph `think` value is one field,
            # not thinking-plus-an-unlabeled-answer, so splitting it would
            # wrongly cannibalize the end of a legitimate think field.
            pass
        elif not (self._answer or "").strip() and (self._thinking or "").strip():
            think = self._thinking.strip()
            parts = [p for p in think.split("\n\n") if p.strip()]
            if len(parts) >= 2:
                self._thinking = "\n\n".join(parts[:-1]).strip()
                self._answer = strip_json_fence(parts[-1].strip())
            else:
                self._answer = strip_json_fence(think)
        else:
            self._answer = strip_json_fence(self._answer)
        return StreamView(
            thinking=self._thinking,
            answer=self._answer,
            is_thinking=False,
            thinking_label=self._label,
        )

    def _render(self) -> None:
        buf = self.buf
        stripped = buf.lstrip()
        if stripped.startswith("{"):
            obj = _try_parse_json_object(buf)
            if obj is not None:
                self._structured = True
                self._seen_open = True
                self._think_closed = True
                self._thinking = str(obj.get("think") or "").strip()
                self._is_thinking = False
                self._label = "Thought"
                self._answer = _structured_answer_text(obj)
                return
            # Still streaming an incomplete object — a live preview of the
            # think field if one has landed yet, otherwise just "thinking".
            self._structured = True
            self._seen_open = True
            self._think_closed = False
            self._is_thinking = True
            self._label = "Thinking"
            self._thinking = _live_think_preview(buf)
            self._answer = ""
            return

        open_idx, open_len = find_think_open(buf)
        close_idx = find_think_close(buf, open_idx + open_len) if open_idx >= 0 else -1

        if open_idx >= 0:
            self._seen_open = True
            after_open = open_idx + open_len
            if close_idx < 0:
                self._think_closed = False
                self._is_thinking = True
                self._label = "Thinking"
                self._thinking = buf[after_open:]
                self._answer = ""
                return
            self._think_closed = True
            close_len = 12 if buf[close_idx:].lower().startswith("</thinking>") else 8
            self._thinking = buf[after_open:close_idx].strip()
            self._is_thinking = False
            self._label = "Thought"
            after_close = close_idx + close_len
            self._answer = strip_json_fence(buf[after_close:].lstrip())
            return

        self._is_thinking = True
        self._label = "Thinking"
        split = buf.find("\n\n")
        if not self._think_closed and split > 40 and len(buf) - split > 12:
            self._seen_open = True
            self._think_closed = True
            self._thinking = buf[:split].strip()
            self._is_thinking = False
            self._label = "Thought"
            self._answer = strip_json_fence(buf[split + 2 :].lstrip())
        elif self._think_closed:
            prior = self._thinking or ""
            idx = buf.find(prior)
            if idx >= 0 and idx + len(prior) + 2 <= len(buf):
                self._answer = strip_json_fence(buf[idx + len(prior) :].lstrip("\n\r "))
            else:
                self._answer = strip_json_fence(buf)
            self._is_thinking = False
            self._label = "Thought"
        else:
            self._seen_open = True
            self._thinking = buf
            self._answer = ""
