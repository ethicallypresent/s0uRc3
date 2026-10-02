"""s0uRc3 — minimal local agent.

Talks to a local llama-server (started automatically from
brain/reasoning_config.json) and recalls/saves long-term memory in
db/memory.sqlite. No tools, no TUI, no JSON action protocol — plain
conversation, on purpose: see README.md.

Usage:
  python main.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

from core.config import load as load_config
from core.llama_server import ensure_from_config
from core.memory import Memory
from core.paths import AgentPaths

SYSTEM_PROMPT = "You are a friendly, direct conversational assistant. Keep replies concise."
MAX_TURNS_KEPT = 20  # user+assistant pairs; trims the oldest once exceeded
RECALL_K = 4

_REMEMBER_RE = re.compile(
    r"(?:remember|note|keep in mind|don'?t forget)(?:\s+that)?\s*[:,]?\s+(.+)",
    re.IGNORECASE,
)
_CHAT_RELEVANT_KINDS = {"fact", "preference"}
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _chat_once(
    base_url: str, api_key: str, model: str, messages: list[dict[str, str]], timeout: int
) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 500}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _keyword_overlap(query: str, content: str) -> float:
    q_words = set(_WORD_RE.findall(query.lower()))
    if not q_words:
        return 0.0
    c_words = set(_WORD_RE.findall(content.lower()))
    return len(q_words & c_words) / len(q_words)


def _recall_block(memory: Memory, query: str) -> str:
    """Top few memories relevant to what the user just said, re-ranked with a
    keyword-overlap bonus on top of the embedding score (the embedding score
    alone barely discriminates topics on this project's small local model)."""
    try:
        hits = memory.retrieve(query, k=200)
    except Exception:  # noqa: BLE001 — recall is a nice-to-have, never fatal
        return ""
    hits = [h for h in hits if h.get("kind") in _CHAT_RELEVANT_KINDS]
    if not hits:
        return ""
    for h in hits:
        h["_rank"] = float(h.get("score") or 0.0) + _keyword_overlap(query, h.get("content") or "")
    hits.sort(key=lambda h: h["_rank"], reverse=True)
    hits = hits[:RECALL_K]
    lines = [f"- {h.get('content')}" for h in hits]
    return (
        "Facts from memory relevant to the user's message. If they answer the "
        "question, use them directly instead of your own general knowledge:\n"
        + "\n".join(lines)
    )


def _maybe_save_from_user(memory: Memory, user_input: str) -> str | None:
    """Save a memory only when the user's own words ask for it."""
    m = _REMEMBER_RE.search(user_input)
    if not m:
        return None
    text = m.group(1).strip().rstrip(".").strip()
    if not text:
        return None
    try:
        eid = memory.save(text, kind="fact", verified=True)
    except ValueError:
        return None
    return text if eid else None


def main() -> int:
    paths = AgentPaths.discover()
    cfg = load_config(paths)
    boot = ensure_from_config(paths.root, cfg)
    if not boot.get("ok"):
        print(f"Could not start/reach the model server: {boot.get('error')}", file=sys.stderr)
        return 1

    llm = cfg.get("llm") or {}
    base_url = str(llm.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")
    api_key = str(llm.get("api_key", "sk-no-key-needed"))
    model = str(llm.get("model", "pocket"))
    timeout = int(llm.get("timeout_sec", 300))

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    memory = Memory(paths.db)
    print("s0uRc3 ready. Say 'remember that ...' to save something. Type 'exit' to quit.\n")
    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            saved = _maybe_save_from_user(memory, user_input)
            if saved:
                print(f"[remembered: {saved}]")

            call_messages = list(messages)
            recall = _recall_block(memory, user_input)
            if recall:
                call_messages.insert(1, {"role": "system", "content": recall})
            call_messages.append({"role": "user", "content": user_input})

            try:
                reply = _chat_once(base_url, api_key, model, call_messages, timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"[error: {exc}]\n")
                continue

            reply = reply.strip() or "(no reply)"
            print(f"agent> {reply}\n")
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": reply})

            if len(messages) > 1 + MAX_TURNS_KEPT * 2:
                messages[:] = [messages[0]] + messages[-MAX_TURNS_KEPT * 2 :]
    finally:
        memory.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
