"""Standalone chat agent — no tools, no JSON action protocol, just conversation
that recalls and saves long-term memory.

Talks to the same local llama-server as s0uRc3's AgentLoop (core/loop.py),
and shares the same long-term memory store (core/memory.py, db/memory.sqlite)
so it can recall lessons/facts/preferences from past chats or from s0uRc3's
own runs (including the biblical fundamentals seeded via scripts/seed_bible_*.py)
— but it skips everything else on purpose: no beliefs, no evolution, no tool
registry, no <think>/json action parsing. That whole action-protocol layer
is exactly what a model this small struggles with (see this project's own
debugging history) — plain chat has none of that surface area to fail on.

Memory is intentionally one-directional and cheap, not a second slow LLM
call every turn (this model is minutes-per-call on CPU):
  - recall: every turn, the top few memories relevant to what the user just
    said are pulled in as brief context (no extra completion needed).
  - save: only when the user's own words ask for it ("remember that ...",
    "note that ...", "don't forget ...") — matches this project's own
    "save only what would change a future plan" memory philosophy, and
    keeps chat responsive instead of doubling every turn's latency.

Usage:
  python chat_agent.py
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


_CHAT_RELEVANT_KINDS = {"fact", "preference"}
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _keyword_overlap(query: str, content: str) -> float:
    q_words = set(_WORD_RE.findall(query.lower()))
    if not q_words:
        return 0.0
    c_words = set(_WORD_RE.findall(content.lower()))
    return len(q_words & c_words) / len(q_words)


def _recall_block(memory: Memory, query: str) -> str:
    """Top few memories relevant to what the user just said — no extra
    completion needed, just an embedding lookup against the same store
    s0uRc3's own runs write to.

    Restricted to fact/preference kinds: "lesson"/"constraint"/"failure"
    entries are specifically s0uRc3's own tool-operating guidance (e.g.
    "Matthew 25 parallels create_skill"), not conversational knowledge — a
    weak model asked a plain question and handed an off-topic tool lesson
    in its context will garble the two together (confirmed: it worked a
    tool-parallel entry into an answer about the Book of John and invented
    a false claim from the collision).

    Also re-ranked with a keyword-overlap bonus on top of the embedding
    score. Confirmed empirically that even real server embeddings from a
    1.2B causal model (mean-pooled, not a model trained for embeddings)
    barely discriminate topics — cosine scores for "what do you know about
    the book of john" clustered in a narrow 0.74-0.80 band with no clear
    separation, and an unrelated "my favorite color is teal" outscored the
    actual John content. An exact word match like "john" is a far stronger
    signal here than this embedding space provides alone — same idea as
    core/tool_registry.py's own keyword-plus-score ranking.
    """
    try:
        # A large k, not RECALL_K*4: retrieve() already scores every row by
        # cosine similarity internally regardless of k (k only truncates the
        # returned list), so this costs nothing extra — but a small k was
        # cutting the pool down to embedding-score order *before* the
        # keyword re-rank below ever got a chance to run, silently dropping
        # exactly the entries keyword overlap was meant to rescue.
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
    """Save a memory only when the user's own words ask for it — deliberate,
    not automatic logging of every exchange."""
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
    print("Chat agent ready (no tools — recalls and saves memory, just conversation).")
    print("Say 'remember that ...' to save something. Type 'exit' to quit.\n")
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

            # Recall is fresh every turn and never persisted into `messages`
            # — it reflects only the current question, not an accumulating
            # pile of past recalls bloating every future call.
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

            # Keep the system prompt plus the most recent turns only — this is
            # deliberately a flat window, not the recall/summarization machinery
            # AgentLoop has, to stay true to "no tools, just chat".
            if len(messages) > 1 + MAX_TURNS_KEPT * 2:
                messages[:] = [messages[0]] + messages[-MAX_TURNS_KEPT * 2 :]
    finally:
        memory.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
