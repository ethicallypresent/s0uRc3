"""Working memory (in-process) + long-term memory. LOCAL EDITION.

SQLite only — the Postgres path is gone. Embeddings come from your local
llama-server's /v1/embeddings endpoint when it's up, and fall back to the
deterministic bag-of-hash embedding when it isn't (dedup + rough recall
still work offline).

Env knobs (all optional):
    LLAMA_EMBED_URL    default http://127.0.0.1:8080/v1  (set to "" to force offline)
    LLAMA_EMBED_MODEL  default "" (llama-server ignores it and uses the loaded model)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger("agent.memory")

ALLOWED_KINDS = {"fact", "lesson", "preference", "failure", "constraint"}
MEMORY_STATUSES = {"proposal", "promoted", "stored"}
MAX_CONTENT_CHARS = 500  # "one dense sentence, not a paragraph" (constitution rule 16)
DEDUP_SCORE_THRESHOLD = 0.98  # near-identical embedding => treat as duplicate
MIN_RETRIEVE_SCORE = 0.05  # below this, a match isn't worth surfacing as relevant
HASH_EMBED_DIM = 64

_embed_warned = False


def _embed_url() -> str:
    return os.environ.get("LLAMA_EMBED_URL", "http://127.0.0.1:8080/v1").rstrip("/")


def _embed_model() -> str:
    return os.environ.get("LLAMA_EMBED_MODEL", "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_embedding(text: str, dim: int = HASH_EMBED_DIM) -> list[float]:
    """Deterministic bag-of-hash embedding so retrieve works offline."""
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _post_embedding(url: str, payload: dict, timeout: int) -> list[float] | None:
    """POST an embedding request; return the vector or None on any failure."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    # OpenAI shape: {"data": [{"embedding": [...]}]} | Ollama native: {"embedding": [...]}
    try:
        if isinstance(data.get("data"), list):
            return [float(x) for x in data["data"][0]["embedding"]]
        return [float(x) for x in data["embedding"]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def server_embedding(text: str, timeout: int = 20) -> list[float] | None:
    """Real semantic embedding from a local server. None if unreachable.

    Tries the OpenAI-compatible /v1/embeddings first, then Ollama's native
    /api/embeddings (older Ollama versions 404 the v1 path).

    timeout was 2s, which looked like a deliberate fast-fail for "no local
    server at all" — but confirmed empirically it also fails a real,
    reachable CPU-only llama-server under normal load (a handful of
    embedding calls in a row would intermittently time out and silently
    fall back to hash embeddings). A dead/refused connection still fails in
    milliseconds regardless of this value, so raising it only helps the
    case that actually needs it.
    """
    base = _embed_url()
    if not base:
        return None
    native_base = base[:-3] if base.endswith("/v1") else base
    model = _embed_model()

    attempts = [
        (f"{base}/embeddings", {"model": model, "input": text}),
        (f"{native_base}/api/embeddings", {"model": model, "prompt": text}),
    ]
    last_err: Exception | None = None
    for url, payload in attempts:
        try:
            vec = _post_embedding(url, payload, timeout)
        except Exception as exc:  # noqa: BLE001 - shouldn't happen, be safe
            last_err = exc
            continue
        if vec is not None:
            return vec
        last_err = RuntimeError(f"bad response from {url}")
    global _embed_warned
    if not _embed_warned:
        log.warning("local embeddings unavailable (%s); using offline hash embeddings", last_err)
        _embed_warned = True
    return None


def get_embedding_tagged(text: str) -> tuple[list[float], str]:
    """Return (vector, space). Spaces 'server' and 'hash' must never be mixed."""
    emb = server_embedding(text)
    if emb is not None:
        return emb, "server"
    return hash_embedding(text), "hash"


def get_embedding(text: str) -> list[float]:
    """Best available embedding: server first, hash fallback."""
    return get_embedding_tagged(text)[0]


def infer_embed_space(vec: list[float], stored: str | None = None) -> str:
    if stored in ("server", "hash"):
        return stored
    return "hash" if len(vec) == HASH_EMBED_DIM else "server"


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0  # different embedding spaces => not comparable
    return sum(x * y for x, y in zip(a, b))


@dataclass
class WorkingMemory:
    current_goal: str = ""
    active_plan: list[dict[str, Any]] = field(default_factory=list)
    scratchpad: list[str] = field(default_factory=list)
    step: int = 0
    last_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "active_plan": self.active_plan,
            "scratchpad": self.scratchpad[-12:],
            "step": self.step,
            "last_result": self.last_result,
        }

    def note(self, text: str) -> None:
        self.scratchpad.append(f"[{_now()}] {text}")


class LongTermMemory:
    """SQLite-backed store. No server, no drivers, one file."""

    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.sqlite = sqlite3.connect(self.db_dir / "memory.sqlite", check_same_thread=False)
        self.sqlite.row_factory = sqlite3.Row
        self._init_sqlite()
        log.info("LongTermMemory using SQLite at %s", self.db_dir / "memory.sqlite")

    def _init_sqlite(self) -> None:
        self.sqlite.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
              id TEXT PRIMARY KEY,
              user_id TEXT,
              timestamp TEXT,
              role TEXT,
              kind TEXT,
              content TEXT,
              verified INTEGER,
              embedding TEXT,
              meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
            """
        )
        cols = {row[1] for row in self.sqlite.execute("PRAGMA table_info(episodes)")}
        if "status" not in cols:
            self.sqlite.execute(
                "ALTER TABLE episodes ADD COLUMN status TEXT DEFAULT 'stored'"
            )
        if "embed_space" not in cols:
            self.sqlite.execute(
                "ALTER TABLE episodes ADD COLUMN embed_space TEXT DEFAULT 'hash'"
            )
        self.sqlite.commit()

    def _all_rows(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.sqlite.execute(
                "SELECT * FROM episodes WHERE user_id = ?", (user_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def _is_near_duplicate(self, content: str, embedding: list[float], user_id: str) -> bool:
        for row in self._all_rows(user_id):
            existing_emb = json.loads(row["embedding"])
            if cosine(embedding, existing_emb) >= DEDUP_SCORE_THRESHOLD:
                return True
        return False

    def save(
        self,
        content: str,
        *,
        role: str = "memory",
        kind: str = "lesson",
        user_id: str = "default",
        verified: bool = False,
        meta: dict | None = None,
        embedding: list[float] | None = None,
        skip_dedup: bool = False,
        status: str = "stored",
        embed_space: str | None = None,
    ) -> str:
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"invalid memory kind {kind!r}; must be one of {sorted(ALLOWED_KINDS)}")
        if status not in MEMORY_STATUSES:
            raise ValueError(f"invalid memory status {status!r}")

        content = content.strip()
        if not content:
            raise ValueError("cannot save empty memory content")
        if len(content) > MAX_CONTENT_CHARS:
            log.warning(
                "Memory content is %d chars (limit %d) — truncating. "
                "Save one dense sentence, not a transcript.",
                len(content), MAX_CONTENT_CHARS,
            )
            content = content[:MAX_CONTENT_CHARS].rstrip() + "…"

        if embedding is not None:
            emb = embedding
            space = infer_embed_space(emb, embed_space)
        else:
            emb, space = get_embedding_tagged(content)

        if not skip_dedup and self._is_near_duplicate(content, emb, user_id):
            log.info("Skipping near-duplicate memory: %r", content)
            return ""

        eid = str(uuid4())
        meta_obj = dict(meta or {})
        meta_obj.setdefault("status", status)
        row = (
            eid, user_id, _now(), role, kind, content, int(verified),
            json.dumps(emb), json.dumps(meta_obj), status, space,
        )
        try:
            with self._lock:
                self.sqlite.execute(
                    "INSERT INTO episodes "
                    "(id, user_id, timestamp, role, kind, content, verified, embedding, meta, status, embed_space) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
                self.sqlite.commit()
        except Exception as exc:
            log.error("Failed to persist memory (sqlite): %s", exc)
            raise
        return eid

    def promote(self, episode_id: str) -> bool:
        """Mark a proposal as long-term verified. Returns False if missing."""
        eid = (episode_id or "").strip()
        if not eid:
            return False
        with self._lock:
            cur = self.sqlite.execute(
                "UPDATE episodes SET verified = 1, status = 'promoted' WHERE id = ?",
                (eid,),
            )
            self.sqlite.commit()
            return cur.rowcount > 0

    def retrieve(self, query: str, k: int = 5, *, user_id: str = "default") -> list[dict[str, Any]]:
        q, q_space = get_embedding_tagged(query)
        rows = self._all_rows(user_id)
        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            space = infer_embed_space(emb, row.get("embed_space"))
            if space != q_space:
                continue  # different manifold — cosine would be a lie
            scored.append((cosine(q, emb), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, row in scored[:k]:
            if score < MIN_RETRIEVE_SCORE:
                break  # sorted descending, so nothing after this clears the bar either
            status = row.get("status") or (row.get("meta") and json.loads(row["meta"]).get("status")) or "stored"
            out.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "content": row["content"],
                    "verified": bool(row["verified"]),
                    "status": status if status in MEMORY_STATUSES else "stored",
                    "score": round(score, 4),
                }
            )
        return out

    def close(self) -> None:
        try:
            self.sqlite.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("Error closing memory store: %s", exc)


class Memory:
    def __init__(self, db_dir: Path, user_id: str = "default"):
        self.wm = WorkingMemory()
        self.ltm = LongTermMemory(db_dir)
        self.user_id = user_id

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self.ltm.retrieve(query, k=k, user_id=self.user_id)

    def save(self, content: str, **kwargs: Any) -> str:
        kwargs.setdefault("user_id", self.user_id)
        return self.ltm.save(content, **kwargs)

    def propose(self, content: str, **kwargs: Any) -> str:
        """Write a hypothesis. Not trusted as light until promote()."""
        kwargs.setdefault("user_id", self.user_id)
        kwargs["verified"] = False
        kwargs["status"] = "proposal"
        return self.ltm.save(content, **kwargs)

    def promote(self, episode_id: str) -> bool:
        return self.ltm.promote(episode_id)

    def close(self) -> None:
        self.ltm.close()
