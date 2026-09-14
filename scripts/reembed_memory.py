#!/usr/bin/env python3
"""Re-embed every existing memory row with the live embedding backend.

Needed once, right after fixing the llama-server launch args to actually
expose /v1/embeddings (core/llama_server.py: --embeddings --pooling mean).
Every row saved before that fix is tagged embed_space="hash" (the offline
fallback), and Memory.retrieve() skips any row whose embed_space differs
from the current query's space ("different manifold — cosine would be a
lie") — so without this, all existing memories (including everything
seeded by scripts/seed_bible_*.py) would silently stop being recallable
the moment real embeddings came online, not because they were deleted but
because they'd never match again.

Requires the llama-server to be up and actually serving /v1/embeddings
(run core/llama_server.ensure_from_config first, or just have main.py/
chat_agent.py running).

Usage:
  python scripts/reembed_memory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.memory import get_embedding_tagged
from core.paths import AgentPaths


def main() -> int:
    paths = AgentPaths.discover()
    from core.memory import Memory

    memory = Memory(paths.db)
    ltm = memory.ltm
    try:
        rows = ltm.sqlite.execute("SELECT id, content, embed_space FROM episodes").fetchall()
        total = len(rows)
        updated = 0
        already = 0
        for row in rows:
            if row["embed_space"] == "server":
                already += 1
                continue
            emb, space = get_embedding_tagged(row["content"])
            if space != "server":
                # Server still unreachable for this call — stop rather than
                # half-migrate the table into a mixed, confusing state.
                print(
                    f"Embedding backend not returning 'server' space (got {space!r}) — "
                    "is the llama-server up with --embeddings? Stopping without changes "
                    f"beyond the {updated} rows already migrated this run."
                )
                return 1
            ltm.sqlite.execute(
                "UPDATE episodes SET embedding = ?, embed_space = ? WHERE id = ?",
                (json.dumps(emb), space, row["id"]),
            )
            updated += 1
        ltm.sqlite.commit()
        print(f"Re-embedded {updated}/{total} rows ({already} already on 'server' space).")
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    sys.exit(main())
