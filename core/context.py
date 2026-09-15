"""Turn context: memory, beliefs, skills, tools, evolution, named entities."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.evolution import EvolutionLog
from core.memory import Memory
from core.paths import AgentPaths
from core.reflection import BeliefStore
from core.skill_manager import SkillManager
from core.tool_registry import ToolRegistry
from core.world_model import WorldModel

log = logging.getLogger("agent.context")


def _slim_memory_hits(hits: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for h in hits:
        key = f"{h.get('kind')}|{(h.get('content') or '')[:80]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "kind": h.get("kind"),
                "content": (h.get("content") or "")[:240],
                "verified": bool(h.get("verified")),
                "status": h.get("status") or "stored",
                "score": h.get("score"),
            }
        )
        if len(out) >= limit:
            break
    return out


def gather_context(
    paths: AgentPaths | None = None,
    *,
    query: str,
    memory_topk: int = 8,
) -> dict[str, Any]:
    """Pull memory, beliefs, skills, and tools into one compact dict."""
    paths = paths or AgentPaths.discover()
    q = (query or "").strip() or "general preferences lessons constraints"

    memory = Memory(paths.db)
    world_model = WorldModel(paths.db)
    tools = ToolRegistry(paths)
    skills = SkillManager(paths)
    evolution = EvolutionLog(paths.db)
    beliefs = BeliefStore(paths.db)

    try:
        queries = [
            q,
            f"{q} lesson preference constraint",
            "failure what went wrong",
            "coding workspace write_file skill",
            "user preference how to respond",
        ]
        pooled: list[dict[str, Any]] = []
        for item in queries:
            try:
                pooled.extend(memory.retrieve(item, k=max(3, memory_topk // 2)))
            except Exception as exc:  # noqa: BLE001
                log.warning("memory query %r failed: %s", item[:40], exc)

        recalled = _slim_memory_hits(pooled, limit=memory_topk)

        belief_snap = {
            k: {
                "p_success": round(float(v.p_success), 3),
                "observations": int(v.observations),
            }
            for k, v in sorted(beliefs.beliefs.items(), key=lambda kv: -kv[1].p_success)[:12]
        }

        skill_rows = skills.list_skills(q, k=8, include_drafts=True)
        slim_skills = [
            {
                "name": s.get("name"),
                "description": (s.get("description") or "")[:120],
                "kind": s.get("kind"),
                "status": s.get("status") or ("draft" if s.get("draft") else "approved"),
            }
            for s in skill_rows
            if isinstance(s, dict) and s.get("name")
        ]

        tool_rows = tools.list_filtered(q, k=10)
        slim_tools = []
        seen_t: set[str] = set()
        for t in tool_rows:
            n = str(t.get("name") or "")
            if not n or n in seen_t:
                continue
            seen_t.add(n)
            slim_tools.append({"name": n, "description": (t.get("description") or "")[:120]})

        last_trace_summary = None
        lt = paths.db / "last_trace.json"
        if lt.exists():
            try:
                raw = json.loads(lt.read_text(encoding="utf-8"))
                last_trace_summary = {
                    "goal": (raw.get("goal") or "")[:160],
                    "ok": raw.get("ok"),
                    "rewarded": raw.get("rewarded"),
                    "steps": len(raw.get("trace") or []),
                }
            except (OSError, json.JSONDecodeError):
                pass

        brain_files = {
            "system_prompt": (paths.brain / "system_prompt.md").exists(),
            "constitution": (paths.brain / "constitution.md").exists(),
            "system_prompt_full": (paths.brain / "system_prompt.full.md").exists(),
            "constitution_full": (paths.brain / "constitution.full.md").exists(),
            "reasoning_config": (paths.brain / "reasoning_config.json").exists(),
        }

        return {
            "note": (
                "Before answering, reference recalled memory, beliefs, skills, "
                "and tools. Cite what you use in your reasoning."
            ),
            "query": q,
            "brain_files": brain_files,
            "memory": {
                "working": memory.wm.to_dict(),
                "recalled": recalled,
                "recall_count": len(recalled),
            },
            "beliefs": belief_snap,
            "skills": slim_skills,
            "tools": slim_tools,
            "evolution": {
                "leaderboard": evolution.leaderboard()[:8],
                "gaps": (evolution.gaps[-5:] if getattr(evolution, "gaps", None) else []),
            },
            "last_trace": last_trace_summary,
            "named_entities": world_model.recall(q, k=8),
        }
    finally:
        memory.close()
        world_model.close()


def gather_context_json(query: str, *, memory_topk: int = 8) -> str:
    box = gather_context(query=query, memory_topk=memory_topk)
    return json.dumps(box, ensure_ascii=False, separators=(",", ":"), default=str)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Dump agent context as JSON")
    ap.add_argument("query", nargs="?", default="general")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.write(gather_context_json(args.query, memory_topk=args.topk))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
