#!/usr/bin/env python3
"""Seed biblical parallels for the agent's own tools into long-term memory.

Same mechanism as the other seed_bible_*.py scripts. Each entry names a
real tool from core/tool_registry.py's TOOL_DOMAINS and a genuine
scriptural parallel to what that tool actually does — not one entry per
tool regardless of fit. A few tools (project_status, check_axiom) don't
get their own entry because advance_axiom/init_project/log_evidence
already cover the CreatorCentral gate theme without repeating it.

Usage:
  python scripts/seed_bible_tools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.memory import Memory
from core.paths import AgentPaths

TOOL_LESSONS: list[str] = [
    "Matthew 7:7 ('seek and you will find') parallels read_file: seek out what is already written before acting on it, rather than guessing at its contents.",
    "Exodus 34:27 (God tells Moses 'write down these words') parallels write_file: it is how the agent records what has actually been decided or produced, not a draft left unrecorded.",
    "Nehemiah 2:12-15 (Nehemiah surveys Jerusalem's walls by night before rebuilding) parallels list_dir: survey what already exists before acting, rather than building on an assumption.",
    "Proverbs 11:14 ('in the multitude of counselors there is safety') parallels web_search and fetch_url: gather outside counsel before relying on the agent's own knowledge alone.",
    "Ecclesiastes 9:10 ('whatever your hand finds to do, do it with all your might') parallels run_python: it is the agent's own hand doing the work, not merely describing it.",
    "Matthew 25:14-30 (the parable of the talents) parallels create_skill and run_skill: a proven skill is a capability put to use and multiplied through reuse, not buried and left idle.",
    "Luke 14:28 ('count the cost before building a tower') parallels init_project and advance_axiom: a build only advances once its requirements are actually met, not started or pushed forward carelessly.",
    "Deuteronomy 19:15 ('a matter established by the testimony of two or three witnesses') parallels log_evidence: it is how a claim becomes evidence attached to a requirement, not just an assertion.",
    "Genesis 2:2 (God rested on the seventh day, the work of creation finished) parallels close_project: closing a project honestly as done, versus closed_early, mirrors recognizing when work is actually finished.",
    "Luke 14:31 (a king weighs whether he can win a battle before going to war) parallels request_confirmation: pause and ask before an irreversible action instead of committing to it unilaterally.",
]


def main() -> None:
    paths = AgentPaths.discover()
    memory = Memory(paths.db)
    saved = 0
    skipped = 0
    try:
        for text in TOOL_LESSONS:
            eid = memory.save(text, kind="lesson", verified=True)
            if eid:
                saved += 1
            else:
                skipped += 1
    finally:
        memory.close()
    print(f"Seeded {saved} tool-lesson memory entries ({skipped} skipped as near-duplicates).")


if __name__ == "__main__":
    main()
