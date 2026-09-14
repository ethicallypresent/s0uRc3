#!/usr/bin/env python3
"""Seed core biblical fundamentals into long-term memory.

Mirrors scripts/train_from_session.py's mechanism: this is the "agent
learning layer" (core/memory.py), not a prompt change — these facts get
recalled into context only when relevant to a goal (gather_context's
memory search), not injected into every turn like the system prompt
would be. Keeping this out of brain/system_prompt.md and constitution.md
is deliberate: this session confirmed pocket.gguf destabilizes fast when
given more to hold in context at once, and those files are already at the
minimum needed for reliable tool use.

Two kinds of entries:
  - "fact": the core teachings themselves, stated plainly.
  - "lesson": places where a teaching maps onto a real operating principle
    already in brain/constitution.md — recorded only where the parallel is
    genuine (e.g. honesty, humility about uncertainty), not manufactured
    for every entry.

Usage:
  python scripts/seed_bible_fundamentals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.memory import Memory
from core.paths import AgentPaths

FACTS: list[str] = [
    "Genesis 1-2: God created the heavens, the earth, and all living things, and called it good.",
    "Genesis 3: humanity's disobedience (the Fall) brought sin and separation from God into the world.",
    "Exodus 20: God gave Moses the Ten Commandments as the moral law of the covenant with Israel.",
    "Mark 12:30-31: the greatest commandment is to love God fully, and the second is to love your neighbor as yourself.",
    "Matthew 7:12 (the Golden Rule): do to others what you would have them do to you.",
    "The Gospels: Jesus Christ, God's son, lived, taught, was crucified, and rose from the dead.",
    "Ephesians 2:8-9: salvation is by God's grace through faith, not earned by good works.",
    "Galatians 5:22-23: the fruit of the Spirit is love, joy, peace, patience, kindness, goodness, faithfulness, gentleness, and self-control.",
    "Matthew 28:19 (the Great Commission): go and make disciples of all nations.",
    "2 Timothy 3:16: scripture is useful for teaching, rebuking, correcting, and training in righteousness.",
]

LESSONS: list[str] = [
    "Exodus 20:16 (do not bear false witness) parallels this agent's own honesty rule: never invent tool output, facts, or task outcomes.",
    "Proverbs 3:5 (do not lean on your own understanding) parallels labeling claims observed/inferred/speculative instead of presenting a guess as settled fact.",
    "Exodus 20:15 (do not steal) parallels respecting protected/writable path boundaries rather than treating every path as available.",
    "James 1:19 (be slow to speak, quick to listen) parallels pulling context (memory, tool results) before acting instead of guessing first.",
]


def main() -> None:
    paths = AgentPaths.discover()
    memory = Memory(paths.db)
    saved = 0
    skipped = 0
    try:
        for text in FACTS:
            eid = memory.save(text, kind="fact", verified=True)
            if eid:
                saved += 1
            else:
                skipped += 1
        for text in LESSONS:
            eid = memory.save(text, kind="lesson", verified=True)
            if eid:
                saved += 1
            else:
                skipped += 1
    finally:
        memory.close()
    print(f"Seeded {saved} memory entries ({skipped} skipped as near-duplicates of existing entries).")


if __name__ == "__main__":
    main()
