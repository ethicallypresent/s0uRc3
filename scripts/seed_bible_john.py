#!/usr/bin/env python3
"""Seed core content of the Gospel of John into long-term memory.

Same mechanism as scripts/seed_bible_fundamentals.py — kept as a separate
seed (own file, own list) rather than merged into the general fundamentals
set, per request. See that script's docstring for why this goes through
memory rather than the system prompt.

Usage:
  python scripts/seed_bible_john.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.memory import Memory
from core.paths import AgentPaths

JOHN_FACTS: list[str] = [
    "John 1:1-14: in the beginning was the Word, the Word was with God and was God, and the Word became flesh and dwelt among us.",
    "John 1:29: John the Baptist calls Jesus 'the Lamb of God who takes away the sin of the world.'",
    "John 3:16: for God so loved the world that he gave his one and only Son, so that whoever believes in him will not perish but have eternal life.",
    "John's seven signs: water into wine at Cana (2:1-11), healing an official's son (4:46-54), healing a paralytic at Bethesda (5:1-15), feeding the 5,000 (6:1-14), walking on water (6:16-21), healing a man born blind (9:1-7), and raising Lazarus from the dead (11:1-44).",
    "John's seven 'I am' statements by Jesus: the bread of life (6:35), the light of the world (8:12), the gate (10:9), the good shepherd (10:11), the resurrection and the life (11:25), the way the truth and the life (14:6), and the true vine (15:1).",
    "John 13: at the Last Supper, Jesus washes his disciples' feet and gives a new commandment: love one another as I have loved you.",
    "John 14:6: Jesus said, 'I am the way and the truth and the life; no one comes to the Father except through me.'",
    "John 14-16: Jesus promises his disciples the Holy Spirit, the Advocate, who will remain with them after he is gone.",
    "John 18-19: Jesus is arrested, tried before Pilate, and crucified.",
    "John 20: Jesus rises from the dead and appears to Mary Magdalene and then to the disciples, including Thomas, who had doubted.",
    "John 20:31: the book's stated purpose is that readers 'may believe that Jesus is the Messiah, the Son of God, and that by believing you may have life in his name.'",
    "John 21: in the epilogue, the risen Jesus appears at the Sea of Galilee and restores Peter, telling him three times to 'feed my sheep.'",
]


def main() -> None:
    paths = AgentPaths.discover()
    memory = Memory(paths.db)
    saved = 0
    skipped = 0
    try:
        for text in JOHN_FACTS:
            eid = memory.save(text, kind="fact", verified=True)
            if eid:
                saved += 1
            else:
                skipped += 1
    finally:
        memory.close()
    print(f"Seeded {saved} memory entries from the Book of John ({skipped} skipped as near-duplicates).")


if __name__ == "__main__":
    main()
