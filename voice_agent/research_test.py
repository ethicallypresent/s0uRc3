"""RESEARCH — standalone test for the Wikipedia lookup loop.py uses.

Type a query, see what Wikipedia article (if any) loop.py would pull in as
background context before BRAIN replies. No mic, no TTS, no LLM involved —
pure network+parsing test, so you can sanity-check this piece on its own
before trusting it inside the full voice loop.

Usage:
  python research_test.py
  python research_test.py "when was the eiffel tower built"
"""
from __future__ import annotations

import sys

from research import research


def _run(query: str) -> None:
    result = research(query)
    if not result.get("ok"):
        print(f"[no result: {result.get('error')}]\n")
        return
    print(f"-- {result['title']} --")
    print(result["extract"])
    print(f"({result['url']})\n")


def main() -> int:
    query = " ".join(sys.argv[1:]).strip()
    if query:
        _run(query)
        return 0

    print("Type a query to research, 'exit' to quit.\n")
    try:
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            _run(query)
    finally:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
