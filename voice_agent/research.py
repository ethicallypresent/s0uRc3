"""RESEARCH — Wikipedia lookup used by loop.py to ground replies in facts.

No API key: Wikipedia's search + page-summary REST endpoints are free and
keyless. (An earlier attempt at DuckDuckGo HTML scraping was dropped after
testing confirmed it's blocked by an anti-bot challenge for any plain HTTP
request, no JS involved — not usable headless at all.)

This module is the network+parsing layer only. loop.py decides *when* to
call it (see classify_needs_research there) and what to do with the result;
this stays a pure "query in, Wikipedia summary out" function so it can be
tested on its own — see research_test.py.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "s0uRc3-voice-agent/1.0 (https://github.com/ethicallypresent/s0uRc3)"
SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
TIMEOUT = 10


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_titles(query: str, limit: int = 3) -> list[str]:
    params = urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": str(limit), "format": "json"}
    )
    data = _get_json(f"{SEARCH_URL}?{params}")
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def page_summary(title: str) -> dict[str, Any] | None:
    """None means "skip this title" — missing page or a disambiguation page
    (a list of unrelated articles sharing a name, not useful as context)."""
    url = SUMMARY_URL.format(urllib.parse.quote(title.replace(" ", "_")))
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if data.get("type") == "disambiguation":
        return None
    return {
        "title": data.get("title", title),
        "extract": data.get("extract", ""),
        "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
    }


def research(query: str, max_chars: int = 700) -> dict[str, Any]:
    """Look up `query` on Wikipedia. Returns {"ok": True, "title", "extract",
    "url"} or {"ok": False, "error"}. Never raises — a lookup failure should
    never crash a voice turn, just skip the extra context."""
    try:
        titles = search_titles(query)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"ok": False, "error": f"search failed: {exc}"}
    if not titles:
        return {"ok": False, "error": "no results"}
    for title in titles:
        try:
            summary = page_summary(title)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return {"ok": False, "error": f"summary failed: {exc}"}
        if summary is not None:
            summary["extract"] = summary["extract"][:max_chars]
            return {"ok": True, **summary}
    return {"ok": False, "error": "no usable summary (all results were disambiguation pages)"}
