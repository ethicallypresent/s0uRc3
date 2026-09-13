"""Compose-smoke the terminal GUI when textual is installed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

textual = pytest.importorskip("textual")


@pytest.fixture(autouse=True)
def isolated_agent_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """build_app() calls AgentPaths.discover() with no arguments, which
    resolves to the real repo root — without this, this test writes into the
    real db/chat_history.json every run (confirmed: it's how a stray
    "Kurama" /help message ended up replayed in a live user session)."""
    root = tmp_path / "agent"
    for sub in ("brain", "core", "db", "skills", "tools", "workspace"):
        (root / sub).mkdir(parents=True)
    (root / "brain" / "constitution.md").write_text("# test constitution\n")
    (root / "brain" / "system_prompt.md").write_text("# test system prompt\n")
    (root / "brain" / "reasoning_config.json").write_text(
        '{"model": "gpt-4o", "max_steps": 5, "skill_topk": 8, "tool_topk": 8, "memory_topk": 5}'
    )
    from core.paths import AgentPaths

    real_discover = AgentPaths.discover.__func__
    monkeypatch.setattr(AgentPaths, "discover", classmethod(lambda cls, *a, **kw: real_discover(cls, start=root)))
    return root


@pytest.mark.asyncio
async def test_tui_composes_dry_run():
    from tui.app import build_app

    app = build_app(dry_run=True)
    async with app.run_test() as pilot:
        assert app.query_one("#prompt")
        assert app.query_one("#chat")
        assert app.query_one("#status")
        await pilot.press("/")
        await pilot.press("h")
        await pilot.press("e")
        await pilot.press("l")
        await pilot.press("p")
        await pilot.press("enter")
        await pilot.pause()
        chat = app.query_one("#chat")
        assert len(list(chat.children)) >= 2
