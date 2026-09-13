"""Compose-smoke the simplified terminal GUI, and lock in the throttled
streaming contract that's the entire point of this module: tokens must not
touch the UI until a drain tick runs, and one drain must reflect everything
buffered since the last one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

textual = pytest.importorskip("textual")


@pytest.fixture(autouse=True)
def isolated_agent_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """build_simple_app() calls AgentPaths.discover() with no arguments, which
    resolves to the real repo root — without this, every test here would
    read/write the real db/chat_history.json (and did, before this fixture
    existed: a stray "Kurama" help-text message from this exact test suite
    ended up replayed in a live session as if it were part of that
    conversation)."""
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
async def test_simple_tui_composes_dry_run():
    from tui.simple_app import build_simple_app

    app = build_simple_app(dry_run=True)
    async with app.run_test() as pilot:
        assert app.query_one("#prompt")
        assert app.query_one("#chat")
        assert app.query_one("#status")
        chat = app.query_one("#chat")
        before = len(chat.lines)
        for ch in "/help":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()
        assert len(chat.lines) > before


@pytest.mark.asyncio
async def test_tokens_do_not_touch_the_live_widget_before_a_drain_tick():
    from tui.simple_app import build_simple_app

    app = build_simple_app(dry_run=True)
    async with app.run_test():
        live = app.query_one("#live")
        app._on_token("Hello")
        app._on_token(", ")
        app._on_token("world")
        # No call_from_thread happened in _on_token — nothing should have
        # reached the widget yet, however many tokens arrived.
        assert str(live.render()) == ""

        app._drain_tokens()
        # One tick reflects everything buffered since the last one, in one shot.
        assert "Hello, world" in str(live.render())


@pytest.mark.asyncio
async def test_drain_with_nothing_pending_is_a_cheap_noop():
    from tui.simple_app import build_simple_app

    app = build_simple_app(dry_run=True)
    async with app.run_test():
        live = app.query_one("#live")
        app._drain_tokens()  # nothing buffered
        assert str(live.render()) == ""
