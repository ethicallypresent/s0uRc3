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


@pytest.mark.asyncio
async def test_ctrl_c_is_a_noop_when_idle():
    """Nothing is running, so Ctrl+C must not open a steering prompt (and
    must not fall through to whatever Textual would otherwise do with it)."""
    from tui.simple_app import build_simple_app

    app = build_simple_app(dry_run=True)
    async with app.run_test() as pilot:
        assert app._busy is False
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._steering is False


@pytest.mark.asyncio
async def test_ctrl_c_while_busy_opens_steering_and_queues_on_submit():
    """End-to-end against the real AgentLoop.queue_steer, not a fake — the
    app's own dry-run bootstrap worker constructs a real AgentLoop shortly
    after mount regardless of dry_run (dry_run only affects loop.run() at
    call time), so a test that swaps in a fake loop right after mount can
    lose a race against that worker overwriting app.loop. Waiting for
    bootstrap first avoids that race and exercises the real integration."""
    from tui.simple_app import build_simple_app

    app = build_simple_app(dry_run=True)
    async with app.run_test() as pilot:
        for _ in range(50):
            if app.loop is not None:
                break
            await pilot.pause()
        assert app.loop is not None, "bootstrap never completed"
        app._busy = True

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._steering is True
        assert app._prompt.disabled is False

        for ch in "go check the other file":
            await pilot.press(ch if ch != " " else "space")
        await pilot.press("enter")
        await pilot.pause()

        assert app.loop._take_pending_steer() == "go check the other file"
        assert app._steering is False
        assert app._prompt.disabled is True  # run is still busy, so locked again
