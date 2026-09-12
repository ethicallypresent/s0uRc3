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
