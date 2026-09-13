"""Ctrl+C interrupt/steer: a human mid-run course-correction must reach the
model as this step's perception.user_input, be visibly flagged as an
interrupt (not confused with the routine step-1 goal echo), and never be
lost or duplicated across steps.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.loop import AgentLoop, build_user_packet
from core.memory import Memory
from core.paths import AgentPaths


def _fake_loop() -> SimpleNamespace:
    return SimpleNamespace(_steer_lock=threading.Lock(), _pending_steer="")


def test_queue_steer_then_take_returns_it_once():
    loop = _fake_loop()
    queued = AgentLoop.queue_steer(loop, "  focus on the auth bug instead  ")
    assert queued == "focus on the auth bug instead"
    assert AgentLoop._take_pending_steer(loop) == "focus on the auth bug instead"
    assert AgentLoop._take_pending_steer(loop) == ""  # consumed, not re-delivered


def test_queue_steer_ignores_blank_text():
    loop = _fake_loop()
    assert AgentLoop.queue_steer(loop, "   ") == ""
    assert AgentLoop._take_pending_steer(loop) == ""


def test_queue_steer_overwrites_a_still_pending_one():
    """If two interrupts arrive before the loop reaches a step boundary,
    the latest one wins rather than queueing up a backlog."""
    loop = _fake_loop()
    AgentLoop.queue_steer(loop, "first")
    AgentLoop.queue_steer(loop, "second")
    assert AgentLoop._take_pending_steer(loop) == "second"


@pytest.fixture
def agent_root(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    for sub in ("brain", "core", "db", "skills", "tools", "workspace"):
        (root / sub).mkdir(parents=True)
    (root / "brain" / "constitution.md").write_text("# test constitution\n")
    (root / "brain" / "system_prompt.md").write_text("# test system prompt\n")
    return root


def _packet_obj(raw_packet: str) -> dict:
    return json.loads(raw_packet.split("PACKET:\n", 1)[1])


def test_step_one_goal_echo_is_not_flagged_as_an_interrupt(agent_root: Path):
    paths = AgentPaths.discover(start=agent_root)
    memory = Memory(paths.db)
    try:
        packet = build_user_packet(
            goal="do the thing",
            user_input="do the thing",
            memory=memory,
            tools=[],
            skills=[],
            step=1,
            paths=paths,
        )
    finally:
        memory.close()
    assert "just interrupted mid-task" not in packet
    assert _packet_obj(packet)["perception"]["user_input"] == "do the thing"


def test_mid_run_steer_is_flagged_as_an_interrupt_and_reaches_the_model(agent_root: Path):
    paths = AgentPaths.discover(start=agent_root)
    memory = Memory(paths.db)
    try:
        packet = build_user_packet(
            goal="original goal",
            user_input="actually, check the other file first",
            memory=memory,
            tools=[],
            skills=[],
            step=3,
            paths=paths,
        )
    finally:
        memory.close()
    assert "just interrupted mid-task" in packet
    obj = _packet_obj(packet)
    assert obj["perception"]["user_input"] == "actually, check the other file first"
    assert obj["user_goal"] == "original goal"  # original goal is preserved, not replaced


def test_step_beyond_one_with_no_steer_is_unflagged(agent_root: Path):
    paths = AgentPaths.discover(start=agent_root)
    memory = Memory(paths.db)
    try:
        packet = build_user_packet(
            goal="original goal",
            user_input="",
            memory=memory,
            tools=[],
            skills=[],
            step=2,
            paths=paths,
        )
    finally:
        memory.close()
    assert "just interrupted mid-task" not in packet
