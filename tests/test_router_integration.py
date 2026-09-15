"""The router's tier must actually reach the model and the loop's own
quality checks, not just exist as a classification. Covers: build_user_packet
carries reasoning_budget and the matching wrapped instruction per tier
(including how that composes with the Ctrl+C steer interrupt note), and
_check_think treats an empty think block as compliant (not weak) for act.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.loop import AgentLoop, build_user_packet
from core.memory import Memory
from core.paths import AgentPaths
from core.router import ACT, DELIBERATE, GLANCE


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


def _build(
    agent_root: Path, tier: str, *, step: int = 1, user_input: str = "read main.py", structured: bool = False,
) -> str:
    paths = AgentPaths.discover(start=agent_root)
    memory = Memory(paths.db)
    try:
        return build_user_packet(
            goal="read main.py", user_input=user_input, memory=memory,
            tools=[], skills=[], step=step, paths=paths, tier=tier, structured=structured,
        )
    finally:
        memory.close()


def test_act_packet_says_skip_think_entirely(agent_root: Path):
    packet = _build(agent_root, ACT)
    assert "skip <think> entirely" in packet
    assert _packet_obj(packet)["reasoning_budget"].startswith("reasoning_budget: act")


def test_glance_packet_caps_reasoning_at_about_fifty_tokens(agent_root: Path):
    packet = _build(agent_root, GLANCE)
    assert "~50 tokens" in packet
    assert _packet_obj(packet)["reasoning_budget"].startswith("reasoning_budget: glance")


def test_deliberate_packet_keeps_the_full_reasoning_instruction(agent_root: Path):
    packet = _build(agent_root, DELIBERATE)
    assert "observed / inferred / speculative" in packet
    assert _packet_obj(packet)["reasoning_budget"].startswith("reasoning_budget: deliberate")


def test_deliberate_tier_still_carries_the_human_interrupt_note(agent_root: Path):
    """A live Ctrl+C steer always escalates to deliberate (core/router.py),
    so the two must actually compose: the interrupt note must not have been
    dropped when the tier-specific wrapped text was added."""
    packet = _build(agent_root, DELIBERATE, step=3, user_input="actually, check the other file first")
    assert "just interrupted mid-task" in packet
    assert "reasoning_budget: deliberate" in packet


def test_structured_act_packet_has_no_think_tag_phrasing(agent_root: Path):
    """core/action_schema.py folds think into the response schema — the
    packet must never tell a structured-mode model to use <think> tags,
    since the grammar constrains the whole completion to one JSON object."""
    packet = _build(agent_root, ACT, structured=True)
    assert "<think>" not in packet
    assert "leave the think field empty" in packet.lower()


def test_structured_glance_packet_has_no_think_tag_phrasing(agent_root: Path):
    packet = _build(agent_root, GLANCE, structured=True)
    assert "<think>" not in packet
    assert "~50 tokens" in packet


def test_structured_deliberate_packet_has_no_think_tag_phrasing(agent_root: Path):
    packet = _build(agent_root, DELIBERATE, structured=True)
    assert "<think>" not in packet
    assert "observed / inferred / speculative" in packet


def test_structured_packet_never_carries_the_anti_echo_warning(agent_root: Path):
    """Only meaningful for the free-text path — a structured completion
    literally cannot echo the packet back and still satisfy the schema."""
    packet = _build(agent_root, DELIBERATE, structured=True)
    assert "never write the packet below" not in packet.lower()


def test_no_tier_defaults_to_deliberate_not_silently_understimulated():
    """A caller that forgets to pass tier (or an older code path) must not
    silently fall into a truncated act/glance budget."""
    import inspect

    sig = inspect.signature(build_user_packet)
    assert sig.parameters["tier"].default == DELIBERATE


def test_check_think_treats_empty_act_think_as_ok_not_weak():
    fake_self = SimpleNamespace(cfg={}, _weak_think=None)
    verdict = AgentLoop._check_think(fake_self, "", dry_run=False, tier=ACT)
    assert verdict["ok"] is True
    assert fake_self._weak_think is None  # untouched — act never flips weak_think


def test_check_think_still_flags_a_genuinely_empty_deliberate_block():
    fake_self = SimpleNamespace(cfg={}, _weak_think=None, events=SimpleNamespace(fire=lambda *a, **kw: None))
    verdict = AgentLoop._check_think(fake_self, "", dry_run=False, tier=DELIBERATE)
    assert verdict["ok"] is False
    assert fake_self._weak_think is True
