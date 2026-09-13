"""Difficulty router: classify_turn must be a pure function over signals
the loop already has — no I/O, no randomness, no model call. Every case
here is deterministic: same inputs in, same tier out.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.router import ACT, DELIBERATE, GLANCE, classify_turn, reasoning_budget_line


def _classify(**overrides):
    base = dict(
        goal="do something",
        mode="action",
        step=1,
        last_result=None,
        tool_candidates=None,
        should_create_skill=False,
        steer="continue",
        weak_think=False,
    )
    base.update(overrides)
    return classify_turn(**base)


def test_trivial_chat_is_act():
    assert _classify(goal="hi", mode="chat") == ACT
    assert _classify(goal="Thanks!", mode="chat") == ACT


def test_substantive_chat_is_glance_not_deliberate():
    assert _classify(goal="what do you think about this design?", mode="chat") == GLANCE


def test_destructive_ask_is_always_deliberate_even_in_chat_mode():
    assert _classify(goal="delete the old backups", mode="chat") == DELIBERATE
    assert _classify(goal="please delete workspace/old.txt", mode="action", step=1,
                      tool_candidates=[{"name": "write_file", "p_success": 0.99}]) == DELIBERATE


def test_first_step_of_action_defaults_to_deliberate_with_no_track_record():
    assert _classify(mode="action", step=1, tool_candidates=None) == DELIBERATE
    assert _classify(mode="action", step=1, tool_candidates=[{"name": "list_dir", "p_success": 0.67}]) == DELIBERATE


def test_first_step_with_a_well_proven_tool_match_is_act():
    assert _classify(mode="action", step=1, tool_candidates=[{"name": "list_dir", "p_success": 0.95}]) == ACT


def test_continuing_a_working_plan_with_a_proven_tool_is_act():
    assert _classify(
        mode="action", step=3, last_result={"ok": True},
        tool_candidates=[{"name": "read_file", "p_success": 0.85}],
    ) == ACT


def test_continuing_with_an_unproven_tool_is_glance_not_deliberate():
    assert _classify(
        mode="action", step=3, last_result={"ok": True},
        tool_candidates=[{"name": "read_file", "p_success": 0.7}],
    ) == GLANCE


def test_human_interrupt_always_escalates_to_deliberate():
    """A live mid-run steer (e.g. s0uRc3's Ctrl+C) is exactly the moment to
    actually think, not rush an act-tier response that might ignore it —
    even when everything else about the turn looks routine."""
    assert _classify(
        mode="action", step=5, last_result={"ok": True},
        tool_candidates=[{"name": "read_file", "p_success": 0.99}],
        human_interrupted=True,
    ) == DELIBERATE
    assert _classify(mode="chat", goal="hi", human_interrupted=True) == DELIBERATE


def test_a_failed_last_step_escalates_to_deliberate():
    assert _classify(
        mode="action", step=3, last_result={"ok": False},
        tool_candidates=[{"name": "read_file", "p_success": 0.95}],
    ) == DELIBERATE


def test_pending_skill_draft_escalates_to_deliberate():
    assert _classify(mode="action", step=2, last_result={"ok": True}, should_create_skill=True) == DELIBERATE


def test_weak_think_or_replan_steer_escalates_to_deliberate():
    assert _classify(mode="action", step=2, last_result={"ok": True}, weak_think=True) == DELIBERATE
    assert _classify(mode="action", step=2, last_result={"ok": True}, steer="replan") == DELIBERATE
    assert _classify(mode="action", step=2, last_result={"ok": True}, steer="stop_spin") == DELIBERATE


def test_classify_is_pure_same_input_same_output():
    kwargs = dict(
        goal="read main.py", mode="action", step=2, last_result={"ok": True},
        tool_candidates=[{"name": "read_file", "p_success": 0.72}],
        should_create_skill=False, steer="continue", weak_think=False,
    )
    results = {classify_turn(**kwargs) for _ in range(50)}
    assert len(results) == 1


def test_reasoning_budget_line_is_one_line_per_tier():
    for tier in (ACT, GLANCE, DELIBERATE):
        line = reasoning_budget_line(tier)
        assert "\n" not in line
        assert line.startswith("reasoning_budget:")
