"""Per-tier token budgets (thinking-budgets item 3): the budget for a step
comes from the difficulty router's tier, never one flat cap for everything.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.contracts import KNOWN_LLM_DEFAULTS
from core.loop import AgentLoop
from core.router import ACT, DELIBERATE, GLANCE


def _budget(tier: str, llm_overrides: dict | None = None) -> int:
    fake_self = SimpleNamespace(cfg={"llm": {**KNOWN_LLM_DEFAULTS, **(llm_overrides or {})}})
    return AgentLoop._choose_max_tokens(fake_self, tier)


def test_act_gets_the_smallest_budget():
    assert _budget(ACT) == KNOWN_LLM_DEFAULTS["act_max_tokens"]


def test_glance_gets_the_middle_budget():
    assert _budget(GLANCE) == KNOWN_LLM_DEFAULTS["glance_max_tokens"]


def test_deliberate_gets_the_full_budget():
    """create_skill (a full python module in skill.code) is covered here:
    classify_turn escalates a pending skill draft straight to deliberate,
    whose budget is the largest, not a separate structured tier."""
    assert _budget(DELIBERATE) == KNOWN_LLM_DEFAULTS["deliberate_max_tokens"]


def test_deliberate_falls_back_to_the_legacy_max_tokens_key():
    """A config that only ever set the old flat "max_tokens" (pre-dating
    the router) still gets a real deliberate budget instead of never
    reaching a value."""
    llm = {k: v for k, v in KNOWN_LLM_DEFAULTS.items() if k != "deliberate_max_tokens"}
    llm["max_tokens"] = 999
    fake_self = SimpleNamespace(cfg={"llm": llm})
    assert AgentLoop._choose_max_tokens(fake_self, DELIBERATE) == 999


def test_budgets_are_configurable_not_hardcoded():
    assert _budget(ACT, {"act_max_tokens": 42}) == 42
    assert _budget(GLANCE, {"glance_max_tokens": 111}) == 111
    assert _budget(DELIBERATE, {"deliberate_max_tokens": 999}) == 999
