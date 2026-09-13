"""Difficulty router — thinking budgets.

Classifies a turn into act / glance / deliberate using only signals the
loop already has *before* it builds a packet or calls the model: goal
text, step number, last result, ranked tool candidates (with earned
p_success), whether a skill draft is coming, the steering signal, and
whether the previous think block was flagged weak. No model call, no
prompt interpretation — the decision to think must not itself require
thinking. `classify_turn` is a pure function: same inputs, same tier,
every time.

- act: familiar pattern, answer directly / emit the tool call. No `<think>`.
- glance: mildly uncertain. One short `<think>`, then act.
- deliberate: novel, ambiguous, or high-stakes. Full reasoning.
"""

from __future__ import annotations

from typing import Any

ACT = "act"
GLANCE = "glance"
DELIBERATE = "deliberate"

# Escalate on sight — a cheap keyword check, not a judgment call the model
# should be making about its own irreversible actions.
_DESTRUCTIVE_CUES = (
    "delete", "remove", "overwrite", "wipe", "drop ", "uninstall",
    "format ", "destroy", "kill ", "revoke", "rm -rf", "truncate",
)

_TRIVIAL_CHAT = frozenset(
    {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ok", "okay", "cool", "lol"}
)

# Above the unproven-tool prior (0.67, core/tool_registry.py::_belief_p) —
# a tool only reaches these by actually earning a track record.
HIGH_CONFIDENCE_P = 0.9
CONTINUE_CONFIDENCE_P = 0.8


def _is_destructive_ask(goal: str) -> bool:
    g = (goal or "").lower()
    return any(cue in g for cue in _DESTRUCTIVE_CUES)


def _is_trivial_chat(goal: str) -> bool:
    g = (goal or "").lower().strip().rstrip("!.? ")
    return g in _TRIVIAL_CHAT


def _top_candidate_p(tool_candidates: list[dict[str, Any]] | None) -> float:
    """p_success of the *top-ranked* candidate only — list_filtered already
    sorts by relevance-to-this-goal, so candidate 0 is the model's best
    guess. Taking a max across all k candidates would let an unrelated but
    lucky tool falsely signal confidence for a goal it doesn't match."""
    if not tool_candidates:
        return 0.0
    try:
        return float(tool_candidates[0].get("p_success") or 0.0)
    except (TypeError, ValueError, AttributeError, IndexError):
        return 0.0


def classify_turn(
    *,
    goal: str,
    mode: str,
    step: int,
    last_result: dict[str, Any] | None,
    tool_candidates: list[dict[str, Any]] | None,
    should_create_skill: bool,
    steer: str,
    weak_think: bool,
    human_interrupted: bool = False,
) -> str:
    if _is_destructive_ask(goal):
        return DELIBERATE

    # A live Ctrl+C steer (core/loop.py::queue_steer) just handed the model
    # fresh human guidance mid-task — that's exactly the moment to actually
    # think about it, not rush an act-tier response that might ignore it.
    if human_interrupted:
        return DELIBERATE

    if mode == "chat":
        return ACT if _is_trivial_chat(goal) else GLANCE

    # mode == "action"
    if should_create_skill or weak_think or steer in ("replan", "stop_spin"):
        return DELIBERATE

    top_p = _top_candidate_p(tool_candidates)
    last_ok = bool(isinstance(last_result, dict) and last_result.get("ok"))

    if step == 1:
        # No observed history yet this run — only a well-proven tool match
        # for this exact kind of ask is confident enough to skip thinking.
        return ACT if top_p >= HIGH_CONFIDENCE_P else DELIBERATE

    if not last_ok:
        return DELIBERATE

    if top_p >= CONTINUE_CONFIDENCE_P:
        return ACT

    return GLANCE


# Token budgets per tier (item 3). ACT's cap covers the JSON action only —
# there is no separate way to give the *reasoning portion specifically* a
# token limit within one completion (llama.cpp's max_tokens bounds the
# whole response), so GLANCE's ~50-token reasoning budget is approximated
# by making the *total* budget tight enough that a long think block simply
# doesn't fit alongside the JSON action. This is deliberately a completion
# cap that emulates a reasoning-only cap, not a literal one — flagged in
# the handoff notes, not silently presented as more precise than it is.
TIER_MAX_TOKENS_DEFAULTS = {
    ACT: 120,
    GLANCE: 250,
    DELIBERATE: 800,
}


def reasoning_budget_line(tier: str) -> str:
    """One line telling the model the budget the router already decided —
    not asking it to decide. Included in the packet only; never a standing
    instruction in the system prompt."""
    if tier == ACT:
        return "reasoning_budget: act — skip <think> entirely, output only the json action."
    if tier == GLANCE:
        return "reasoning_budget: glance — <think> in one short line (~50 tokens), then act."
    return "reasoning_budget: deliberate — think fully: want, known, uncertain, risk, next."
