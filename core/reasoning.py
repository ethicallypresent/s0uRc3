"""Reasoning engine — first-principles protocol for deliberate turns.

Two phases, gated by whether the plan is still real:

  Phase 1 (break it down) fires while the task has no real plan yet —
  state the problem plainly, surface assumptions, keep only the ones
  that are law rather than convention, reduce to bedrock, rebuild
  upward. It ends in `update_plan`, which is exactly what turns the
  plan from seed to real — so it can only ever fire once per task,
  never spin (no "breaking down forever").

  Phase 2 (act from it) fires once a real plan exists — cheapest test
  first, act on the fundamentals rather than "that's how it's usually
  done", treat the result as evidence that updates the plan rather
  than a verdict to defend, only build on what survived.

Only ever attached on the `deliberate` tier (core/router.py decides
that before this module runs) — act/glance turns never see this.
"""

from __future__ import annotations

from typing import Any

_PHASE_1 = """\
REASONING — Phase 1, break it down (no real plan yet):
1. State the problem in one plain sentence.
2. List 2-4 assumptions you're carrying, including invisible ones.
3. Keep only the assumptions that are law (physics/math/logic). Drop convention.
4. What's left is the bedrock — what you're actually sure of.
5. From that bedrock alone, rebuild the approach.
Then emit update_plan: the pieces as ordered nodes, one in_progress. Don't act on a node yet."""

_PHASE_2 = """\
REASONING — Phase 2, act from it (plan is real):
6. Take the cheapest step that could prove this node wrong.
7. Act on the fundamentals, not "that's how it's usually done."
8. Result disagrees with the plan? Update the plan, don't defend it.
9. Build only on what just passed."""

_TRAILER = "\nLabel each claim observed (tool/user/verified memory), inferred, or speculative."


def build_scaffold(goal: str, *, mode: str, step: int, last_result: dict[str, Any] | None, plan_seed: bool = True) -> str:
    body = _PHASE_1 if plan_seed else _PHASE_2
    return body + _TRAILER


def packet_with_reasoning(
    packet_body: str,
    *,
    goal: str,
    mode: str,
    step: int,
    last_result: dict[str, Any] | None,
    plan_seed: bool = True,
) -> str:
    scaffold = build_scaffold(goal, mode=mode, step=step, last_result=last_result, plan_seed=plan_seed)
    return f"{scaffold}\n\n{packet_body}"
