"""Intra-run steering after each tool result.

Steer values:
  continue  — keep going
  replan    — last result failed; try another tool or update the plan
  stop_spin — same tool failed repeatedly; halt
  rest      — plan is done or repeating a successful tool; finish
"""

from __future__ import annotations

from typing import Any

from core.event_bus import Component, Event
from core.reflection import TraceStep


def evaluate_steer(
    trace: list[TraceStep],
    last_result: dict[str, Any] | None,
    *,
    beliefs: dict[str, Any] | None = None,
    plan_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not trace:
        return {
            "steer": "continue",
            "reason": "no trace yet",
            "repeating": False,
            "last_failed": False,
            "belief_p": None,
        }
    last = trace[-1]
    last_failed = not bool(last.result_ok)
    if isinstance(last_result, dict) and last_result.get("ok") is False:
        last_failed = True

    tool_names = [
        (s.tool_name or "")
        for s in trace[-2:]
        if s.action == "use_tool" and (s.tool_name or "")
    ]
    # 2 in a row, not 3: a small local model re-issuing the very same
    # successful tool call is never useful information, only wasted steps
    # (each a multi-minute round trip on CPU) — no reason to let it happen
    # a third time before cutting it off.
    repeating = len(tool_names) >= 2 and len(set(tool_names)) == 1
    fail_streak = 0
    for step in reversed(trace):
        if step.result_ok:
            break
        fail_streak += 1

    belief_p = None
    if beliefs:
        key = f"tool:{last.tool_name}" if last.tool_name else last.action
        raw = beliefs.get(key) or beliefs.get(last.action)
        if isinstance(raw, dict):
            belief_p = raw.get("p_success")
        elif raw is not None:
            belief_p = getattr(raw, "p_success", None)

    health = plan_health or {}
    if health.get("all_done") and last.action != "finish":
        return {
            "steer": "rest",
            "reason": "plan nodes are done",
            "repeating": repeating,
            "last_failed": last_failed,
            "belief_p": belief_p,
        }
    if repeating and fail_streak >= 3:
        return {
            "steer": "stop_spin",
            "reason": f"repeated {tool_names[-1]} failed {fail_streak} times",
            "repeating": True,
            "last_failed": True,
            "belief_p": belief_p,
            "tool": tool_names[-1],
        }
    if repeating and not last_failed:
        return {
            "steer": "rest",
            "reason": f"same tool {tool_names[-1]} called twice in a row with nothing new",
            "repeating": True,
            "last_failed": False,
            "belief_p": belief_p,
            "tool": tool_names[-1],
        }
    if last_failed and belief_p is not None and float(belief_p) < 0.35:
        return {
            "steer": "rest",
            "reason": "low success rate after a failed tool",
            "repeating": repeating,
            "last_failed": True,
            "belief_p": belief_p,
        }
    if last_failed:
        return {
            "steer": "replan",
            "reason": "last tool failed — switch tool or update_plan",
            "repeating": repeating,
            "last_failed": True,
            "belief_p": belief_p,
        }
    return {
        "steer": "continue",
        "reason": "last measurement ok",
        "repeating": repeating,
        "last_failed": False,
        "belief_p": belief_p,
    }


class Steering(Component):
    """Holds the latest steer so the next packet can cite it."""

    name = "steering"

    def __init__(self) -> None:
        self.last: dict[str, Any] = evaluate_steer([], None)

    def evaluate(
        self,
        trace: list[TraceStep],
        last_result: dict[str, Any] | None,
        *,
        beliefs: dict[str, Any] | None = None,
        plan_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.last = evaluate_steer(
            trace, last_result, beliefs=beliefs, plan_health=plan_health
        )
        return self.last

    def receive(self, event: Event) -> list[Event]:
        if event.kind != "evaluate":
            return []
        payload = event.payload or {}
        steer = self.evaluate(
            list(payload.get("trace") or []),
            payload.get("last_result") if isinstance(payload.get("last_result"), dict) else None,
            beliefs=payload.get("beliefs") if isinstance(payload.get("beliefs"), dict) else None,
            plan_health=payload.get("plan_health") if isinstance(payload.get("plan_health"), dict) else None,
        )
        channel = "control" if steer["steer"] in ("stop_spin", "rest") else "feedback"
        return [
            Event(
                channel=channel,
                kind="steer",
                payload=steer,
                step=event.step,
                source=self.name,
            )
        ]
