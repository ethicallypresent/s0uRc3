"""core/reflection.py — Post-run reflection, pattern analysis, and Bayesian
belief updating over action quality.

Called at the end of every AgentLoop.run(). Produces:
  • Dense 'lesson'/'failure'/'constraint' memories saved via Memory.
  • Updated Beta-distribution beliefs persisted to db/beliefs.json.
  • A ReflectionReport attached to the loop's return value.

Bayesian model
--------------
For each action_type we track Beta(α, β):
  α = pseudo-count of "this type contributed to run success"
  β = pseudo-count of "this type contributed to run failure"
  P(step_ok | action_type) = α / (α + β)

Updating after a run:
  • Successful run:  every action type used gets α += 1.
  • Failed run:      blame is distributed geometrically — the step
    closest to the failure gets the most:
      blame_k  = γ^(n − 1 − k),   γ = 0.8
      credit_k = 1 − blame_k
    β += blame_k, α += credit_k per step.

Counterfactual model
--------------------
For each action taken we define a canonical "worst alternative." The
cost of that wrong choice is:

  p_correct = P(step_ok | action_taken)        (from Beta belief)
  p_wrong   = P(step_ok | worst_alternative)   (from Beta belief)
  lift_k    = p_correct / p_wrong              (how much better correct was)

Run-level analysis uses the product rule (steps treated as approximately
independent — a simplification that holds when steps are weakly coupled,
which is true for a goal-directed loop where each step reacts to the
previous result):

  P(run_success | all_correct) = ∏_k p_correct_k
  P(run_success | all_wrong)   = ∏_k p_wrong_k
  run_lift = P(correct) / P(wrong)

When the model is new, priors are Beta(2,1) for every type, giving
p_success ≈ 0.67 for everything. lift ≈ 1.0 at first — honest: with no
data we cannot claim a correct choice is better. The lift grows as
observations accumulate and the beliefs differentiate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("agent.reflection")

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

BELIEF_FILE = "beliefs.json"
DECAY_GAMMA = 0.8            # geometric decay for failure blame
MIN_LIFT_TO_FLAG = 1.5       # flag a step as a "key decision" if lift >= this
LOW_EFFICIENCY_THRESHOLD = 0.6   # ok_steps / total_steps below this is flagged
MAX_LESSON_CHARS = 480       # fits constitution rule 21 (one dense sentence)

# Default Beta parameters — slight success bias because the agent wouldn't
# be taking an action type at all if it were expected to fail most of the time.
_DEFAULT_ALPHA = 2.0
_DEFAULT_BETA  = 1.0

# For each action type: (worst_alternative_label, E[steps_wasted], reason_template)
# "steps_wasted" is the expected number of extra steps needed to recover from
# the wrong choice, before progress can resume.
WORST_ALTERNATIVES: dict[str, tuple[str, int, str]] = {
    "use_tool": (
        "create_skill",
        3,
        "drafting an unnecessary skill would waste ≥{wasted} steps and consume "
        "a draft slot that may be needed later",
    ),
    "create_skill": (
        "misapply_tool",
        4,
        "forcing an ill-fitting tool instead of drafting the right skill would "
        "require ≥{wasted} recovery steps to detect and correct the mismatch",
    ),
    "update_plan": (
        "ignore_plan_drift",
        2,
        "ignoring a stale plan causes ≥{wasted} misaligned steps before "
        "reality forces replanning anyway",
    ),
    "save_memory": (
        "discard_lesson",
        0,
        "discarding the lesson means future runs pay the discovery cost again "
        "instead of retrieving it from memory",
    ),
    "finish": (
        "continue_unbounded",
        5,
        "not finishing when the goal is met wastes ≥{wasted} steps toward "
        "budget exhaustion and risks the plan diverging from the completed goal",
    ),
    "request_confirmation": (
        "proceed_without_confirm",
        0,
        "proceeding without confirmation on a destructive action risks an "
        "irreversible outcome (constitution rule 14); recovery cost is unbounded",
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------

@dataclass
class TraceStep:
    """One turn of the agent loop, captured for post-run analysis."""
    step: int
    action: str             # action type (e.g. "use_tool", "create_skill")
    rationale: str          # from the model's JSON rationale field
    result_ok: bool
    result_error: str = ""
    drafts_used: int = 0    # cumulative draft count at this point in the run
    approved_skills: int = 0  # how many approved skills existed at this step
    think_snippet: str = ""   # first 200 chars of the <think> block
    tool_name: str = ""       # use_tool name, if any — needed for spin detection


@dataclass
class ActionBelief:
    """Beta distribution over P(step_ok) for one action type."""
    action: str
    alpha: float = _DEFAULT_ALPHA
    beta: float  = _DEFAULT_BETA
    observations: int = 0

    @property
    def p_success(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success_weight: float, failure_weight: float) -> None:
        self.alpha += success_weight
        self.beta  += failure_weight
        self.observations += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "alpha": round(self.alpha, 4),
            "beta":  round(self.beta,  4),
            "p_success": round(self.p_success, 4),
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionBelief":
        return cls(
            action=d["action"],
            alpha=float(d.get("alpha", _DEFAULT_ALPHA)),
            beta =float(d.get("beta",  _DEFAULT_BETA)),
            observations=int(d.get("observations", 0)),
        )


# -----------------------------------------------------------------------
# Belief persistence
# -----------------------------------------------------------------------

class BeliefStore:
    """Loads and saves Beta-distribution beliefs to db/beliefs.json.

    Atomic writes (write-to-temp + os.replace) so a crash during a save
    never produces a corrupt file — same pattern as EvolutionLog.
    """

    def __init__(self, db_dir: Path):
        self.path = db_dir / BELIEF_FILE
        self.beliefs: dict[str, ActionBelief] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not load beliefs from %s: %s", self.path, exc)
            return
        for key, val in raw.items():
            try:
                self.beliefs[key] = ActionBelief.from_dict(val)
            except (KeyError, TypeError) as exc:
                log.warning("Skipping malformed belief %r: %s", key, exc)

    def get(self, action: str) -> ActionBelief:
        if action not in self.beliefs:
            self.beliefs[action] = ActionBelief(action=action)
        return self.beliefs[action]

    def save(self) -> None:
        payload = {k: v.to_dict() for k, v in self.beliefs.items()}
        text = json.dumps(payload, indent=2)
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".beliefs.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                os.replace(tmp, self.path)
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except OSError as exc:
            log.error("Could not persist beliefs to %s: %s", self.path, exc)


# -----------------------------------------------------------------------
# Pattern detection
# -----------------------------------------------------------------------

@dataclass
class Pattern:
    name: str
    severity: str  # "info" | "warn" | "critical"
    detail: str


class PatternDetector:
    """Identifies behavioral anti-patterns in an action trace."""

    def __init__(self, cfg: dict[str, Any]):
        self.max_drafts = int(cfg.get("max_skill_drafts_per_run", 3))
        self.max_steps  = int(cfg.get("max_steps", 50))

    def detect(self, trace: list[TraceStep], run_ok: bool) -> list[Pattern]:
        if not trace:
            return []

        patterns: list[Pattern] = []
        n = len(trace)
        actions = [s.action for s in trace]
        ok_steps = sum(1 for s in trace if s.result_ok)
        efficiency = ok_steps / n

        create_steps  = [s for s in trace if s.action == "create_skill"]
        finish_steps  = [s for s in trace if s.action == "finish"]

        # Skill over-creation: used all draft slots
        if len(create_steps) >= self.max_drafts:
            patterns.append(Pattern(
                name="draft_cap_reached",
                severity="warn",
                detail=(
                    f"All {self.max_drafts} draft slot(s) consumed this run. "
                    "Verify no approved skill could have covered the need — "
                    "constitution rule 10 requires a fit-check before drafting."
                ),
            ))

        # Skill blindness: drafted while the approved catalog was non-empty
        blind = [s for s in create_steps if s.approved_skills > 0]
        if blind:
            patterns.append(Pattern(
                name="skill_blindness",
                severity="warn",
                detail=(
                    f"{len(blind)} skill(s) drafted when the approved catalog "
                    f"already contained {blind[0].approved_skills} skill(s). "
                    "A closer fit-check before drafting would have preserved draft budget."
                ),
            ))

        # Low step efficiency: many steps that returned ok=False
        if efficiency < LOW_EFFICIENCY_THRESHOLD and n >= 5:
            patterns.append(Pattern(
                name="low_step_efficiency",
                severity="warn",
                detail=(
                    f"{ok_steps}/{n} steps returned ok=True "
                    f"({efficiency:.0%}). High failure rates usually indicate "
                    "a plan mismatch or the wrong tool being selected repeatedly."
                ),
            ))

        # Budget exhaustion: failed and hit the step limit
        if not run_ok and n >= self.max_steps:
            patterns.append(Pattern(
                name="budget_exhausted",
                severity="critical",
                detail=(
                    f"Run consumed all {self.max_steps} steps without reaching finish. "
                    "The goal likely needs decomposition into smaller sub-goals, "
                    "or finish detection needs to trigger earlier."
                ),
            ))

        # Plan drift: many plan updates suggest unstable goal understanding
        plan_updates = actions.count("update_plan")
        if plan_updates >= 3:
            patterns.append(Pattern(
                name="plan_drift",
                severity="info",
                detail=(
                    f"{plan_updates} plan updates in one run. "
                    "Frequent replanning suggests the initial plan was underspecified "
                    "or the goal decomposition was too coarse."
                ),
            ))

        # Memory spam: many saves in one run risks near-duplicate lessons
        memory_saves = actions.count("save_memory")
        if memory_saves >= 4:
            patterns.append(Pattern(
                name="memory_spam",
                severity="warn",
                detail=(
                    f"{memory_saves} save_memory calls in one run. "
                    "Constitution rule 19 asks for dense, non-redundant lessons. "
                    "Check db/memory.sqlite for near-duplicate entries from this run."
                ),
            ))

        # Repeated confirmation blocks: plan should front-load these
        confirms = actions.count("request_confirmation")
        if confirms >= 2:
            patterns.append(Pattern(
                name="repeated_confirmation_blocks",
                severity="info",
                detail=(
                    f"{confirms} confirmation requests in one run. "
                    "If the goal predictably involves destructive actions, "
                    "the plan should surface all of them before any are attempted."
                ),
            ))

        # Positive: finished well within budget
        if finish_steps and run_ok:
            at_step = finish_steps[0].step
            budget_used = at_step / self.max_steps
            if budget_used <= 0.5:
                patterns.append(Pattern(
                    name="efficient_finish",
                    severity="info",
                    detail=(
                        f"Goal completed at step {at_step}/{self.max_steps} "
                        f"({budget_used:.0%} of step budget). Efficient execution."
                    ),
                ))

        return patterns


# -----------------------------------------------------------------------
# Counterfactual model
# -----------------------------------------------------------------------

@dataclass
class CounterfactualStep:
    step: int
    action_taken: str
    worst_alternative: str
    p_correct: float         # P(step_ok | action_taken)
    p_wrong: float           # P(step_ok | worst_alternative)
    individual_lift: float   # p_correct / p_wrong
    steps_wasted_if_wrong: int
    reason: str
    is_key_decision: bool    # True when individual_lift >= MIN_LIFT_TO_FLAG


class CounterfactualModel:
    """Estimates the consequence of taking the worst alternative at each step.

    The product-rule assumption (steps are approximately independent) holds
    when each step re-conditions on the previous result — which is exactly
    what the agent loop does. It breaks down for strongly correlated steps
    (e.g. a 5-step sequence that is all-or-nothing), but for a general-purpose
    agent loop it's a reasonable first-order model.
    """

    def __init__(self, beliefs: BeliefStore, cfg: dict[str, Any]):
        self.beliefs  = beliefs
        self.max_steps = int(cfg.get("max_steps", 50))

    def analyze(self, trace: list[TraceStep]) -> list[CounterfactualStep]:
        n = len(trace)
        results = []
        for i, step in enumerate(trace):
            steps_remaining = n - i
            alt, wasted_raw, reason_tmpl = WORST_ALTERNATIVES.get(
                step.action,
                ("unknown", 1, "no counterfactual model defined for this action type"),
            )
            wasted = min(wasted_raw, steps_remaining)
            reason = reason_tmpl.format(wasted=wasted)

            p_correct = self.beliefs.get(step.action).p_success
            p_wrong   = self.beliefs.get(alt).p_success
            p_wrong_safe = max(p_wrong, 0.01)  # floor to avoid division by zero
            lift = p_correct / p_wrong_safe

            results.append(CounterfactualStep(
                step=step.step,
                action_taken=step.action,
                worst_alternative=alt,
                p_correct=round(p_correct, 4),
                p_wrong  =round(p_wrong,   4),
                individual_lift=round(lift, 3),
                steps_wasted_if_wrong=wasted,
                reason=reason,
                is_key_decision=(lift >= MIN_LIFT_TO_FLAG),
            ))
        return results

    def run_level_analysis(self, steps: list[CounterfactualStep]) -> dict[str, Any]:
        """Compute run-level success probability under correct vs worst actions.

        Uses log-sum to avoid floating-point underflow on long runs.
        """
        if not steps:
            return {
                "p_success_correct": 1.0,
                "p_success_wrong":   1.0,
                "lift": 1.0,
                "total_steps_wasted_if_all_wrong": 0,
                "key_decision_steps": [],
                "interpretation": "No steps to analyze.",
            }

        log_p_correct = sum(math.log(max(s.p_correct, 1e-9)) for s in steps)
        log_p_wrong   = sum(math.log(max(s.p_wrong,   1e-9)) for s in steps)

        p_success_correct = math.exp(log_p_correct)
        p_success_wrong   = math.exp(log_p_wrong)
        lift = p_success_correct / max(p_success_wrong, 1e-9)

        total_wasted = sum(s.steps_wasted_if_wrong for s in steps)
        key = [s for s in steps if s.is_key_decision]

        interpretation = (
            f"Correct action selection at every step made a successful run "
            f"{lift:.1f}× more likely than the worst-alternative sequence would have. "
            f"Had all wrong choices been made, an estimated {total_wasted} additional "
            f"steps would have been wasted before recovery or budget exhaustion. "
            f"The {len(key)} key decision(s) at step(s) "
            f"{[s.step for s in key] or ['none']} each carried a lift "
            f"≥{MIN_LIFT_TO_FLAG}× individually."
        )

        return {
            "p_success_correct": round(p_success_correct, 6),
            "p_success_wrong":   round(p_success_wrong,   6),
            "lift": round(lift, 3),
            "total_steps_wasted_if_all_wrong": total_wasted,
            "key_decision_steps": [s.step for s in key],
            "interpretation": interpretation,
        }


# -----------------------------------------------------------------------
# Belief updater
# -----------------------------------------------------------------------

class BeliefUpdater:
    """Soft, decay-weighted update of Beta beliefs after a run.

    Successful run: every action type that appeared gets α += 1 (all
    contributed to the good outcome).

    Failed run: blame decays geometrically from the last step backward
    (γ = 0.8). The step immediately before failure gets blame ≈ 1.0;
    the first step gets blame ≈ γ^(n−1). Credit = 1 − blame, so early
    steps still get partial credit even in a failed run.

    This is a soft heuristic, not a rigorous causal attribution — it
    captures "late steps are more likely the proximate cause of failure"
    without claiming to identify the true causal chain.
    """

    def update(
        self, beliefs: BeliefStore, trace: list[TraceStep], run_ok: bool
    ) -> None:
        n = len(trace)
        for i, step in enumerate(trace):
            b = beliefs.get(step.action)
            if run_ok:
                b.update(success_weight=1.0, failure_weight=0.0)
            else:
                rank_from_end = n - 1 - i
                blame  = DECAY_GAMMA ** rank_from_end
                credit = 1.0 - blame
                b.update(success_weight=credit, failure_weight=blame)


# -----------------------------------------------------------------------
# Report and engine
# -----------------------------------------------------------------------

@dataclass
class ReflectionReport:
    run_ok: bool
    steps_taken: int
    patterns: list[Pattern]
    counterfactual_steps: list[CounterfactualStep]
    run_analysis: dict[str, Any]
    lessons: list[dict[str, Any]]   # ready for Memory.save()

    def one_line_summary(self) -> str:
        status = "succeeded" if self.run_ok else "failed"
        lift   = self.run_analysis.get("lift", 1.0)
        keys   = self.run_analysis.get("key_decision_steps", [])
        crits  = [p.name for p in self.patterns if p.severity == "critical"]
        warns  = [p.name for p in self.patterns if p.severity == "warn"]
        parts  = [
            f"Run {status} in {self.steps_taken} step(s).",
            f"Correct choices were {lift:.1f}× more likely to succeed than worst alternatives.",
        ]
        if keys:
            parts.append(f"Key decisions at step(s): {keys}.")
        if crits:
            parts.append(f"Critical: {', '.join(crits)}.")
        if warns:
            parts.append(f"Warnings: {', '.join(warns)}.")
        return " ".join(parts)


class ReflectionEngine:
    """Entry point for post-run reflection.

    Construct once per AgentLoop instance and call reflect() at the end
    of every run(). The engine holds the BeliefStore across calls so
    beliefs accumulate across runs rather than starting fresh each time.

    Usage in loop.py:
        # __init__:
        self.reflection = ReflectionEngine(self.cfg, self.paths.db)
        self._trace: list[TraceStep] = []

        # after each _dispatch:
        self._trace.append(TraceStep(...))

        # before each run() return:
        report = self.reflection.reflect(self._trace, result)
        for item in report.lessons:
            self.memory.save(item["text"], kind=item["kind"], verified=item["verified"])
        result["reflection"] = report.run_analysis
        self._trace = []   # reset for next run
    """

    def __init__(self, cfg: dict[str, Any], db_dir: Path):
        self.cfg          = cfg
        self.beliefs      = BeliefStore(db_dir)
        self.updater      = BeliefUpdater()
        self.detector     = PatternDetector(cfg)
        self.cf_model     = CounterfactualModel(self.beliefs, cfg)

    def reflect(
        self, trace: list[TraceStep], result: dict[str, Any]
    ) -> ReflectionReport:
        run_ok = bool(result.get("ok"))

        # 1. Update beliefs (before analysis so this run's data feeds in)
        self.updater.update(self.beliefs, trace, run_ok)
        self.beliefs.save()

        # 2. Pattern detection
        patterns = self.detector.detect(trace, run_ok)

        # 3. Counterfactual analysis
        cf_steps    = self.cf_model.analyze(trace)
        run_analysis = self.cf_model.run_level_analysis(cf_steps)

        # 4. Distil lessons (constitution rule 21: one dense sentence)
        lessons = self._distil_lessons(trace, patterns, cf_steps, run_analysis, run_ok)

        report = ReflectionReport(
            run_ok=run_ok,
            steps_taken=len(trace),
            patterns=patterns,
            counterfactual_steps=cf_steps,
            run_analysis=run_analysis,
            lessons=lessons,
        )
        log.info("Reflection complete: %s", report.one_line_summary())
        return report

    def _lesson(self, text: str, kind: str = "lesson") -> dict[str, Any]:
        return {"text": text[:MAX_LESSON_CHARS], "kind": kind, "verified": True}

    def _distil_lessons(
        self,
        trace: list[TraceStep],
        patterns: list[Pattern],
        cf_steps: list[CounterfactualStep],
        run_analysis: dict[str, Any],
        run_ok: bool,
    ) -> list[dict[str, Any]]:
        lessons: list[dict[str, Any]] = []

        # Only persist what a future run can act on. Bayesian lift / step-index
        # commentary is analysis for the report, not a reusable lesson.
        for p in patterns:
            if p.severity in ("warn", "critical"):
                lessons.append(self._lesson(p.detail))

        if not run_ok and trace:
            last = trace[-1]
            who = last.tool_name or last.action
            err = (last.result_error or "unknown error").strip()[:160]
            lessons.append(self._lesson(
                f"{who} failed: {err}" if err else f"{who} failed.",
                kind="failure",
            ))

        max_steps = int(self.cfg.get("max_steps", 50))
        if max_steps > 0 and len(trace) / max_steps >= 0.8:
            lessons.append(self._lesson(
                f"Used {len(trace)}/{max_steps} steps. Split the goal or finish earlier.",
                kind="constraint",
            ))

        return lessons


# -----------------------------------------------------------------------
# Trace persistence + CLI (Reward button / standalone reflect)
# -----------------------------------------------------------------------

LAST_TRACE_FILE = "last_trace.json"


def save_last_trace(db_dir: Path, *, goal: str, result: dict[str, Any], trace: list[TraceStep]) -> Path:
    """Persist the most recent run so the UI can reward it later."""
    path = db_dir / LAST_TRACE_FILE
    payload = {
        "saved_at": _now(),
        "goal": goal,
        "ok": bool(result.get("ok")),
        "result": {
            "ok": result.get("ok"),
            "error": result.get("error"),
            "steps": result.get("steps"),
            "finish": result.get("finish"),
            "status": result.get("status"),
        },
        "trace": [
            {
                "step": t.step,
                "action": t.action,
                "rationale": t.rationale,
                "result_ok": t.result_ok,
                "result_error": t.result_error,
                "drafts_used": t.drafts_used,
                "approved_skills": t.approved_skills,
                "think_snippet": t.think_snippet,
                "tool_name": t.tool_name,
            }
            for t in trace
        ],
        "rewarded": False,
    }
    text = json.dumps(payload, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".last_trace.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_last_trace(db_dir: Path) -> dict[str, Any] | None:
    path = db_dir / LAST_TRACE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load last trace: %s", exc)
        return None


def trace_from_payload(payload: dict[str, Any]) -> list[TraceStep]:
    out: list[TraceStep] = []
    for item in payload.get("trace") or []:
        out.append(
            TraceStep(
                step=int(item.get("step", 0)),
                action=str(item.get("action", "")),
                rationale=str(item.get("rationale", "")),
                result_ok=bool(item.get("result_ok")),
                result_error=str(item.get("result_error", "")),
                drafts_used=int(item.get("drafts_used", 0)),
                approved_skills=int(item.get("approved_skills", 0)),
                think_snippet=str(item.get("think_snippet", "")),
                tool_name=str(item.get("tool_name", "")),
            )
        )
    return out


def reward_last_run(
    cfg: dict[str, Any],
    db_dir: Path,
    *,
    note: str = "",
    memory: Any | None = None,
) -> dict[str, Any]:
    """Treat the last run as a positive example and reinforce beliefs + lessons.

    Used by the studio Reward button — separate from chat. Marks the saved
    trace as rewarded so double-clicks don't spam identical lessons.
    """
    payload = load_last_trace(db_dir)
    if not payload:
        return {"ok": False, "error": "no_last_trace", "message": "No previous run to reward."}
    if payload.get("rewarded"):
        return {
            "ok": True,
            "already_rewarded": True,
            "message": "Last run was already rewarded. Run the agent again first.",
            "goal": payload.get("goal"),
        }

    trace = trace_from_payload(payload)
    if not trace:
        return {"ok": False, "error": "empty_trace", "message": "Last trace has no steps."}

    engine = ReflectionEngine(cfg, db_dir)
    # Force success labeling so BeliefUpdater applies full α credit.
    fake_result = {"ok": True, "rewarded": True, "steps": len(trace)}
    report = engine.reflect(trace, fake_result)

    # Extra reward nudge: +1 α on every action that appeared (on top of reflect).
    for step in trace:
        b = engine.beliefs.get(step.action)
        b.update(success_weight=1.0, failure_weight=0.0)
    engine.beliefs.save()

    lessons_saved = 0
    if memory is not None:
        if note.strip():
            memory.save(note.strip()[:MAX_LESSON_CHARS], kind="preference", verified=True)
            lessons_saved += 1
        for item in report.lessons:
            memory.save(item["text"], kind=item.get("kind", "lesson"), verified=True)
            lessons_saved += 1

    payload["rewarded"] = True
    payload["rewarded_at"] = _now()
    payload["reward_note"] = note
    (db_dir / LAST_TRACE_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = report.one_line_summary()
    return {
        "ok": True,
        "already_rewarded": False,
        "goal": payload.get("goal"),
        "steps": len(trace),
        "lessons_saved": lessons_saved,
        "summary": summary,
        "run_analysis": report.run_analysis,
        "message": f"Rewarded. {summary}",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m core.reflection [--reward] [--note TEXT]"""
    import argparse
    import sys

    from core.loop import load_config
    from core.memory import Memory
    from core.paths import AgentPaths

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="s0uRc3 reflection / reward")
    ap.add_argument("--reward", action="store_true", help="Reinforce the last run as a good example")
    ap.add_argument("--note", default="", help="Optional note saved with the reward preference")
    ap.add_argument("--reflect-only", action="store_true", help="Analyze last trace without forcing success")
    args = ap.parse_args(argv)

    paths = AgentPaths.discover()
    cfg = load_config(paths)
    memory = Memory(paths.db)
    try:
        if args.reflect_only:
            payload = load_last_trace(paths.db)
            if not payload:
                print(json.dumps({"ok": False, "error": "no_last_trace"}, indent=2))
                return 2
            engine = ReflectionEngine(cfg, paths.db)
            report = engine.reflect(trace_from_payload(payload), {"ok": bool(payload.get("ok"))})
            out = {
                "ok": True,
                "mode": "reflect",
                "summary": report.one_line_summary(),
                "run_analysis": report.run_analysis,
                "lessons": report.lessons,
            }
        else:
            # --reward is the default when this module is invoked by the studio button
            out = reward_last_run(cfg, paths.db, note=args.note, memory=memory)
            out["mode"] = "reward"
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 1
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())