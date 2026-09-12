"""Perceive -> think -> act loop. LOCAL EDITION.

Same loop as the original scaffold, but the LLM backend is a local
llama-server (OpenAI-compatible API) instead of litellm + cloud keys.
Stdlib only: no litellm, no openai package needed.

Config (reasoning_config.json) drives everything:

    "llm": {
        "base_url": "http://localhost:11434/v1",
        "model": "pocket.gguf",
        "api_key": "sk-no-key-needed",
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout_sec": 120,
        "dry_run_on_no_server": true
    }

If llama-server isn't reachable and dry_run_on_no_server is true, the loop
falls back to the deterministic dry-run policy (same as the old NO_LLM path).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Literal

from core.boot import initiate
from core.brain import Brain, load_system_prompt as _load_system_prompt
from core.contracts import ACTIONS, KNOWN_LLM_DEFAULTS, MAX_GOAL_CHARS
from core.evolution import EvolutionLog
from core.steer import Steering
from core.measure import judge, observed_text
from core.memory import Memory
from core.maintain import rest as idle_rest
from core.event_bus import EventBus, Event
from core.packet import fit_payload, perception_block, slim_last_result
from core.paths import AgentPaths
from core.permission import ALLOW, ALLOW_RUN, DENY, inspect_request, normalize_decision, permission_key
from core.physics import WorldState, evaluate_think
from core.plan import needs_real_plan, normalize_plan, plan_health
from core.context import gather_context
from core.reasoning import packet_with_reasoning
from core.reflection import ReflectionEngine, TraceStep, save_last_trace
from core.complete import complete_finish
from core.season import archive_stale_skills
from core.skill_manager import SkillManager
from core.tool_registry import ToolRegistry
from core.world_model import WorldModel

log = logging.getLogger("agent.loop")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

DEFAULT_ALLOWED_ACTIONS = ACTIONS
DEFAULT_REFINE_MAX_PASSES = 3

REVISE_SYSTEM = """# Code Reviser

You receive: (1) the current code, (2) the test failures it produced.
You do not decide whether the code is good — the test harness already
decided it isn't. Your only job: produce revised code that addresses
the failures.

Rules:
- Change the minimum necessary to fix the reported failures.
- Do not refactor unrelated code. Do not add features.
- Preserve the existing interface: skill name, manifest, function
  signatures, inputs/outputs.
- Output ONLY the revised code. No explanation, no commentary.
"""

PermissionDecision = Literal["allow", "deny", "allow_run"]
PermissionCallback = Callable[[dict[str, Any]], str]


def denied_result(name: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Standard result when the human refuses a tool or confirmation."""
    observed: dict[str, Any] = {"denied": True, "name": name}
    if extra:
        for key, val in extra.items():
            if key not in observed:
                observed[key] = val
    return {
        "ok": False,
        "error": "user_denied",
        "from": name or "unknown",
        "observed": observed,
    }

DEFAULT_LLM_CFG = KNOWN_LLM_DEFAULTS

# Named load profiles the UI can apply. Values merge into llm.*.
LOAD_PRESETS: dict[str, dict[str, Any]] = {
    "cpu_optimal": {
        "num_ctx": 4096,
        "num_batch": 256,
        "num_gpu": 0,
        "num_thread": 0,
        "max_tokens": 800,
        "keep_alive": "10m",
        "timeout_sec": 300,
    },
    "balanced": {
        "num_ctx": 8192,
        "num_batch": 512,
        "num_gpu": -1,
        "num_thread": 0,
        "max_tokens": 1200,
        "keep_alive": "15m",
        "timeout_sec": 240,
    },
    "max_context": {
        "num_ctx": 32768,
        "num_batch": 512,
        "num_gpu": 0,
        "num_thread": 0,
        "max_tokens": 900,
        "keep_alive": "15m",
        "timeout_sec": 600,
    },
    "fast_draft": {
        "num_ctx": 2048,
        "num_batch": 128,
        "num_gpu": 0,
        "num_thread": 0,
        "max_tokens": 400,
        "keep_alive": "5m",
        "timeout_sec": 180,
    },
}


class LLMTransientError(RuntimeError):
    """Raised for errors that justify a fallback-model retry."""

    def __init__(self, trigger: str, *args: Any) -> None:
        super().__init__(trigger, *args)
        self.trigger = trigger


def load_config(paths: AgentPaths) -> dict[str, Any]:
    from core.config import load as load_config_file
    from core.resources import apply_resource_allocation

    return apply_resource_allocation(load_config_file(paths))


def load_system_prompt(paths: AgentPaths) -> str:
    """Concatenated identity+law. Canonical organ is core.brain.Brain."""
    return _load_system_prompt(paths)


def _try_load_action_json(raw: str) -> dict[str, Any]:
    """Parse an action object, with light salvage for truncated/malformed JSON."""
    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        # Common 1B-model failure: truncated ```json fence or trailing comma.
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        # If the object was cut off mid-string, close open braces/brackets.
        if cleaned.count("{") > cleaned.count("}"):
            cleaned = cleaned + ("}" * (cleaned.count("{") - cleaned.count("}")))
        if cleaned.count("[") > cleaned.count("]"):
            cleaned = cleaned + ("]" * (cleaned.count("[") - cleaned.count("]")))
        try:
            action = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: pull the action name and build a minimal finish/use_tool.
            am = re.search(r'"action"\s*:\s*"(\w+)"', raw)
            if not am:
                # Free-text salvage: model sometimes skips JSON entirely.
                low = (raw or "").lower()
                if "finish" in low or "hello" in low or "hi " in low:
                    action = {
                        "action": "finish",
                        "rationale": "salvaged free-text reply",
                        "finish": {
                            "status": "success",
                            "summary": (raw or "").strip()[:400] or "done",
                            "artifacts": [],
                        },
                    }
                    log.warning("Salvaged free-text finish from malformed output")
                    return action
                raise
            kind = am.group(1)
            action = {"action": kind, "rationale": "salvaged from malformed model output"}
            if kind == "finish":
                sm = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
                action["finish"] = {
                    "status": "success",
                    "summary": (sm.group(1) if sm else "completed"),
                    "artifacts": [],
                }
            elif kind == "use_tool":
                tm = re.search(r'"name"\s*:\s*"(\w+)"', raw)
                action["tool"] = {"name": tm.group(1) if tm else "list_dir", "args": {"path": ".", "glob": "workspace/*"}}
            log.warning("Salvaged partial action %r from malformed JSON", kind)
    if not isinstance(action, dict) or "action" not in action:
        raise ValueError("action field missing")
    # Small models sometimes nest the action: {"action": {"action": "finish", ...}}
    # or emit {"action": {"use_tool": {...}}}. Normalize to a string kind.
    kind = action.get("action")
    if isinstance(kind, dict):
        if isinstance(kind.get("action"), str):
            action = {**action, **kind}
            kind = action.get("action")
        else:
            for candidate in DEFAULT_ALLOWED_ACTIONS:
                if candidate in kind:
                    nested = kind.get(candidate)
                    action = {
                        **action,
                        "action": candidate,
                        **(nested if isinstance(nested, dict) else {}),
                    }
                    kind = candidate
                    break
    if not isinstance(kind, str):
        raise ValueError(f"action field must be a string, got {type(kind).__name__}: {kind!r}")
    action["action"] = kind
    return action


def parse_model_output(text: str) -> tuple[str, dict[str, Any]]:
    think = ""
    m = THINK_RE.search(text or "")
    if m:
        think = m.group(1).strip()
    jm = JSON_RE.search(text or "")
    raw = jm.group(1) if jm else None
    if raw is None:
        # last-resort: first JSON object
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        if start >= 0 and end > start:
            raw = text[start : end + 1]
        elif start >= 0:
            # Truncated output: take from first brace to end and let salvage close it.
            raw = text[start:]
    if not raw:
        # Let the loop decide continue vs finish (needs step / last_result).
        raise ValueError("no JSON action in model output")
    action = _try_load_action_json(raw)
    return think, action


def goal_mode(goal: str) -> str:
    """Return 'chat' or 'action'.

    Chat: conversational reply is enough (greetings, Q&A, thanks).
    Action: needs tools / skills / desktop work. Defaults back to chat when
    the utterance doesn't look like a task.
    """
    g = (goal or "").lower().strip()
    if not g:
        return "chat"
    # Explicit action cues
    action_cues = (
        "list ", "list your", "find ", "search ", "read ", "write ", "create ",
        "make ", "fix ", "run ", "open ", "delete ", "install ", "build ",
        "test ", "edit ", "save ", "download ", "fetch ", "clone ", "commit ",
        "screenshot", "clipboard", "shell", "cmd", "powershell", "file",
        "folder", "directory", "workspace", "skill", "tool",
    )
    if any(c in g for c in action_cues):
        return "action"
    # Pure chat / social
    chat_cues = (
        "hello", "hi", "hey", "thanks", "thank you", "how are you", "who are you",
        "what can you", "good morning", "good night", "lol", "ok", "okay", "cool",
    )
    if g in {"hi", "hello", "hey", "yo", "sup"} or any(g.startswith(c) for c in chat_cues):
        # "say hello and list your tools" is action (contains list/tool)
        if "tool" in g or "list" in g:
            return "action"
        return "chat"
    # Short utterances without verbs → chat
    if len(g.split()) <= 4 and not any(c in g for c in action_cues):
        return "chat"
    return "action"


def _is_simple_chat_goal(goal: str) -> bool:
    """Backward-compatible alias: true when a quick tool→finish is OK."""
    return goal_mode(goal) == "action" and "tool" in (goal or "").lower()


def build_user_packet(
    *,
    goal: str,
    user_input: str,
    memory: Memory,
    tools: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    memory_topk: int = 5,
    evolution: dict[str, Any] | None = None,
    step: int = 1,
    paths: AgentPaths | None = None,
    world: WorldState | None = None,
    system: str = "",
    num_ctx: int = 4096,
    max_tokens: int = 800,
    max_packet_tokens: int | None = None,
    weak_think: bool = False,
    named: list[dict[str, Any]] | None = None,
    steer: dict[str, Any] | None = None,
    judgment: dict[str, Any] | None = None,
) -> str:
    # Octopus box: multi-arm recall over SQLite + beliefs + skills + tools + evolution.
    paths = paths or AgentPaths.discover()
    box = gather_context(paths, query=goal or user_input, memory_topk=max(int(memory_topk), 8))
    # Overlay THIS run's working memory (gather opens a separate facade).
    box.setdefault("memory", {})["working"] = memory.wm.to_dict()

    last = slim_last_result(memory.wm.last_result)

    slim_tools = box.get("tools") or []
    if tools:
        seen: set[str] = set()
        slim_tools = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            n = str(t["name"])
            if n in seen:
                continue
            seen.add(n)
            slim_tools.append(
                {
                    "name": n,
                    "description": (t.get("description") or "")[:120],
                    "domain": t.get("domain"),
                    "p_success": t.get("p_success"),
                }
            )

    slim_skills = box.get("skills") or []
    if skills:
        slim_skills = [
            {
                "name": s.get("name"),
                "description": (s.get("description") or "")[:120],
                "kind": s.get("kind"),
                "status": s.get("status"),
            }
            for s in skills
            if isinstance(s, dict) and s.get("name")
        ]

    mode = goal_mode(goal)
    health = plan_health(memory.wm.active_plan)
    payload = {
        "user_goal": goal,
        "mode": mode,
        "perception": perception_block(
            user_input=user_input,
            last=last if isinstance(last, dict) else None,
            world=world,
            goal=goal,
        ),
        "run_state": (world.snapshot() if world is not None else {}),
        "context": {
            "note": box.get("note"),
            "brain_files": box.get("brain_files"),
            "memory": {
                "working": memory.wm.to_dict(),
                "recalled": (box.get("memory") or {}).get("recalled") or [],
                "recall_count": (box.get("memory") or {}).get("recall_count") or 0,
            },
            "beliefs": box.get("beliefs") or {},
            "last_trace": box.get("last_trace"),
        },
        "plan": memory.wm.active_plan,
        "plan_health": health,
        "judgment": judgment or {},
        "tool_registry": slim_tools,
        "skill_registry": slim_skills,
        "evolution": evolution or box.get("evolution") or {},
        "named_entities": named if named is not None else (box.get("named_entities") or []),
        "steer": steer or {"steer": "continue", "reason": "no steer yet"},
    }
    if weak_think:
        payload["perception"]["unknown"] = list(payload["perception"].get("unknown") or []) + [
            "previous think lacked observed/inferred/speculative labels"
        ]
    if needs_real_plan(mode=mode, step=step, nodes=memory.wm.active_plan):
        payload["perception"]["unknown"] = list(payload["perception"].get("unknown") or []) + [
            "action-mode past two steps still has the seed plan; emit update_plan"
        ]
    if system:
        payload = fit_payload(
            payload,
            system=system,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            max_packet_tokens=max_packet_tokens,
        )
    body = json.dumps(payload, separators=(",", ":"), default=str)
    wrapped = (
        f"Mode hint: {mode}. Reach into context "
        f"(memory, beliefs, skills, tools) before you act. Reference what you "
        f"pulled in <think>. Label claims observed / inferred / speculative. "
        f"Cite named_entities, plan_health, judgment, and steer.steer. "
        f"Treat tool.observed as fact; error is not a success. "
        f"One plan node in_progress. Finish with observed / inferred / unknown. "
        f"Then output <think>…</think> and one json action. "
        f"Do not echo this packet.\n\nPACKET:\n{body}"
    )
    return packet_with_reasoning(
        wrapped,
        goal=goal,
        mode=mode,
        step=step,
        last_result=last if isinstance(last, dict) else None,
    )


def _classify_llm_exception(exc: Exception) -> str:
    """Best-effort mapping of a raised exception to a fallback_triggers name."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg:
        return "timeout"
    if "ratelimit" in name or "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if "exceed_context" in msg or "context size" in msg or "n_prompt_tokens" in msg:
        return "context_overflow"
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return "rate_limit"
        if 500 <= exc.code <= 599:
            return "5xx_error"
        # 4xx are not "unreachable" — do not trigger dry-run-on-no-server.
        return f"http_{exc.code}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        reason = str(getattr(exc, "reason", "") or exc).lower()
        if "timed out" in reason or "timeout" in reason:
            return "timeout"
        return "connection_error"
    if "apierror" in name or "internalserver" in name or "503" in msg or "500" in msg:
        return "5xx_error"
    return "unknown_error"


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Surface server-side error bodies (llama-server returns JSON errors).
        try:
            body = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            body = ""
        raise LLMTransientError(_classify_llm_exception(exc), f"HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMTransientError(_classify_llm_exception(exc), str(exc)) from exc


def _server_options(llm: dict[str, Any]) -> dict[str, Any]:
    """Map config load knobs into Ollama's `options` object."""
    opts: dict[str, Any] = {}
    for key, cast in (
        ("num_ctx", int),
        ("num_batch", int),
        ("num_gpu", int),
        ("num_thread", int),
        ("top_k", int),
        ("mirostat", int),
        ("top_p", float),
        ("repeat_penalty", float),
    ):
        if key not in llm or llm[key] is None:
            continue
        val = cast(llm[key])
        # 0 threads / unset means "let Ollama decide"
        if key == "num_thread" and val <= 0:
            continue
        opts[key] = val
    return opts


def _chat_payload(
    llm: dict[str, Any],
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat payload.

    llama.cpp (llama-server) takes load knobs on the process CLI, not in the
    JSON body. Ollama accepts keep_alive + options in the request.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature is not None else llm.get("temperature", 0.2),
        "max_tokens": max_tokens if max_tokens is not None else llm.get("max_tokens", 800),
    }
    if "top_p" in llm and llm["top_p"] is not None:
        payload["top_p"] = float(llm["top_p"])
    backend = str(llm.get("backend") or "llamacpp").lower()
    if backend in ("ollama", "both"):
        keep_alive = llm.get("keep_alive")
        if keep_alive:
            payload["keep_alive"] = keep_alive
        opts = _server_options(llm)
        if opts:
            payload["options"] = opts
    return payload


def call_llm(system: str, packet: str, cfg: dict[str, Any], *, model: str | None = None) -> str:
    """Call a local llama-server via its OpenAI-compatible /chat/completions."""
    chunks: list[str] = []
    for piece in call_llm_stream(system, packet, cfg, model=model):
        chunks.append(piece)
    return "".join(chunks)


def call_llm_stream(
    system: str,
    packet: str,
    cfg: dict[str, Any],
    *,
    model: str | None = None,
    on_token: Callable[[str], None] | None = None,
):
    """Yield text tokens from llama.cpp (OpenAI SSE). Falls back to one-shot."""
    llm = cfg.get("llm") or {}
    base_url = llm.get("base_url", DEFAULT_LLM_CFG["base_url"]).rstrip("/")
    use_model = model or llm.get("model", DEFAULT_LLM_CFG["model"])
    api_key = llm.get("api_key", DEFAULT_LLM_CFG["api_key"])
    timeout = int(llm.get("timeout_sec", DEFAULT_LLM_CFG["timeout_sec"]))
    if int(llm.get("num_gpu") or 0) <= 0:
        timeout = max(timeout, 900)

    payload = _chat_payload(
        llm,
        model=use_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": packet},
        ],
    )
    payload["stream"] = True
    log.debug("LLM stream -> %s/chat/completions (model=%s)", base_url, use_model)

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = obj["choices"][0].get("delta") or {}
                    piece = delta.get("content") or ""
                except (KeyError, IndexError, TypeError):
                    piece = ""
                if piece:
                    if on_token:
                        on_token(piece)
                    yield piece
    except urllib.error.HTTPError as exc:
        # Some servers reject stream — fall back to non-stream once.
        try:
            body = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            body = ""
        low = body.lower()
        stream_rejected = exc.code in (404, 415, 501) or (
            exc.code == 400 and "stream" in low
        )
        if stream_rejected:
            log.warning("stream unsupported (%s); falling back to one-shot", exc.code)
            payload.pop("stream", None)
            data = _post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if text:
                if on_token:
                    on_token(text)
                yield text
            return
        log.error("LLM HTTP %s: %s", exc.code, body[:500])
        raise LLMTransientError(_classify_llm_exception(exc), f"HTTP {exc.code}: {body[:400]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMTransientError(_classify_llm_exception(exc), str(exc)) from exc


def warmup_model(cfg: dict[str, Any], *, model: str | None = None) -> bool:
    """Fire a tiny completion so Ollama loads weights before the real run.

    Uses the configured load options (ctx, gpu layers, keep_alive) so the
    warm model matches what the run will actually use. Returns True on
    success. Failures are logged and return False — warmup is best-effort.
    """
    llm = cfg.get("llm") or {}
    if not llm.get("warmup", True):
        log.info("Warmup disabled in config")
        return False
    base_url = llm.get("base_url", DEFAULT_LLM_CFG["base_url"]).rstrip("/")
    use_model = model or llm.get("model", DEFAULT_LLM_CFG["model"])
    api_key = llm.get("api_key", DEFAULT_LLM_CFG["api_key"])
    timeout = int(llm.get("timeout_sec", DEFAULT_LLM_CFG["timeout_sec"]))
    payload = _chat_payload(
        llm,
        model=use_model,
        messages=[{"role": "user", "content": "ping"}],
        temperature=0.0,
        max_tokens=1,
    )
    log.info(
        "Warming up model %s at %s (ctx=%s, gpu=%s, keep_alive=%s) ...",
        use_model,
        base_url,
        llm.get("num_ctx"),
        llm.get("num_gpu"),
        llm.get("keep_alive"),
    )
    try:
        _post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
        log.info("Warmup complete")
        return True
    except LLMTransientError as exc:
        log.warning("Warmup failed (%s): %s", exc.trigger, exc)
        return False


def apply_load_preset(cfg: dict[str, Any], preset: str) -> dict[str, Any]:
    """Merge a named load preset into cfg['llm'] and return cfg."""
    profile = LOAD_PRESETS.get(preset)
    if not profile:
        raise KeyError(f"unknown load preset: {preset}")
    llm = dict(cfg.get("llm") or {})
    llm.update(profile)
    llm["load_preset"] = preset
    cfg = dict(cfg)
    cfg["llm"] = llm
    return cfg


_DRY_RUN_PROJECT = "dry_run_demo"
_BUILD_TOOL_NAMES = {"init_project", "check_axiom", "log_evidence", "advance_axiom"}
_BUILD_GOAL_CUES = ("axiom", "project", "creatorcentral", "build a", "init a")


def _dry_run_build_step(last: dict[str, Any] | None, goal: str) -> dict[str, Any] | None:
    """One step of a scripted CreatorCentral walk: init -> check -> log -> advance -> finish."""
    if not last:
        return {
            "action": "use_tool",
            "rationale": "demonstrate CreatorCentral: start a project at INITIATION",
            "tool": {
                "name": "init_project",
                "args": {"name": _DRY_RUN_PROJECT, "owner": "s0uRc3", "purpose": (goal or "dry-run demo")[:200]},
            },
        }
    frm = last.get("from")
    if frm == "init_project":
        return {
            "action": "use_tool",
            "rationale": "check what INITIATION still needs",
            "tool": {"name": "check_axiom", "args": {"name": _DRY_RUN_PROJECT}},
        }
    if frm == "check_axiom":
        missing = last.get("missing") or []
        if missing:
            return {
                "action": "use_tool",
                "rationale": f"log evidence for missing requirement {missing[0]}",
                "tool": {
                    "name": "log_evidence",
                    "args": {"name": _DRY_RUN_PROJECT, "requirement": missing[0], "text": f"dry-run evidence for {missing[0]}"},
                },
            }
        return {
            "action": "use_tool",
            "rationale": "requirements met; advance the gate",
            "tool": {"name": "advance_axiom", "args": {"name": _DRY_RUN_PROJECT}},
        }
    if frm == "log_evidence":
        return {
            "action": "use_tool",
            "rationale": "re-check after logging evidence",
            "tool": {"name": "check_axiom", "args": {"name": _DRY_RUN_PROJECT}},
        }
    if frm == "advance_axiom":
        if not last.get("advanced"):
            return {
                "action": "finish",
                "rationale": "gate refused or project already at rest; demo complete",
                "finish": {
                    "status": "success",
                    "summary": f"CreatorCentral demo: gate held ({last.get('error', 'already_at_rest')}).",
                    "artifacts": [],
                },
            }
        return {
            "action": "finish",
            "rationale": "axiom gate advanced; demo complete",
            "finish": {
                "status": "success",
                "summary": f"CreatorCentral demo: advanced past {last.get('completed', 'an axiom')}.",
                "artifacts": [],
            },
            "memory_to_save": [
                {"kind": "lesson", "text": "advance_axiom refuses to move on without logged evidence.", "verified": True}
            ],
        }
    return None


def dry_run_policy(packet: dict[str, Any]) -> str:
    """Deterministic stand-in so the scaffold is testable without a server."""
    last = packet["perception"].get("last_tool_result")
    evo = packet.get("evolution") or {}
    goal = (packet.get("user_goal") or "").lower()
    catalog = packet.get("skill_registry") or []

    is_build_walk = isinstance(last, dict) and last.get("from") in _BUILD_TOOL_NAMES
    if is_build_walk or any(c in goal for c in _BUILD_GOAL_CUES):
        build_action = _dry_run_build_step(last if isinstance(last, dict) else None, goal)
        if build_action is not None:
            think = (
                f"intent: {goal[:80]}\n"
                f"observed: last_tool_result from={last.get('from') if isinstance(last, dict) else None}\n"
                f"inferred: CreatorCentral demo chose {build_action['action']}\n"
                f"speculative: none\n"
                f"next: {build_action['action']}"
            )
            return f"<think>\n{think}\n</think>\n```json\n{json.dumps(build_action, indent=2)}\n```"

    have_normalizer = any((s.get("name") or "") == "normalize_text" for s in catalog)
    extendish = any(w in goal for w in ("extend", "skill", "normal"))
    if have_normalizer and extendish and not (last and last.get("from") == "run_skill"):
        action = {
            "action": "use_tool",
            "rationale": "reuse approved skill",
            "tool": {"name": "run_skill", "args": {"name": "normalize_text", "args": {"text": " Hello "}}},
        }
        think = (
            f"intent: {goal[:80]}\n"
            f"observed: approved normalize_text skill in catalog\n"
            f"inferred: reuse beats a new draft\n"
            f"speculative: none\n"
            f"next: run_skill"
        )
        return f"<think>\n{think}\n</think>\n```json\n{json.dumps(action, indent=2)}\n```"

    if last and last.get("promoted"):
        action = {
            "action": "use_tool",
            "rationale": "exercise the skill just promoted",
            "tool": {
                "name": "run_skill",
                "args": {"name": last.get("name"), "args": {"text": "hello"}},
            },
        }
    elif last and last.get("from") == "run_skill" and last.get("ok"):
        action = {
            "action": "finish",
            "rationale": "new capability works",
            "finish": {
                "status": "success",
                "summary": f"extended library with {last.get('name', 'skill')}",
                "artifacts": [],
            },
            "memory_to_save": [
                {"kind": "lesson", "text": "A passing skill test is the promotion gate.", "verified": True}
            ],
        }
    elif evo.get("should_create") and not (last and last.get("name")):
        action = {
            "action": "create_skill",
            "rationale": evo.get("reason") or "capability gap",
            "skill": {
                "name": "normalize_text",
                "description": "Trim and lowercase text so later skills share one cleaner.",
                "inputs": {"text": "str"},
                "outputs": {"ok": "bool", "text": "str"},
                "code": _NORMALIZE_SKILL,
            },
        }
    elif last and last.get("from") == "refine_code":
        action = {
            "action": "finish",
            "rationale": "refine_code completed",
            "finish": {
                "status": "success" if last.get("ok") else "blocked",
                "summary": last.get("error") or f"refined {last.get('name', 'skill')}",
                "artifacts": [],
            },
        }
    elif last and not last.get("ok") and last.get("error") == "tests failed; draft kept for revision" and last.get("name"):
        action = {
            "action": "refine_code",
            "rationale": "draft tests failed; revise against harness failures",
            "refine": {"skill_name": last.get("name")},
        }
    elif last and last.get("ok") and last.get("from") == "list_dir":
        action = {
            "action": "finish",
            "rationale": "listed workspace",
            "finish": {"status": "success", "summary": "workspace listed", "artifacts": []},
            "memory_to_save": [{"kind": "lesson", "text": "Dry-run loop can finish after one tool.", "verified": True}],
        }
    else:
        action = {
            "action": "use_tool",
            "rationale": "inspect workspace first",
            "tool": {"name": "list_dir", "args": {"path": ".", "glob": "workspace/*"}},
        }
    last_ok = last.get("ok") if isinstance(last, dict) else None
    think = (
        f"intent: {goal[:80]}\n"
        f"observed: last_tool_result ok={last_ok!s}\n"
        f"inferred: dry-run policy chose {action['action']}\n"
        f"speculative: none\n"
        f"next: {action['action']}"
    )
    return f"<think>\n{think}\n</think>\n```json\n{json.dumps(action, indent=2)}\n```"


_NORMALIZE_SKILL = '''"""
SKILL MANIFEST
name: normalize_text
description: Trim and lowercase text
kind: transform
inputs: {"text": "str"}
outputs: {"ok": "bool", "text": "str"}
dependencies: []
author: self
version: 1
"""

TESTS = [
    {"args": {"text": "  Hi "}, "expect_ok": True},
]

def run(text: str = "") -> dict:
    return {"ok": True, "name": "normalize_text", "text": str(text).strip().lower()}
'''


class AgentLoop:
    def __init__(self, paths: AgentPaths | None = None):
        state = initiate(paths)
        self.paths = state.paths
        self.cfg = state.config
        self.boot_issues = state.issues
        self.brain = Brain.load(self.paths, cfg=self.cfg)
        # CPU prefill of the full brain is minutes-long and the client
        # cancels; compact identity+law still binds. GPU can take the full text.
        llm0 = self.cfg.get("llm") or {}
        prompt_ctx = int(llm0.get("num_ctx") or 4096)
        if int(llm0.get("num_gpu") or 0) <= 0:
            prompt_ctx = min(prompt_ctx, 4096)
        self.system_prompt = self.brain.system_message(num_ctx=prompt_ctx)
        self.world = WorldState.from_config(self.cfg, identity_hash=self.brain.identity_hash)
        self.events = EventBus()
        self.events.attach(self.brain)
        self.events.attach(self.world)
        self.world_model = WorldModel(self.paths.db)
        self.steering = Steering()
        self.events.attach(self.world_model)
        self.events.attach(self.steering)
        boot = self.brain.boot()
        if not boot.get("ok"):
            log.error("brain boot failed: %s", boot.get("issues"))
        self.events.fire(Event(channel="control", kind="boot", source="loop"), target="brain")
        self.memory = Memory(self.paths.db)
        self.skills = SkillManager(self.paths)
        self.tools = ToolRegistry(self.paths)
        self.tools.attach_skill_runner(self._run_skill_tracked)
        self.drafts_this_run = 0
        self.evolution = EvolutionLog(self.paths.db)
        self.reflection = ReflectionEngine(self.cfg, self.paths.db)
        self._trace: list[TraceStep] = []
        self._weak_think = False
        self._last_judgment: dict[str, Any] = {"verdict": "unknown", "reason": "no act yet"}
        self.allowed_actions = set(self.cfg.get("allowed_actions") or DEFAULT_ALLOWED_ACTIONS)
        self.fallback_triggers = set(self.cfg.get("fallback_triggers") or [])
        self.on_token: Callable[[str], None] | None = None
        # Unset (tests, --once, dry-run) means auto-allow. The TUI always installs this.
        self.on_permission: PermissionCallback | None = None
        self._run_allowed: set[str] = set()
        self.cancel_requested = False
        self._dry_run = False
        if "refine_max_passes" not in self.cfg:
            self.cfg["refine_max_passes"] = DEFAULT_REFINE_MAX_PASSES

    def close(self) -> None:
        self.memory.close()
        self.world_model.close()

    def _run_skill_tracked(self, name: str, args: dict | None = None, allow_draft: bool = False) -> dict:
        out = self.skills.run(name, args, allow_draft=allow_draft)
        self.evolution.record_use(name, bool(out.get("ok")), str(out.get("error", "")))
        out["from"] = "run_skill"
        out["name"] = name
        return out

    def _belief_map(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.reflection.beliefs.beliefs.items()}

    def _complete_finish(self, finish: dict[str, Any] | None, *, goal: str, last_result: dict[str, Any] | None = None) -> dict[str, Any]:
        last = last_result
        if last is None and isinstance(self.memory.wm.last_result, dict):
            last = self.memory.wm.last_result
        return complete_finish(
            finish,
            world=self.world,
            plan=self.memory.wm.active_plan if isinstance(self.memory.wm.active_plan, list) else [],
            last_result=last if isinstance(last, dict) else None,
            goal=goal,
        )

    def _record_tool_belief(self, tool_name: str, ok: bool) -> None:
        if not tool_name:
            return
        belief = self.reflection.beliefs.get(f"tool:{tool_name}")
        if ok:
            belief.update(1.0, 0.0)
        else:
            belief.update(0.0, 1.0)
        try:
            self.reflection.beliefs.save()
        except OSError as exc:
            log.warning("Could not persist tool belief: %s", exc)

    def _get_model_output(self, packet: str, packet_obj: dict[str, Any], dry_run: bool) -> str:
        """Get raw model text, handling dry-run/no-server fallback and retries."""
        if dry_run:
            return dry_run_policy(packet_obj)
        try:
            return self._call_with_timeout(self.system_prompt, packet, self.cfg, model=None)
        except LLMTransientError as exc:
            log.warning("LLM call failed trigger=%s detail=%s", exc.trigger, exc)
            llm = self.cfg.get("llm") or {}
            base = str(llm.get("base_url") or DEFAULT_LLM_CFG["base_url"])
            if exc.trigger in ("connection_error", "timeout") and llm.get("dry_run_on_no_server", True):
                from core.llama_server import is_healthy

                if not is_healthy(base if base.endswith("/v1") else base.rstrip("/") + "/v1"):
                    log.warning("llama-server unreachable; falling back to dry-run policy")
                    return dry_run_policy(packet_obj)
                log.warning(
                    "llama-server still healthy after %s — not treating as down",
                    exc.trigger,
                )
            fallback = self.cfg.get("fallback_model")
            primary = llm.get("model")
            if (
                fallback
                and fallback != primary
                and exc.trigger in self.fallback_triggers
            ):
                log.info("Retrying with fallback model %s after %s", fallback, exc.trigger)
                return self._call_with_timeout(self.system_prompt, packet, self.cfg, model=fallback)
            raise

    def _call_with_timeout(self, system: str, packet: str, cfg: dict[str, Any], *, model: str | None) -> str:
        # Stream tokens when a callback is set; still bound by timeout via a worker.
        timeout = int((cfg.get("llm") or {}).get("timeout_sec", DEFAULT_LLM_CFG["timeout_sec"]))
        # CPU prefill of a multi-k prompt can take minutes with no SSE bytes;
        # do not abort and call that "unreachable".
        if int((cfg.get("llm") or {}).get("num_gpu") or 0) <= 0:
            timeout = max(timeout, 900)
        on_token = getattr(self, "on_token", None)

        def _run() -> str:
            parts: list[str] = []
            for piece in call_llm_stream(system, packet, cfg, model=model, on_token=on_token):
                parts.append(piece)
            return "".join(parts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                return future.result(timeout=timeout + 15)
            except concurrent.futures.TimeoutError as exc:
                raise LLMTransientError("timeout") from exc

    def _parse_with_retry(self, raw: str, packet: str, packet_obj: dict[str, Any], dry_run: bool) -> tuple[str, dict[str, Any]]:
        try:
            return parse_model_output(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("Malformed model output: %s", exc)
            log.debug("Raw model output (truncated): %s", (raw or "")[:1200])
            if dry_run or "malformed_output" not in self.fallback_triggers:
                raise
            # Prefer a short repair pass over blindly re-running the full prompt
            # on the same small local model (which often repeats the failure).
            try:
                repaired = self._repair_model_output(raw, packet)
                return parse_model_output(repaired)
            except (LLMTransientError, ValueError, json.JSONDecodeError) as repair_exc:
                log.warning("Repair pass failed: %s", repair_exc)
            fallback = self.cfg.get("fallback_model")
            primary = (self.cfg.get("llm") or {}).get("model")
            if fallback and fallback != primary:
                log.info("Retrying with fallback model %s after malformed_output", fallback)
                raw = self._call_with_timeout(self.system_prompt, packet, self.cfg, model=fallback)
                return parse_model_output(raw)
            raise

    def _repair_model_output(self, bad_raw: str, packet: str) -> str:
        """Ask the model to emit a clean think+json block from its broken reply."""
        repair_system = (
            "You are a JSON repair assistant for an agent loop. "
            "Given a broken previous reply, output ONLY a valid "
            "<think>...</think> block then a ```json action object. "
            "Keep <think> under 400 characters. One action only. No commentary."
        )
        repair_packet = json.dumps(
            {
                "instruction": "Fix the previous reply into valid agent output.",
                "broken_reply": (bad_raw or "")[:2500],
                "original_packet_excerpt": packet[:1500],
            },
            indent=2,
        )
        # Temporarily clamp tokens for the repair call.
        llm = dict(self.cfg.get("llm") or {})
        cfg = dict(self.cfg)
        cfg["llm"] = {**llm, "max_tokens": min(int(llm.get("max_tokens", 800)), 600)}
        return self._call_with_timeout(repair_system, repair_packet, cfg, model=None)

    def _sanitize_memory_items(self, items: list[dict[str, Any]], last_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        """The model's own 'verified: true' claim is not proof.

        Only pass verified through when the current step's tool/skill result
        actually succeeded — i.e. there is something in hand to have verified
        it against. Otherwise downgrade to unverified and note why, rather
        than silently trusting self-reported certainty.
        """
        corroborated = bool(last_result and last_result.get("ok"))
        out = []
        for item in items or []:
            verified = bool(item.get("verified")) and corroborated
            if item.get("verified") and not verified:
                log.info("Downgrading unverified memory claim: %r", item.get("text"))
            out.append({**item, "verified": verified})
        return out

    def _finalize_run(self, goal: str, result: dict[str, Any]) -> dict[str, Any]:
        """Persist trace, run reflection, attach report — then reset trace."""
        try:
            save_last_trace(self.paths.db, goal=goal, result=result, trace=self._trace)
        except OSError as exc:
            log.warning("Could not save last trace: %s", exc)
        try:
            report = self.reflection.reflect(self._trace, result)
            for item in report.lessons:
                try:
                    self.memory.propose(item["text"], kind=item.get("kind", "lesson"))
                except (ValueError, OSError) as exc:
                    log.warning("Could not propose reflection lesson: %s", exc)
            result = {
                **result,
                "reflection": report.run_analysis,
                "reflection_summary": report.one_line_summary(),
            }
            try:
                result["archive"] = archive_stale_skills(self.evolution, self.skills, self.paths.db)
            except Exception as exc:  # noqa: BLE001
                log.warning("Season pass failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — never fail the run on reflection
            log.warning("Reflection failed: %s", exc)
            result = {**result, "reflection_error": str(exc)}
        result = {
            **result,
            "run_state": self.world.snapshot(),
            "event_bus": {"components": self.events.component_names, "events": len(self.events.trace)},
            "steer": dict(self.steering.last),
            "judgment": dict(self._last_judgment),
            "plan_health": plan_health(self.memory.wm.active_plan),
        }
        self._trace = []
        try:
            idle_rest(self.paths)
        except Exception as exc:  # noqa: BLE001
            log.warning("idle maintenance failed: %s", exc)
        return result

    def _observe(self, *, step: int, source: str, content: str, ok: bool = True) -> None:
        self.events.fire(
            Event(
                channel="input",
                kind="observe",
                payload={"step": step, "source": source, "content": content, "ok": ok},
                step=step,
                source=source,
            )
        )

    def _commit_memories(
        self, items: list[dict[str, Any]], last_result: dict[str, Any] | None
    ) -> list[str]:
        """Proposals first. Promotion only after corroboration (the review)."""
        cleaned = self._sanitize_memory_items(items, last_result)
        review = bool(self.cfg.get("memory_promotion_requires_review", True))
        ids: list[str] = []
        for item in cleaned:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            kind = str(item.get("kind") or "lesson")
            if review:
                eid = self.memory.propose(text, kind=kind)
                if item.get("verified") and eid:
                    self.memory.promote(eid)
            else:
                eid = self.memory.save(text, kind=kind, verified=bool(item.get("verified")))
            if eid:
                ids.append(eid)
        return ids

    def _ask_permission(self, request: dict[str, Any]) -> str:
        """Return allow or deny. Hostile payloads never reach the human callback."""
        ok, request, err = inspect_request(request)
        if not ok or request is None:
            log.error("REJECT loop/permission %s", err)
            return DENY
        key = permission_key(request)
        if key in self._run_allowed:
            return ALLOW
        cb = self.on_permission
        if cb is None:
            return ALLOW
        try:
            raw = cb(request)
        except Exception as exc:  # noqa: BLE001
            log.warning("on_permission raised: %s", exc)
            return DENY
        decision = normalize_decision(raw)
        if decision == ALLOW_RUN:
            self._run_allowed.add(key)
            return ALLOW
        if decision == ALLOW:
            return ALLOW
        return DENY

    def _resume_confirmation(self, action: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Prompt the human, then either dispatch the intended tool or continue the loop."""
        req = action.get("confirmation_request") or {}
        if not isinstance(req, dict):
            req = {}
        tool = action.get("tool") or {}
        if not isinstance(tool, dict):
            tool = {}
        name = str(tool.get("name") or req.get("action_type") or "confirmation")
        args = tool.get("args") if isinstance(tool.get("args"), dict) else {}
        if not args and req.get("target"):
            args = {"target": req.get("target")}
        request = {
            "kind": "confirmation",
            "name": name,
            "args": args,
            "description": str(req.get("description") or action.get("rationale") or "Needs confirmation"),
            "target": str(req.get("target") or ""),
            "action_type": str(req.get("action_type") or ""),
        }
        decision = self._ask_permission(request)
        if decision != ALLOW:
            log.info("confirmation denied: %s", request)
            return denied_result("request_confirmation", extra=dict(req)), "request_confirmation"
        if tool.get("name"):
            log.info("confirmation granted; dispatching tool %s", tool.get("name"))
            return self.tools.call(str(tool.get("name") or ""), args), "use_tool"
        return {
            "ok": True,
            "from": "request_confirmation",
            "confirmed": True,
            "observed": dict(req) if req else {"confirmed": True},
        }, "request_confirmation"

    def _check_think(self, think: str, *, dry_run: bool) -> dict[str, Any]:
        verdict = evaluate_think(
            think,
            min_chars=int(self.cfg.get("think_block_min_chars") or 20),
            require_block=bool(self.cfg.get("require_think_block", True)),
            require_labels=bool(self.cfg.get("think_block_must_cite_source", True)) and not dry_run,
        )
        if not verdict["ok"]:
            self._weak_think = True
            self.events.fire(
                Event(
                    channel="control",
                    kind="weak_think",
                    payload=verdict,
                    target="world",
                ),
                target="world",
            )
            log.info("low light: %s", verdict.get("issues"))
        else:
            self._weak_think = False
        return verdict

    def run(self, goal: str, dry_run: bool = False, max_steps: int | None = None) -> dict[str, Any]:
        goal = str(goal or "").strip()
        if not goal:
            return {"ok": False, "error": "empty_goal", "steps": 0}
        if len(goal) > MAX_GOAL_CHARS:
            log.error("REJECT loop/empty_goal oversized_goal")
            return {"ok": False, "error": "args_too_large", "steps": 0}
        if max_steps is not None:
            try:
                max_steps = int(max_steps)
            except (TypeError, ValueError):
                return {"ok": False, "error": "invalid_max_steps", "steps": 0}
            if max_steps < 1:
                return {"ok": False, "error": "invalid_max_steps", "steps": 0}
        self.memory.seed_goal(goal)
        self._trace = []
        self._weak_think = False
        self._last_judgment = {"verdict": "unknown", "reason": "no act yet"}
        self._run_allowed = set()
        self.cancel_requested = False
        self._dry_run = bool(dry_run)
        self.world = WorldState.from_config(self.cfg, identity_hash=self.brain.identity_hash)
        self.events.attach(self.world)
        self._observe(step=0, source="user", content=goal, ok=True)
        configured_cap = int(self.cfg.get("max_steps", 50))
        action_cap = self.cfg.get("max_actions_per_run")
        if action_cap is not None:
            configured_cap = min(configured_cap, int(action_cap))
        effective_max_steps = min(configured_cap, max_steps) if max_steps else configured_cap
        self.world.energy = float(effective_max_steps)
        llm_cfg = self.cfg.get("llm") or {}
        num_ctx = int(llm_cfg.get("num_ctx") or 4096)
        max_tokens = int(llm_cfg.get("max_tokens") or 800)

        if not dry_run:
            warmup_model(self.cfg)

        for step in range(1, effective_max_steps + 1):
            if self.cancel_requested:
                return self._finalize_run(
                    goal,
                    {
                        "ok": False,
                        "error": "cancelled",
                        "status": "cancelled",
                        "steps": max(0, step - 1),
                    },
                )
            self.memory.wm.step = step
            try:
                live_hash = self.brain.fingerprint_on_disk(self.paths)
            except (OSError, FileNotFoundError) as exc:
                log.error("brain identity unreadable: %s", exc)
                live_hash = ""
            identity_replies = self.events.fire(
                Event(
                    channel="control",
                    kind="identity_check",
                    payload={"hash": live_hash},
                    step=step,
                    target="world",
                ),
                target="world",
            )
            if any(r.kind == "conservation_violation" for r in identity_replies):
                return self._finalize_run(
                    goal,
                    {
                        "ok": False,
                        "error": "identity_conservation_violated",
                        "steps": step,
                    },
                )
            if self.world.at_ground_state:
                self.memory.save(
                    f"Run reached ground state after {step - 1} steps.",
                    kind="failure",
                    verified=True,
                )
                return self._finalize_run(
                    goal,
                    {
                        "ok": False,
                        "error": "max_steps_exceeded",
                        "steps": max(0, step - 1),
                        "evolution": self.evolution.leaderboard(),
                    },
                )
            self.events.fire(
                Event(channel="control", kind="tick", step=step, energy_cost=self.world.energy_cost, target="world"),
                target="world",
            )
            catalog = self.skills.list_skills(goal, k=self.cfg.get("skill_topk", 8), include_drafts=True)
            should, reason = self.evolution.should_create_skill(
                goal, [s.get("name", "") for s in catalog], self.memory.wm.last_result
            )
            if should:
                self.evolution.record_gap(goal, reason)
            packet = build_user_packet(
                goal=goal,
                user_input=goal if step == 1 else "",
                memory=self.memory,
                tools=self.tools.list_filtered(
                    goal, k=self.cfg.get("tool_topk", 8), beliefs=self._belief_map()
                ),
                skills=catalog,
                memory_topk=int(self.cfg.get("memory_topk", 8)),
                world=self.world,
                system=self.system_prompt,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                max_packet_tokens=1200 if int((self.cfg.get("llm") or {}).get("num_gpu") or 0) <= 0 else None,
                weak_think=self._weak_think,
                named=self.world_model.recall(goal, k=8),
                steer=self.steering.last,
                judgment=self._last_judgment,
                evolution={
                    "should_create": should,
                    "reason": reason,
                    "leaderboard": self.evolution.leaderboard(),
                    "gaps": self.evolution.gaps[-5:],
                },
                step=step,
                paths=self.paths,
            )
            # Packet may be wrapped with reasoning lenses + anti-echo instructions.
            json_blob = packet
            for marker in ("PACKET:\n", "STATE_PACKET_JSON:\n"):
                if marker in packet:
                    json_blob = packet.split(marker, 1)[1]
                    break
            packet_obj = json.loads(json_blob)

            try:
                raw = self._get_model_output(packet, packet_obj, dry_run)
                think, action = self._parse_with_retry(raw, packet, packet_obj, dry_run)
                self._check_think(think, dry_run=dry_run)
            except (LLMTransientError, ValueError, json.JSONDecodeError) as exc:
                log.warning("Step %d: model output unusable (%s) — synthesizing loop action", step, exc)
                think = ""
                last = self.memory.wm.last_result
                if step >= 2 and isinstance(last, dict) and last.get("ok"):
                    tools = [
                        t.get("name")
                        for t in (packet_obj.get("tool_registry") or [])
                        if isinstance(t, dict) and t.get("name")
                    ]
                    tool_list = ", ".join(str(n) for n in tools[:12]) or "(see registry)"
                    action = {
                        "action": "finish",
                        "rationale": f"synthesized finish after unusable output ({exc})",
                        "finish": {
                            "status": "success",
                            "summary": f"Hello — I'm s0uRc3, the Builder Agent. Tools: {tool_list}.",
                            "artifacts": [],
                        },
                    }
                else:
                    action = {
                        "action": "use_tool",
                        "rationale": f"synthesized continue after unusable output ({exc})",
                        "tool": {"name": "list_dir", "args": {"path": ".", "glob": "*"}},
                    }

            kind = action.get("action")
            if not isinstance(kind, str) or kind not in self.allowed_actions:
                # 1B models sometimes put the user goal / a sentence in "action".
                if isinstance(kind, str) and kind.strip() and kind not in self.allowed_actions:
                    last = self.memory.wm.last_result
                    mode = goal_mode(goal)
                    # Chat (or post-tool): treat free-text as the user-facing reply.
                    if mode == "chat" or (step >= 2 and isinstance(last, dict) and last.get("ok")):
                        summary = kind.strip()[:800]
                        if mode == "action" and isinstance(last, dict) and last.get("ok"):
                            tools = [
                                t.get("name")
                                for t in (packet_obj.get("tool_registry") or [])
                                if isinstance(t, dict) and t.get("name")
                            ]
                            tool_list = ", ".join(str(n) for n in tools[:12])
                            if tool_list and "tool" in goal.lower():
                                summary = f"{summary}\n\nTools: {tool_list}"
                        log.warning("Step %d: free-text → finish (%s mode)", step, mode)
                        action = {
                            "action": "finish",
                            "rationale": "normalized free-text reply",
                            "finish": {"status": "success", "summary": summary, "artifacts": []},
                            "memory_to_save": action.get("memory_to_save") or [],
                        }
                        kind = "finish"
                    else:
                        log.warning("Step %d: normalizing free-text action %r → use_tool", step, kind)
                        action = {
                            "action": "use_tool",
                            "rationale": f"normalized free-text action ({kind[:80]!r})",
                            "tool": {"name": "list_dir", "args": {"path": ".", "glob": "*"}},
                            "memory_to_save": action.get("memory_to_save") or [],
                        }
                        kind = "use_tool"
                else:
                    log.error("Step %d: model emitted disallowed action %r", step, kind)
                    self.memory.save(f"Rejected disallowed action: {kind!r}", kind="failure", verified=True)
                    return self._finalize_run(
                        goal,
                        {
                            "ok": False,
                            "error": f"disallowed_action: {kind}",
                            "steps": step,
                            "evolution": self.evolution.leaderboard(),
                        },
                    )

            self.memory.wm.note(f"think:{think[:200]}")

            if kind == "request_confirmation":
                log.info("Step %d: requesting human confirmation", step)
                result, kind = self._resume_confirmation(action)
            else:
                result = self._dispatch(action)
            self.memory.wm.last_result = result
            self._last_judgment = judge(result if isinstance(result, dict) else None)
            source = "tool:" + str((action.get("tool") or {}).get("name") or kind)
            if kind == "finish":
                source = "finish"
            elif kind == "create_skill":
                source = "skill:" + str((action.get("skill") or {}).get("name") or "draft")
            elif kind == "update_plan":
                source = "plan"
            elif kind == "save_memory":
                source = "memory"
            elif kind == "request_confirmation":
                source = "confirmation"
            cone_text = observed_text(result if isinstance(result, dict) else None, fallback=kind)
            self._observe(step=step, source=source, content=cone_text or kind, ok=bool(result.get("ok")))
            if kind == "use_tool":
                self._record_tool_belief(str((action.get("tool") or {}).get("name") or ""), bool(result.get("ok")))
            self.events.fire(
                Event(
                    channel="output",
                    kind=str(kind),
                    payload={"ok": bool(result.get("ok"))},
                    step=step,
                    source="loop",
                )
            )
            tool_name = str((action.get("tool") or {}).get("name") or "") if kind == "use_tool" else ""
            self._trace.append(
                TraceStep(
                    step=step,
                    action=kind,
                    rationale=str(action.get("rationale", "")),
                    result_ok=bool(result.get("ok")),
                    result_error=str(result.get("error", "")),
                    think_snippet=think[:200],
                    drafts_used=self.drafts_this_run,
                    approved_skills=len([s for s in catalog if not s.get("draft")]),
                    tool_name=tool_name,
                )
            )
            steer = self.steering.evaluate(
                self._trace,
                result if isinstance(result, dict) else None,
                beliefs=self._belief_map(),
                plan_health=plan_health(self.memory.wm.active_plan),
            )
            self.events.fire(
                Event(channel="feedback", kind="steer", payload=steer, step=step, source="steering")
            )
            if steer.get("steer") == "stop_spin":
                log.warning("Step %d: sky stop_spin — %s", step, steer.get("reason"))
                finish = self._complete_finish(
                    {
                        "status": "blocked",
                        "summary": str(steer.get("reason") or "repeated failing tool"),
                    },
                    goal=goal,
                    last_result=result if isinstance(result, dict) else None,
                )
                return self._finalize_run(
                    goal,
                    {
                        "ok": False,
                        "error": "stop_spin",
                        "status": "blocked",
                        "steps": step,
                        "reason": steer.get("reason"),
                        "finish": finish,
                        "evolution": self.evolution.leaderboard(),
                    },
                )
            if steer.get("steer") == "rest" and kind != "finish":
                log.info("Step %d: rest — %s", step, steer.get("reason"))
                finish = self._complete_finish(
                    {
                        "status": "success",
                        "summary": observed_text(result if isinstance(result, dict) else None)
                        or str(steer.get("reason") or "resting"),
                    },
                    goal=goal,
                    last_result=result if isinstance(result, dict) else None,
                )
                return self._finalize_run(
                    goal,
                    {
                        "ok": True,
                        "steps": step,
                        "status": "success",
                        "reason": steer.get("reason"),
                        "finish": finish,
                        "evolution": self.evolution.leaderboard(),
                    },
                )

            # Anti-spin: 1B models often repeat use_tool forever. For simple
            # greeting / "list your tools" goals, one successful tool step is enough.
            if (
                kind == "use_tool"
                and result.get("ok")
                and _is_simple_chat_goal(goal)
            ):
                tools = [
                    t.get("name")
                    for t in (packet_obj.get("tool_registry") or [])
                    if isinstance(t, dict) and t.get("name")
                ]
                tool_list = ", ".join(str(n) for n in tools[:12]) or "(see registry)"
                summary = f"Hello — I'm s0uRc3, the Builder Agent. Tools: {tool_list}."
                log.info("Step %d: simple-chat goal satisfied after tool — finishing", step)
                finish = self._complete_finish(
                    {"status": "success", "summary": summary},
                    goal=goal,
                    last_result=result if isinstance(result, dict) else None,
                )
                return self._finalize_run(
                    goal,
                    {
                        "ok": True,
                        "steps": step,
                        "result": result,
                        "finish": finish,
                        "evolution": self.evolution.leaderboard(),
                    },
                )

            if kind == "finish":
                # Guard: tiny models love to finish on turn 1. If the goal still
                # needs work and we have no tool measurement yet, keep looping.
                finish_body = action.get("finish") or {}
                has_tool = any(
                    str(getattr(o, "source", "")).startswith("tool:") for o in self.world.observations
                )
                needs_work = step == 1 and not has_tool and goal_mode(goal) == "action"
                if needs_work:
                    log.warning("Step %d: rejected premature finish — continuing loop", step)
                    action = {
                        "action": "use_tool",
                        "rationale": "premature finish blocked; inspect first",
                        "tool": {"name": "list_dir", "args": {"path": ".", "glob": "*"}},
                    }
                    kind = "use_tool"
                    result = self._dispatch(action)
                    self.memory.wm.last_result = result
                    self._observe(
                        step=step,
                        source="tool:list_dir",
                        content=observed_text(result if isinstance(result, dict) else None, fallback="list_dir"),
                        ok=bool(result.get("ok")),
                    )
                    self._last_judgment = judge(result if isinstance(result, dict) else None)
                    if self._trace:
                        self._trace[-1] = TraceStep(
                            step=step,
                            action=kind,
                            rationale=str(action.get("rationale", "")),
                            result_ok=bool(result.get("ok")),
                            result_error=str(result.get("error", "")),
                            think_snippet=think[:200],
                            drafts_used=self.drafts_this_run,
                            approved_skills=len([s for s in catalog if not s.get("draft")]),
                            tool_name="list_dir",
                        )
                    continue

                self._commit_memories(action.get("memory_to_save") or [], result)
                finish = self._complete_finish(
                    finish_body,
                    goal=goal,
                    last_result=result if isinstance(result, dict) else None,
                )
                return self._finalize_run(
                    goal,
                    {
                        "ok": finish.get("status") != "failed",
                        "steps": step,
                        "result": result,
                        "finish": finish,
                        "evolution": self.evolution.leaderboard(),
                    },
                )

        self.memory.propose(
            f"Run exhausted its {effective_max_steps}-step budget without finishing.",
            kind="failure",
        )
        return self._finalize_run(
            goal,
            {
                "ok": False,
                "error": "max_steps_exceeded",
                "steps": effective_max_steps,
                "evolution": self.evolution.leaderboard(),
            },
        )

    def _refine_code(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Generate → test → revise. Tests decide pass/fail; the model only revises."""
        name = str(spec.get("skill_name") or "").strip()
        if not name or "\x00" in name:
            return {"ok": False, "error": "invalid_skill_name", "from": "refine_code", "observed": {}}
        try:
            max_passes = int(spec.get("max_passes") or self.cfg.get("refine_max_passes") or DEFAULT_REFINE_MAX_PASSES)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_max_passes", "from": "refine_code", "observed": {}}
        max_passes = max(1, min(max_passes, 8))

        try:
            path, code = self.skills.get_code(name)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "from": "refine_code", "observed": {}}
        if not code or path is None:
            return {"ok": False, "error": f"skill not found: {name!r}", "from": "refine_code", "observed": {}}
        safe = path.stem
        draft_path = self.skills.drafts / f"{safe}.py"
        name = safe
        if path.parent != self.skills.drafts or not draft_path.exists():
            draft_path.write_text(code, encoding="utf-8")

        prev_sig: str | None = None
        failures: list[dict[str, Any]] = []
        for pass_n in range(1, max_passes + 1):
            try:
                code = draft_path.read_text(encoding="utf-8")
            except OSError as exc:
                return {"ok": False, "error": f"could not read draft: {exc}", "from": "refine_code", "observed": {}}
            tested = self.skills.run_skill_tests(name, code)
            failures = list(tested.get("failures") or [])
            sig = _failure_signature(tested.get("test_results") or [])
            passed = bool(tested.get("ok"))
            self.evolution.record_refine(name, pass_n, passed, sig)
            log.info("refine_code %s pass=%s passed=%s sig=%s", name, pass_n, passed, sig[:120])
            if passed:
                dest = self.skills.approved / f"{name}.py"
                if draft_path.exists():
                    import shutil

                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(draft_path), str(dest))
                return {
                    "ok": True,
                    "from": "refine_code",
                    "name": name,
                    "passes_used": pass_n,
                    "promoted": dest.exists(),
                    "observed": {"passes_used": pass_n, "name": name},
                }
            if prev_sig is not None and sig == prev_sig:
                return {
                    "ok": False,
                    "error": "no_progress",
                    "from": "refine_code",
                    "failures": failures,
                    "passes_used": pass_n,
                    "observed": {"error": "no_progress", "failures": failures},
                }
            prev_sig = sig
            if pass_n >= max_passes:
                break
            if self._dry_run:
                return {
                    "ok": False,
                    "error": "dry_run_no_revise",
                    "from": "refine_code",
                    "failures": failures,
                    "passes_used": pass_n,
                    "observed": {"error": "dry_run_no_revise"},
                }
            revised = self._revise_skill_code(code, failures)
            if not revised:
                return {
                    "ok": False,
                    "error": "no_progress",
                    "from": "refine_code",
                    "failures": failures,
                    "passes_used": pass_n,
                    "observed": {"error": "empty_revision"},
                }
            draft_path.write_text(revised, encoding="utf-8")

        return {
            "ok": False,
            "error": "max_passes_exceeded",
            "from": "refine_code",
            "failures": failures,
            "passes_used": max_passes,
            "observed": {"error": "max_passes_exceeded", "failures": failures},
        }

    def _revise_skill_code(self, code: str, failures: list[dict[str, Any]]) -> str:
        """Ask the reviser for new source. It is not asked to grade the code."""
        packet = json.dumps({"current_code": code, "failures": failures}, indent=2, default=str)
        saved = self.on_token
        self.on_token = None
        try:
            raw = self._call_with_timeout(REVISE_SYSTEM, packet, self.cfg, model=None)
        finally:
            self.on_token = saved
        return _extract_revised_code(raw)

    def _dispatch(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = action.get("action")
        if kind == "use_tool":
            tool = action.get("tool") or {}
            if not isinstance(tool, dict):
                return {"ok": False, "error": "args_must_be_object", "from": "use_tool", "observed": {}}
            name = str(tool.get("name") or "")
            args = tool.get("args")
            if args is None or not isinstance(args, dict):
                log.error("REJECT loop/use_tool args_must_be_object")
                return {"ok": False, "error": "args_must_be_object", "from": name or "use_tool", "observed": {}}
            decision = self._ask_permission(
                {
                    "kind": "use_tool",
                    "name": name,
                    "args": args,
                    "description": f"Run tool {name}" if name else "Run a tool",
                    "target": json.dumps(args, default=str)[:400],
                }
            )
            if decision != ALLOW:
                log.info("tool denied: %s %s", name, args)
                return denied_result(name or "use_tool", extra={"args": args})
            return self.tools.call(name, args)
        if kind == "create_skill":
            spec = action.get("skill") or {}
            skill_name = str(spec.get("name") or "unnamed")
            decision = self._ask_permission(
                {
                    "kind": "create_skill",
                    "name": skill_name,
                    "args": {"description": spec.get("description")},
                    "description": f"Create skill draft {skill_name}",
                    "target": skill_name,
                }
            )
            if decision != ALLOW:
                log.info("create_skill denied: %s", skill_name)
                return denied_result("create_skill", extra={"name": skill_name})
            cap = int(self.cfg.get("max_skill_drafts_per_run", 3))
            if self.drafts_this_run >= cap:
                return {"ok": False, "error": "draft cap reached"}
            code = spec.get("code") or _default_skill_code(spec)
            self.drafts_this_run += 1
            created = self.skills.create_draft(
                spec.get("name", "unnamed"),
                code,
                kind=spec.get("kind"),
            )
            self.evolution.record_created(created.get("name") or spec.get("name", "unnamed"), bool(created.get("promoted")))
            return created
        if kind == "refine_code":
            spec = action.get("refine") or {}
            if not isinstance(spec, dict):
                return {"ok": False, "error": "args_must_be_object", "from": "refine_code", "observed": {}}
            skill_name = str(spec.get("skill_name") or "")
            decision = self._ask_permission(
                {
                    "kind": "refine_code",
                    "name": skill_name,
                    "args": {"skill_name": skill_name},
                    "description": f"Revise skill draft {skill_name} against its TESTS",
                    "target": skill_name,
                }
            )
            if decision != ALLOW:
                log.info("refine_code denied: %s", skill_name)
                return denied_result("refine_code", extra={"name": skill_name})
            return self._refine_code(spec)
        if kind == "update_plan":
            raw_nodes = (action.get("plan") or {}).get("nodes")
            nodes = normalize_plan(raw_nodes if isinstance(raw_nodes, list) else self.memory.wm.active_plan)
            self.memory.wm.active_plan = nodes
            health = plan_health(nodes)
            return {
                "ok": True,
                "observed": {"node_ids": [n["id"] for n in nodes], "in_progress": health["in_progress"]},
                "plan": nodes,
                "plan_health": health,
            }
        if kind in ("save_memory", "propose_long_term_memory"):
            saved = self._commit_memories(
                action.get("memory_to_save") or [],
                self.memory.wm.last_result if isinstance(self.memory.wm.last_result, dict) else None,
            )
            return {"ok": True, "ids": saved, "proposed": True}
        if kind == "finish":
            return {"ok": True, "from": "finish", **(action.get("finish") or {})}
        return {"ok": False, "error": f"unknown action {kind}"}


def _failure_signature(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in results:
        if row.get("passed"):
            continue
        parts.append(f"{row.get('test')}:{(row.get('error') or '')[:160]}")
    return "|".join(parts)


def _extract_revised_code(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _default_skill_code(spec: dict[str, Any]) -> str:
    name = spec.get("name") or "unnamed"
    desc = spec.get("description") or ""
    kind = spec.get("kind") or "transform"
    return f'''"""
SKILL MANIFEST
name: {name}
description: {desc}
kind: {kind}
inputs: {json.dumps(spec.get("inputs") or {})}
outputs: {json.dumps(spec.get("outputs") or {})}
dependencies: []
author: self
version: 1
"""

def run(**kwargs):
    return {{"ok": True, "echo": kwargs}}
'''
