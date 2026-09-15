"""core/tool_formatter.py — second-pass tool-arg formatter.

pocket (the reasoning model) decides WHICH tool to call; a second, tiny
model purpose-built for this — google/functiongemma-270m-it, a Gemma-3
derivative Google trained specifically to NOT be a dialogue model but a
function-call formatter — fills in the ARGS. pocket's own arg-filling is
a known weak point for a model this small (core/loop.py already
hand-corrects read_file/write_file's path/content for the same reason,
see the "Small local models cannot reliably fill this argument" comment
there); this generalizes that fix to every tool instead of two
hardcoded ones.

Runs as its own llama-server instance on a separate port
(brain/reasoning_config.json :: tool_formatter / llama_server_tool_formatter).
Best-effort only: any failure (server not running, bad parse, tool name
mismatch) returns None and the caller keeps pocket's own args untouched —
this must never block or fail a turn.

Output format confirmed empirically against a live
ggml-org/functiongemma-270m-it-GGUF (bf16) instance, since the base
model card documents the shape but not the exact escaping rules:

    <start_function_call>call:read_file{max_chars:500,path:<escape>workspace/notes.txt<escape>}

String args are wrapped in <escape>...<escape> (may contain commas,
braces, newlines — anything but the literal token "<escape>"); numeric
and boolean args are bare tokens. llama-server did not populate the
OpenAI-style `tool_calls` field for this brand-new model family in
testing, so this module parses the raw `content` text directly rather
than relying on that field (checked first anyway, in case a future
llama.cpp version adds native support — cheap to keep).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("agent.tool_formatter")

_CALL_RE = re.compile(r"call:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\{(.*)\}", re.DOTALL)
_ESCAPED_ARG_RE = re.compile(r"(\w+)\s*:\s*<escape>(.*?)<escape>", re.DOTALL)

_JSON_TYPES = {"string", "integer", "number", "boolean", "object", "array"}


def args_schema_to_json_schema(tool_entry: dict[str, Any]) -> dict[str, Any]:
    """Convert this codebase's tools/manifest.json arg shape —
    `{"path": {"type": "string", "required": true}}` — into an
    OpenAI-style JSON Schema `parameters` object."""
    raw_args = tool_entry.get("args") if isinstance(tool_entry, dict) else None
    props: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(raw_args, dict):
        for name, spec in raw_args.items():
            if not isinstance(spec, dict):
                continue
            t = str(spec.get("type") or "string")
            if t not in _JSON_TYPES:
                t = "string"
            props[name] = {"type": t}
            if spec.get("required"):
                required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _coerce(value: str, arg_type: str) -> Any:
    value = value.strip()
    if arg_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if arg_type == "number":
        try:
            return float(value)
        except ValueError:
            return value
    if arg_type == "boolean":
        return value.lower() in ("true", "1", "yes")
    return value


def _parse_function_call(text: str, *, arg_types: dict[str, str]) -> dict[str, Any] | None:
    """Parse `call:name{...}` out of raw model output. Escaped string args
    are pulled out first (safe against commas/braces inside their value);
    whatever's left in the body is comma-separated bare key:value pairs."""
    m = _CALL_RE.search(text or "")
    if not m:
        return None
    name = m.group(1)
    body = m.group(2)
    # A trailing <end_function_call> only shows up if it wasn't used as a
    # stop sequence — strip it defensively either way.
    body = body.rsplit("<end_function_call>", 1)[0]

    args: dict[str, Any] = {}
    remaining = body
    for am in _ESCAPED_ARG_RE.finditer(body):
        args[am.group(1)] = am.group(2)
        remaining = remaining.replace(am.group(0), "", 1)

    for part in remaining.split(","):
        part = part.strip().strip("}").strip()
        if not part or ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip()
        if not key or key in args:
            continue
        args[key] = _coerce(val, arg_types.get(key, "string"))

    return {"name": name, "args": args}


def _post(base_url: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_args(
    *,
    cfg: dict[str, Any],
    tool_entry: dict[str, Any],
    goal: str,
    hints: dict[str, str] | None = None,
    pocket_args: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Ask functiongemma to fill in this tool's args from the goal text.

    Returns None on any failure (formatter disabled, server unreachable,
    unparseable output, name mismatch) — the caller must keep pocket's own
    args in that case. Never raises.
    """
    fc = cfg.get("tool_formatter") or {}
    if not fc.get("enabled"):
        return None
    tool_name = str(tool_entry.get("name") or "")
    if not tool_name:
        return None

    raw_args = tool_entry.get("args") if isinstance(tool_entry.get("args"), dict) else {}
    arg_types = {
        k: str((v or {}).get("type") or "string")
        for k, v in raw_args.items()
        if isinstance(v, dict)
    }
    parameters = args_schema_to_json_schema(tool_entry)
    description = str(tool_entry.get("description") or f"Call {tool_name}.")[:200]

    lines = [goal.strip()]
    for k, v in (hints or {}).items():
        if v:
            lines.append(f"{k}: {v}")
    if pocket_args:
        try:
            lines.append(f"draft_args: {json.dumps(pocket_args, default=str)[:300]}")
        except (TypeError, ValueError):
            pass

    payload = {
        "model": fc.get("model", "functiongemma"),
        "messages": [{"role": "user", "content": "\n".join(lines)}],
        "tools": [
            {
                "type": "function",
                "function": {"name": tool_name, "description": description, "parameters": parameters},
            }
        ],
        "temperature": float(fc.get("temperature", 0.0)),
        "max_tokens": int(fc.get("max_tokens", 200)),
        "stop": ["<end_function_call>"],
    }
    base_url = str(fc.get("base_url") or "http://127.0.0.1:8081/v1")
    api_key = str(fc.get("api_key") or "sk-no-key-needed")
    timeout = int(fc.get("timeout_sec", 20))

    try:
        data = _post(base_url, payload, api_key, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("tool_formatter call failed for %s: %s", tool_name, exc)
        return None

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        log.warning("tool_formatter returned unexpected shape for %s: %r", tool_name, data)
        return None
    if not isinstance(message, dict):
        return None

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        try:
            fn = tool_calls[0]["function"]
            call_name = str(fn.get("name") or "")
            raw = fn.get("arguments")
            parsed_args = json.loads(raw) if isinstance(raw, str) else raw
            if call_name == tool_name and isinstance(parsed_args, dict):
                return parsed_args
        except (KeyError, TypeError, json.JSONDecodeError):
            pass

    content = str(message.get("content") or "")
    parsed = _parse_function_call(content, arg_types=arg_types)
    if parsed and parsed["name"] == tool_name:
        return parsed["args"]
    log.warning("tool_formatter: could not parse a %s call from output: %r", tool_name, content[:300])
    return None
