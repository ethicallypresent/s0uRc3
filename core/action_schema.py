"""core/action_schema.py — JSON Schema for grammar-constrained turn output.

Sent as response_format.json_schema on llama.cpp's /v1/chat/completions
when structured_output is enabled (brain/reasoning_config.json). The
server masks sampling to tokens that keep the output schema-valid, which
eliminates the entire class of malformed-output failures core/loop.py's
free-text <think>...</think>{json} path had to repair/salvage around:
truncated or absent JSON, the model echoing its own input packet back,
a repetition loop that never reaches an action — all structurally
impossible once decoding is grammar-constrained. Confirmed live against
a running pocket.gguf instance before this was wired in: 9/9 requests
came back as clean, schema-valid JSON, including the exact phrasings
that previously broke the free-text path.

The shape mirrors the existing internal action dict exactly (see
brain/system_prompt.full.md's worked JSON schema) plus one added field,
`think`, holding what used to be free text inside <think>...</think> —
folding reasoning into the schema itself removes the need for a second
tag-parsing pass, and lets core/loop.py's existing think-quality checks
(_check_think / evaluate_think) run unchanged against a plain string.

`tool.args`, `skill.inputs` and `skill.outputs` are left as open objects
(no `properties` declared) rather than JSON-encoded strings — confirmed
live that llama.cpp's schema-to-grammar conversion handles a genuinely
open `{"type": "object"}` property without issue, so there's no need for
the arguments-as-a-JSON-string workaround some APIs use to dodge this.
"""

from __future__ import annotations

import json
from typing import Any

ACTIONS = (
    "use_tool", "create_skill", "refine_code", "update_plan",
    "save_memory", "request_confirmation", "finish",
)
PLAN_STATUSES = ("pending", "in_progress", "done", "blocked", "failed")
MEMORY_KINDS = ("fact", "lesson", "preference", "failure", "constraint")
CONFIRMATION_TYPES = (
    "delete_file", "overwrite_file", "move_file",
    "send_external_message", "payment", "irreversible_api_call",
)
FINISH_STATUSES = ("success", "blocked", "failed")


def build_schema() -> dict[str, Any]:
    """Every allowed_actions variant in one flat object — variant-specific
    fields are all optional at the top level (only their own inner keys
    are required once present). `think` is optional: act tier is told to
    leave it empty, glance/deliberate are told to fill it."""
    return {
        "type": "object",
        "properties": {
            "think": {"type": "string"},
            "rationale": {"type": "string"},
            "action": {"type": "string", "enum": list(ACTIONS)},
            "tool": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["name", "args"],
            },
            "skill": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "inputs": {"type": "object"},
                    "outputs": {"type": "object"},
                    "code": {"type": "string"},
                },
                "required": ["name", "description", "code"],
            },
            "refine": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "max_passes": {"type": "integer"},
                },
                "required": ["skill_name"],
            },
            "plan": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                                "note": {"type": "string"},
                            },
                            "required": ["id", "title", "status"],
                        },
                    },
                },
                "required": ["nodes"],
            },
            "memory_to_save": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(MEMORY_KINDS)},
                        "text": {"type": "string"},
                        "verified": {"type": "boolean"},
                    },
                    "required": ["kind", "text"],
                },
            },
            "confirmation_request": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": list(CONFIRMATION_TYPES)},
                    "description": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["action_type", "description"],
            },
            "finish": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": list(FINISH_STATUSES)},
                    "summary": {"type": "string"},
                    "artifacts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["status", "summary"],
            },
        },
        "required": ["action", "rationale"],
    }


def response_format() -> dict[str, Any]:
    """The full response_format payload for llama.cpp's /v1/chat/completions."""
    return {"type": "json_schema", "json_schema": {"name": "agent_action", "schema": build_schema()}}


def parse_structured_output(raw: str) -> tuple[str, dict[str, Any]]:
    """Parse one schema-constrained completion into (think, action) — the
    same return shape as core/loop.py's free-text parse_model_output, so
    callers don't need to know which path produced it.

    Raises ValueError/json.JSONDecodeError on anything unparseable, same
    as parse_model_output, so a truly unexpected response (backend ignored
    response_format, connection hiccup mid-stream) still plugs into the
    existing malformed-output fallback/repair path unchanged.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model output")
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a json object, got {type(obj).__name__}")
    if "action" not in obj:
        raise ValueError("no action field in structured output")
    think = str(obj.get("think") or "")
    return think, obj
