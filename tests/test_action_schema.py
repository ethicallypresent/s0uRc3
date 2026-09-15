"""core/action_schema.py — the response_format schema + its parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.action_schema import ACTIONS, build_schema, parse_structured_output, response_format


def test_response_format_wraps_a_json_schema_with_the_full_action_shape():
    rf = response_format()
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert set(ACTIONS) == set(schema["properties"]["action"]["enum"])
    assert "think" in schema["properties"]
    assert "action" in schema["required"] and "rationale" in schema["required"]
    assert "think" not in schema["required"]  # act tier must be able to omit it


def test_schema_is_json_serializable():
    json.dumps(build_schema())


def test_tool_args_is_a_genuinely_open_object():
    """Confirmed live against llama-server: an open {"type": "object"}
    property (no declared sub-properties) is supported by its schema-to-
    grammar conversion — no need to encode args as a JSON string."""
    schema = build_schema()
    args_schema = schema["properties"]["tool"]["properties"]["args"]
    assert args_schema == {"type": "object"}


def test_parse_structured_output_extracts_think_and_keeps_full_action():
    raw = json.dumps({
        "think": "the user wants a file written",
        "rationale": "write it",
        "action": "use_tool",
        "tool": {"name": "write_file", "args": {"path": "workspace/hello.py", "content": "print('hi')"}},
    })
    think, action = parse_structured_output(raw)
    assert think == "the user wants a file written"
    assert action["action"] == "use_tool"
    assert action["tool"]["args"]["path"] == "workspace/hello.py"


def test_parse_structured_output_defaults_missing_think_to_empty_string():
    raw = json.dumps({"rationale": "x", "action": "finish", "finish": {"status": "success", "summary": "done"}})
    think, action = parse_structured_output(raw)
    assert think == ""
    assert action["finish"]["summary"] == "done"


def test_parse_structured_output_strips_a_stray_json_fence():
    """Grammar constraint should make this unnecessary, but stay defensive
    in case a backend fences the response anyway."""
    raw = '```json\n{"rationale":"x","action":"finish","finish":{"status":"success","summary":"ok"}}\n```'
    think, action = parse_structured_output(raw)
    assert action["action"] == "finish"


def test_parse_structured_output_rejects_empty_text():
    with pytest.raises(ValueError):
        parse_structured_output("")


def test_parse_structured_output_rejects_a_json_array():
    with pytest.raises(ValueError):
        parse_structured_output("[1, 2, 3]")


def test_parse_structured_output_rejects_missing_action_field():
    with pytest.raises(ValueError):
        parse_structured_output(json.dumps({"rationale": "x"}))


def test_parse_structured_output_reraises_json_decode_error_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_structured_output("{not json")
