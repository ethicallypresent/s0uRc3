"""Think / answer splitter (WPF parity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tui.stream import StreamSplitter, strip_json_fence


def test_think_then_json_summary():
    s = StreamSplitter()
    raw = (
        "<think>observed: workspace\ninferred: list first</think>\n"
        '```json\n{"action":"finish","finish":{"status":"success","summary":"listed files"}}\n```'
    )
    s.feed(raw)
    view = s.finalize()
    assert "observed: workspace" in view.thinking
    assert view.answer == "listed files"
    assert view.is_thinking is False
    assert view.thinking_label == "Thought"


def test_streaming_open_tag_keeps_answer_empty():
    s = StreamSplitter()
    s.feed("<think>hello")
    view = s.view()
    assert view.is_thinking is True
    assert "hello" in view.thinking
    assert view.answer == ""


def test_thinking_alias_tags():
    s = StreamSplitter()
    s.feed("<thinking>plan</thinking>\nhello user")
    view = s.finalize()
    assert view.thinking == "plan"
    assert "hello user" in view.answer


def test_blank_line_fallback():
    s = StreamSplitter()
    thinking = "x" * 50
    s.feed(thinking + "\n\n" + "the answer is 4")
    view = s.finalize()
    assert thinking in view.thinking
    assert "the answer is 4" in view.answer


def test_strip_json_fence_prefers_summary():
    text = 'intro\n```json\n{"action":"finish","finish":{"summary":"done"}}\n```'
    assert "done" in strip_json_fence(text)


def test_strip_bare_action_hides_json():
    s = StreamSplitter()
    s.feed('<think>n</think>\n{"action":"update_plan","plan":{"nodes":[]}}')
    view = s.finalize()
    assert "update_plan" not in view.answer


def test_leaked_input_packet_is_never_shown():
    """Confirmed live: the model sometimes echoes its own input packet back
    instead of a real answer. core/loop.py always builds that payload with
    "user_goal" as its first key — must never reach the user verbatim."""
    s = StreamSplitter()
    leaked = '{"user_goal":"i need a file","mode":"action","perception":{"user_input":""}}'
    s.feed(f"<think>write a file</think>\n{leaked}")
    view = s.finalize()
    assert "user_goal" not in view.answer
    assert "perception" not in view.answer


def test_strip_json_fence_hides_leaked_packet_directly():
    leaked = '{"user_goal":"x","mode":"action","perception":{}}'
    assert strip_json_fence(leaked) == ""


def test_structured_response_pulls_think_and_rationale_from_the_object():
    s = StreamSplitter()
    raw = '{"think":"the user wants a file","rationale":"write it","action":"use_tool","tool":{"name":"write_file","args":{}}}'
    s.feed(raw)
    view = s.finalize()
    assert view.thinking == "the user wants a file"
    assert view.answer == "write it"
    assert view.is_thinking is False
    assert view.thinking_label == "Thought"


def test_structured_finish_prefers_finish_summary_over_rationale():
    s = StreamSplitter()
    raw = '{"think":"done","rationale":"finishing up","action":"finish","finish":{"status":"success","summary":"wrote the file"}}'
    s.feed(raw)
    view = s.finalize()
    assert view.answer == "wrote the file"


def test_structured_response_streaming_mid_object_shows_live_think_preview():
    s = StreamSplitter()
    s.feed('{"think":"still figuring out the p')
    view = s.view()
    assert view.is_thinking is True
    assert view.thinking == "still figuring out the p"
    assert view.answer == ""


def test_structured_response_never_blank_line_splits_a_multi_paragraph_think():
    """A structured think field can legitimately contain a blank line —
    finalize() must not cannibalize the end of it into a fake answer the
    way it does for the free-text blank-line fallback."""
    s = StreamSplitter()
    raw = json.dumps({
        "think": "first paragraph of reasoning\n\nsecond paragraph of reasoning",
        "rationale": "act now",
        "action": "finish",
        "finish": {"status": "success", "summary": "done"},
    })
    s.feed(raw)
    view = s.finalize()
    assert "second paragraph of reasoning" in view.thinking
    assert view.answer == "done"
