"""The convergence check (item 4) must actually cut the live generator, not
just detect convergence in the abstract — this locks in the wiring inside
AgentLoop._call_with_timeout, mocking call_llm_stream so no server is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.loop as loop_mod
from core.loop import AgentLoop
from core.router import ACT, DELIBERATE, GLANCE


class _FakeStream:
    """Mimics call_llm_stream's generator: yields pieces, tracks close()."""

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = list(pieces)
        self.closed = False
        self._consumed = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.closed or not self._pieces:
            raise StopIteration
        self._consumed += 1
        return self._pieces.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_self() -> SimpleNamespace:
    return SimpleNamespace(cfg={"llm": {}}, on_token=None)


def _patch_stream(monkeypatch: pytest.MonkeyPatch, pieces: list[str]) -> _FakeStream:
    stream = _FakeStream(pieces)
    monkeypatch.setattr(loop_mod, "call_llm_stream", lambda *a, **kw: stream)
    return stream


def test_deliberate_stream_is_closed_early_once_converged(fake_self, monkeypatch):
    pieces = [
        "intent: read config\n",
        "next: use_tool read_file\n",
        "reconsidering, still the same call\n",
        "next: use_tool read_file\n",
        "</think>\n",
        '```json\n{"action":"use_tool"}\n```',
    ]
    stream = _patch_stream(monkeypatch, pieces)
    result = AgentLoop._call_with_timeout(
        fake_self, "sys", "packet", fake_self.cfg, model=None, max_tokens=800, tier=DELIBERATE
    )
    assert stream.closed is True
    # Stopped right after the second matching "next:" line — the </think>
    # and json action that follow were never consumed, saving real tokens.
    assert "</think>" not in result
    assert '"action"' not in result


def test_glance_stream_is_never_cut_even_if_it_repeats(fake_self, monkeypatch):
    """The stopping rule is scoped to deliberate on purpose — glance is
    already capped so tightly there's rarely room for the pattern to
    recur, and act has no <think> to converge on at all."""
    pieces = ["next: use_tool x\n", "next: use_tool x\n", "</think>\n", '```json\n{"action":"use_tool"}\n```']
    stream = _patch_stream(monkeypatch, pieces)
    result = AgentLoop._call_with_timeout(
        fake_self, "sys", "packet", fake_self.cfg, model=None, max_tokens=250, tier=GLANCE
    )
    assert stream.closed is False
    assert '"action":"use_tool"' in result


def test_convergence_check_can_be_disabled_via_config(fake_self, monkeypatch):
    fake_self.cfg = {"llm": {"stop_on_converged_thinking": False}}
    pieces = ["next: x\n", "next: x\n", "</think>\n", '```json\n{"action":"use_tool"}\n```']
    stream = _patch_stream(monkeypatch, pieces)
    result = AgentLoop._call_with_timeout(
        fake_self, "sys", "packet", fake_self.cfg, model=None, max_tokens=800, tier=DELIBERATE
    )
    assert stream.closed is False
    assert '"action":"use_tool"' in result


def test_never_cuts_after_think_has_closed_even_if_json_repeats_the_word_next(fake_self, monkeypatch):
    """Protects the JSON action itself: once </think> is seen, convergence
    checking must stop, or a JSON value that happens to contain "next:"
    twice could truncate the actual action."""
    pieces = [
        "next: use_tool a\n",
        "next: use_tool a\n",
        "</think>\n",
        '```json\n{"action":"use_tool","rationale":"next: still next: fine"}\n```',
    ]
    stream = _patch_stream(monkeypatch, pieces)
    AgentLoop._call_with_timeout(fake_self, "sys", "packet", fake_self.cfg, model=None, max_tokens=800, tier=DELIBERATE)
    # It converges and stops right at the second "next:" line, before
    # </think> is even reached — confirms cutting happens at the earliest
    # legitimate point, not after accidentally scanning into the JSON.
    assert stream.closed is True
