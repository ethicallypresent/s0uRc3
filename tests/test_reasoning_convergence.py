"""Stopping rule (thinking-budgets item 4): a deliberate reasoning block
that restates the same conclusion twice has stopped changing its answer —
more tokens there are polishing, not deciding. Heuristic, not semantic: it
keys off this codebase's own think-block convention of ending with a
`next: <action>` line.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.loop import reasoning_has_converged


def test_no_next_line_yet_has_not_converged():
    assert reasoning_has_converged("intent: do the thing\nobserved: nothing yet") is False


def test_a_single_next_line_has_not_converged():
    """One conclusion isn't a repeat — nothing to compare it against yet."""
    assert reasoning_has_converged("intent: do the thing\nnext: use_tool read_file") is False


def test_repeating_the_same_conclusion_has_converged():
    text = (
        "intent: read the config\n"
        "next: use_tool read_file\n"
        "reconsidering... still seems right\n"
        "next: use_tool read_file\n"
    )
    assert reasoning_has_converged(text) is True


def test_matching_is_case_and_whitespace_insensitive():
    text = "next:   Use_Tool Read_File  \nnext: use_tool read_file\n"
    assert reasoning_has_converged(text) is True


def test_a_changed_conclusion_has_not_converged():
    """The model changed its mind — that's exactly the case where more
    tokens ARE still doing useful work, so this must not fire."""
    text = "next: use_tool list_dir\nnext: use_tool read_file\n"
    assert reasoning_has_converged(text) is False


def test_empty_text_has_not_converged():
    assert reasoning_has_converged("") is False
