"""Minimal smoke test: dry-run the loop end-to-end with no LLM key needed.

Uses a temp directory as a throwaway agent root so it never touches the
real workspace/skills/db.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.loop import AgentLoop
from core.paths import AgentPaths


@pytest.fixture
def agent_root(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    for sub in ("brain", "core", "db", "skills", "tools", "workspace"):
        (root / sub).mkdir(parents=True)
    (root / "brain" / "constitution.md").write_text("# test constitution\n")
    (root / "brain" / "system_prompt.md").write_text("# test system prompt\n")
    (root / "brain" / "reasoning_config.json").write_text(
        '{"model": "gpt-4o", "max_steps": 5, "skill_topk": 8, "tool_topk": 8, "memory_topk": 5}'
    )
    return root


def test_dry_run_finishes(agent_root: Path):
    paths = AgentPaths.discover(start=agent_root)
    loop = AgentLoop(paths)
    result = loop.run("List the workspace and stop.", dry_run=True)
    assert result.get("ok") is True
    loop.memory.close()


def test_dry_run_still_works_with_structured_output_enabled(tmp_path: Path):
    """dry_run_policy() (core/loop.py) always emits the free-text
    <think>+json form, regardless of the structured_output config — a run
    must not try to parse that as a schema-constrained JSON object just
    because the config asked for structured output for real LLM calls."""
    root = tmp_path / "agent2"
    for sub in ("brain", "core", "db", "skills", "tools", "workspace"):
        (root / sub).mkdir(parents=True)
    (root / "brain" / "constitution.md").write_text("# test constitution\n")
    (root / "brain" / "system_prompt.md").write_text("# test system prompt\n")
    (root / "brain" / "reasoning_config.json").write_text(
        '{"model": "gpt-4o", "max_steps": 5, "skill_topk": 8, "tool_topk": 8, '
        '"memory_topk": 5, "structured_output": {"enabled": true}}'
    )
    paths = AgentPaths.discover(start=root)
    loop = AgentLoop(paths)
    assert loop.cfg.get("structured_output", {}).get("enabled") is True
    result = loop.run("List the workspace and stop.", dry_run=True)
    assert result.get("ok") is True
    loop.memory.close()