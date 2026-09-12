"""The agent-tool layer over CreatorCentral must not confuse "the gate said
no" with "the tool failed" — that misreporting once caused Kurama's steering
system to abort a run early after a perfectly successful check_axiom call
that merely reported missing requirements. Locking that in here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.paths import AgentPaths
from core.tool_registry import ToolRegistry
from creatorcentral import creatorcentral as cc


@pytest.fixture(autouse=True)
def isolated_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    projects_dir = tmp_path / "projects"
    monkeypatch.setattr(cc, "PROJECTS_DIR", projects_dir)
    return projects_dir


@pytest.fixture
def tools() -> ToolRegistry:
    return ToolRegistry(AgentPaths.discover())


def test_check_axiom_with_missing_requirements_is_still_a_successful_tool_call(tools: ToolRegistry):
    tools.call("init_project", {"name": "demo", "owner": "alice", "purpose": "x"})
    result = tools.call("check_axiom", {"name": "demo"})
    assert result["ok"] is True
    assert result["observed"]["gate_ok"] is False
    assert set(result["observed"]["missing"]) == {"what", "why", "owner"}


def test_advance_axiom_refusal_is_still_a_successful_tool_call(tools: ToolRegistry):
    tools.call("init_project", {"name": "demo", "owner": "alice", "purpose": "x"})
    result = tools.call("advance_axiom", {"name": "demo"})
    assert result["ok"] is True
    assert result["observed"]["advanced"] is False


def test_advance_axiom_success_reports_advanced_true(tools: ToolRegistry):
    tools.call("init_project", {"name": "demo", "owner": "alice", "purpose": "x"})
    for key in ("what", "why", "owner"):
        tools.call("log_evidence", {"name": "demo", "requirement": key, "text": "evidence"})
    result = tools.call("advance_axiom", {"name": "demo"})
    assert result["ok"] is True
    assert result["observed"]["advanced"] is True
    assert result["observed"]["completed"] == "INITIATION"


def test_check_axiom_on_missing_project_is_a_real_failure(tools: ToolRegistry):
    result = tools.call("check_axiom", {"name": "does_not_exist"})
    assert result["ok"] is False


def test_close_project_reports_closed_early_honestly(tools: ToolRegistry):
    tools.call("init_project", {"name": "demo", "owner": "alice", "purpose": "x"})
    result = tools.call("close_project", {"name": "demo", "reason": "out of time"})
    assert result["ok"] is True
    assert result["observed"]["outcome"] == "closed_early"
