"""CreatorCentral: the advance gate must be real, and close-outs must be honest.

Each test points the module at a throwaway projects/ directory (monkeypatch)
so the actual s0uRc3 self-construction project file is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from creatorcentral import creatorcentral as cc


@pytest.fixture(autouse=True)
def isolated_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    projects_dir = tmp_path / "projects"
    monkeypatch.setattr(cc, "PROJECTS_DIR", projects_dir)
    return projects_dir


def test_init_starts_at_initiation():
    project = cc.init_project("demo", owner="alice", purpose="a test build")
    assert project["current_axiom_index"] == 0
    assert project["status"] == "in_progress"
    status = cc.status_one("demo")
    assert status["current_axiom"] == "INITIATION"


def test_init_twice_fails():
    cc.init_project("demo", owner="alice", purpose="x")
    with pytest.raises(FileExistsError):
        cc.init_project("demo", owner="alice", purpose="x")


def test_advance_refuses_with_missing_requirements():
    cc.init_project("demo", owner="alice", purpose="x")
    result = cc.advance_project("demo")
    assert result["ok"] is False
    assert result["error"] == "requirements_missing"
    assert result["axiom"] == "INITIATION"
    assert set(result["missing"]) == {"what", "why", "owner"}


def test_log_unknown_requirement_key_rejected():
    cc.init_project("demo", owner="alice", purpose="x")
    with pytest.raises(ValueError):
        cc.log_evidence("demo", "not_a_real_key", "text")


def test_log_then_advance_succeeds():
    cc.init_project("demo", owner="alice", purpose="x")
    cc.log_evidence("demo", "what", "a test build")
    cc.log_evidence("demo", "why", "to prove the gate works")
    cc.log_evidence("demo", "owner", "alice")
    check = cc.check_project("demo")
    assert check["ok"] is True
    result = cc.advance_project("demo")
    assert result["ok"] is True
    assert result["completed"] == "INITIATION"
    assert result["next_axiom"] == "IMMUTABILITY"


def test_close_before_rest_is_closed_early_not_done():
    cc.init_project("demo", owner="alice", purpose="x")
    result = cc.close_project("demo", reason="ran out of time")
    assert result["outcome"] == "closed_early"
    status = cc.status_one("demo")
    assert status["status"] == "closed_early"
    assert status["close_reason"] == "ran out of time"


def _satisfy_and_advance(name: str, axiom: dict) -> None:
    for req in axiom["requirements"]:
        cc.log_evidence(name, req["key"], f"evidence for {req['key']}")
    result = cc.advance_project(name)
    assert result["ok"] is True, result


def test_full_walk_from_initiation_to_rest_then_close_done():
    cc.init_project("demo", owner="alice", purpose="x")
    axioms = cc.load_axioms()
    for axiom in axioms:
        _satisfy_and_advance("demo", axiom)
    status = cc.status_one("demo")
    assert status["current_axiom"] == "REST"
    assert status["status"] == "at_rest"
    result = cc.close_project("demo")
    assert result["outcome"] == "done"
    assert cc.status_one("demo")["status"] == "done"


def test_advance_after_rest_refuses():
    cc.init_project("demo", owner="alice", purpose="x")
    for axiom in cc.load_axioms():
        _satisfy_and_advance("demo", axiom)
    result = cc.advance_project("demo")
    assert result["ok"] is False
    assert result["error"] == "already_at_rest"


def test_status_all_lists_projects():
    cc.init_project("one", owner="alice", purpose="x")
    cc.init_project("two", owner="bob", purpose="y")
    listed = {p["name"]: p for p in cc.status_all()}
    assert set(listed) == {"one", "two"}
    assert listed["one"]["axiom"] == "INITIATION"
