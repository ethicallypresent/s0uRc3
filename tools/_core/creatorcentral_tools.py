"""Protected bridge from the agent tool registry to CreatorCentral.

Agent code must not edit this module. Thin wrappers only — all axiom-gate
logic lives in creatorcentral/creatorcentral.py so the CLI and the agent
loop can never disagree about what the gate allows.
"""

from __future__ import annotations

from typing import Any

from core.paths import AgentPaths
from creatorcentral import creatorcentral as cc


def init_project(paths: AgentPaths, name: str, owner: str, purpose: str) -> dict[str, Any]:
    try:
        project = cc.init_project(name, owner, purpose)
    except FileExistsError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "project": project["name"], "axiom": "INITIATION"}


def project_status(paths: AgentPaths, name: str | None = None) -> dict[str, Any]:
    if name:
        try:
            return {"ok": True, **cc.status_one(name)}
        except FileNotFoundError as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": True, "projects": cc.status_all()}


def check_axiom(paths: AgentPaths, name: str) -> dict[str, Any]:
    """ok = the check ran; gate_ok = whether the current axiom's requirements are met.

    These are deliberately different fields: a report of missing requirements
    is a successful tool call, not a tool failure — conflating the two would
    make the agent loop treat an honest "not yet" as a broken tool and abandon
    the run (Kurama's steering aborts after repeated tool failures).
    """
    try:
        result = cc.check_project(name)
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    gate_ok = result.pop("ok")
    return {"ok": True, "gate_ok": gate_ok, **result}


def advance_axiom(paths: AgentPaths, name: str) -> dict[str, Any]:
    """ok = the gate ran; advanced = whether it actually moved to the next axiom.

    A refusal (missing requirements, or already at REST) is the gate doing
    its job correctly, not a tool failure.
    """
    try:
        result = cc.advance_project(name)
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    gate_ok = result.pop("ok")
    return {"ok": True, "advanced": gate_ok, **result}


def log_evidence(paths: AgentPaths, name: str, requirement: str, text: str) -> dict[str, Any]:
    try:
        cc.log_evidence(name, requirement, text)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "project": name, "requirement": requirement}


def close_project(paths: AgentPaths, name: str, reason: str | None = None) -> dict[str, Any]:
    try:
        return cc.close_project(name, reason)
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
