"""CreatorCentral — a gated CLI for walking a build through the ten CREATE axioms.

INITIATION -> IMMUTABILITY -> NAMING -> ORDERING -> ADVERSITY -> FIDELITY ->
MIGRATION -> RECOVERY -> PROTOCOL -> REST.

Stdlib only. Each project is one JSON file under projects/<name>.json.
`advance` is the gate: it will not move a project past an axiom whose
requirements lack evidence, and it always says exactly what's missing.

    python creatorcentral.py init myproj --owner alice --purpose "..."
    python creatorcentral.py log myproj what "Building a thing"
    python creatorcentral.py check myproj
    python creatorcentral.py advance myproj
    python creatorcentral.py status [myproj]
    python creatorcentral.py close myproj [--reason TEXT]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
AXIOMS_PATH = HERE / "axioms.json"
PROJECTS_DIR = HERE / "projects"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_axioms() -> list[dict[str, Any]]:
    data = json.loads(AXIOMS_PATH.read_text(encoding="utf-8"))
    return data["axioms"]


def _project_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
    if not safe:
        raise ValueError("invalid project name")
    return PROJECTS_DIR / f"{safe}.json"


def load_project(name: str) -> dict[str, Any]:
    path = _project_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no such project: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_project(project: dict[str, Any]) -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _project_path(project["name"])
    path.write_text(json.dumps(project, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def current_axiom(project: dict[str, Any], axioms: list[dict[str, Any]]) -> dict[str, Any] | None:
    idx = project["current_axiom_index"]
    if idx >= len(axioms):
        return None
    return axioms[idx]


def init_project(name: str, owner: str, purpose: str) -> dict[str, Any]:
    path = _project_path(name)
    if path.exists():
        raise FileExistsError(f"project already exists: {name}")
    axioms = load_axioms()
    project = {
        "name": name,
        "owner": owner,
        "purpose": purpose,
        "created_at": _now(),
        "current_axiom_index": 0,
        "status": "in_progress",
        "close_reason": None,
        "evidence": {a["name"]: {} for a in axioms},
        "log": [
            {"at": _now(), "event": "init", "detail": f"owner={owner} purpose={purpose!r}"}
        ],
    }
    save_project(project)
    return project


def log_evidence(name: str, requirement_key: str, text: str) -> dict[str, Any]:
    project = load_project(name)
    axioms = load_axioms()
    axiom = current_axiom(project, axioms)
    if axiom is None:
        raise ValueError("project has already reached REST; nothing left to log against")
    valid_keys = {r["key"] for r in axiom["requirements"]}
    if requirement_key not in valid_keys:
        raise ValueError(
            f"'{requirement_key}' is not a requirement of axiom {axiom['name']} "
            f"(valid: {sorted(valid_keys)})"
        )
    project["evidence"][axiom["name"]][requirement_key] = {"text": text, "at": _now()}
    project["log"].append(
        {"at": _now(), "event": "log", "detail": f"{axiom['name']}.{requirement_key}: {text}"}
    )
    save_project(project)
    return project


def check_project(name: str) -> dict[str, Any]:
    """Read-only: which of the current axiom's requirements are met vs missing."""
    project = load_project(name)
    axioms = load_axioms()
    axiom = current_axiom(project, axioms)
    if axiom is None:
        return {"ok": True, "at_rest": True, "axiom": None, "missing": [], "met": []}
    have = project["evidence"].get(axiom["name"], {})
    met = [r["key"] for r in axiom["requirements"] if r["key"] in have]
    missing = [r["key"] for r in axiom["requirements"] if r["key"] not in have]
    return {
        "ok": not missing,
        "at_rest": False,
        "axiom": axiom["name"],
        "axiom_index": project["current_axiom_index"],
        "met": met,
        "missing": missing,
    }


def advance_project(name: str) -> dict[str, Any]:
    """The gate. Refuses to move past an axiom with unchecked requirements."""
    result = check_project(name)
    if result["at_rest"]:
        return {"ok": False, "error": "already_at_rest", "detail": result}
    if not result["ok"]:
        return {
            "ok": False,
            "error": "requirements_missing",
            "axiom": result["axiom"],
            "missing": result["missing"],
        }
    project = load_project(name)
    axioms = load_axioms()
    finished_axiom = axioms[project["current_axiom_index"]]
    project["current_axiom_index"] += 1
    reached_rest = project["current_axiom_index"] >= len(axioms)
    project["log"].append(
        {
            "at": _now(),
            "event": "advance",
            "detail": f"completed {finished_axiom['name']}"
            + (" -> REST reached" if reached_rest else f" -> {axioms[project['current_axiom_index']]['name']}"),
        }
    )
    if reached_rest:
        project["status"] = "at_rest"
    save_project(project)
    return {
        "ok": True,
        "completed": finished_axiom["name"],
        "at_rest": reached_rest,
        "next_axiom": None if reached_rest else axioms[project["current_axiom_index"]]["name"],
    }


def close_project(name: str, reason: str | None = None) -> dict[str, Any]:
    project = load_project(name)
    if project["status"] == "at_rest":
        project["status"] = "done"
        project["close_reason"] = None
        project["log"].append({"at": _now(), "event": "close", "detail": "done: reached REST"})
        outcome = "done"
    else:
        axioms = load_axioms()
        axiom = current_axiom(project, axioms)
        detail = f"closed_early at axiom {axiom['name'] if axiom else 'unknown'}: {reason or 'no reason given'}"
        project["status"] = "closed_early"
        project["close_reason"] = reason or "no reason given"
        project["log"].append({"at": _now(), "event": "close", "detail": detail})
        outcome = "closed_early"
    save_project(project)
    return {"ok": True, "outcome": outcome, "project": project["name"]}


def status_all() -> list[dict[str, Any]]:
    if not PROJECTS_DIR.exists():
        return []
    axioms = load_axioms()
    out = []
    for path in sorted(PROJECTS_DIR.glob("*.json")):
        project = json.loads(path.read_text(encoding="utf-8"))
        idx = project["current_axiom_index"]
        axiom_name = axioms[idx]["name"] if idx < len(axioms) else "REST"
        out.append({"name": project["name"], "status": project["status"], "axiom": axiom_name})
    return out


def status_one(name: str) -> dict[str, Any]:
    project = load_project(name)
    axioms = load_axioms()
    axiom = current_axiom(project, axioms)
    check = check_project(name)
    return {
        "name": project["name"],
        "owner": project["owner"],
        "purpose": project["purpose"],
        "status": project["status"],
        "close_reason": project["close_reason"],
        "current_axiom": axiom["name"] if axiom else "REST",
        "met": check["met"],
        "missing": check["missing"],
        "log": project["log"],
    }


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        project = init_project(args.name, args.owner, args.purpose)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "project": project["name"], "axiom": "INITIATION"}, indent=2))
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    try:
        log_evidence(args.name, args.requirement, args.text)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True}, indent=2))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        result = check_project(args.name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _cmd_advance(args: argparse.Namespace) -> int:
    try:
        result = advance_project(args.name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _cmd_status(args: argparse.Namespace) -> int:
    if args.name:
        try:
            result = status_one(args.name)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        result = status_all()
    print(json.dumps(result, indent=2))
    return 0


def _cmd_close(args: argparse.Namespace) -> int:
    try:
        result = close_project(args.name, args.reason)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="creatorcentral", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Start a new project at INITIATION")
    p.add_argument("name")
    p.add_argument("--owner", required=True)
    p.add_argument("--purpose", required=True)
    p.set_defaults(func=_cmd_init)

    p = sub.add_parser("log", help="Attach evidence to a requirement of the current axiom")
    p.add_argument("name")
    p.add_argument("requirement")
    p.add_argument("text")
    p.set_defaults(func=_cmd_log)

    p = sub.add_parser("check", help="Read-only: show met/missing requirements for the current axiom")
    p.add_argument("name")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("advance", help="Gate: move to the next axiom if requirements are met")
    p.add_argument("name")
    p.set_defaults(func=_cmd_advance)

    p = sub.add_parser("status", help="Show one project's state, or list all projects")
    p.add_argument("name", nargs="?", default=None)
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("close", help="Close a project (honestly: done, or closed_early)")
    p.add_argument("name")
    p.add_argument("--reason", default=None)
    p.set_defaults(func=_cmd_close)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
