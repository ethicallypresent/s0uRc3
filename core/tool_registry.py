"""Tool registry — disk, network, process, and build (CreatorCentral) tools.

Every call is forced through `normalize_tool_result` so the brain can only
treat `observed` as light. Ranking uses domain fit plus earned p_success,
not keyword hope alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from core.measure import normalize_tool_result
from core.paths import AgentPaths

log = logging.getLogger("agent.tools")

TOOL_DOMAINS = {
    "list_dir": "disk",
    "read_file": "disk",
    "write_file": "disk",
    "web_search": "network",
    "fetch_url": "network",
    "run_python": "process",
    "run_skill": "skill",
    "init_project": "build",
    "project_status": "build",
    "check_axiom": "build",
    "advance_axiom": "build",
    "log_evidence": "build",
    "close_project": "build",
}

_NETWORK_HINTS = ("http", "https", "web", "search", "url", "fetch", "online")
_DISK_HINTS = ("file", "dir", "directory", "workspace", "read", "write", "list", "path", "folder")
_PROCESS_HINTS = ("python", "code", "run", "exec", "script")
_BUILD_HINTS = (
    "axiom", "project", "initiation", "immutability", "naming", "ordering",
    "adversity", "fidelity", "migration", "recovery", "protocol", "rest",
    "advance", "gate", "evidence", "close out", "build", "creatorcentral",
)


class ToolRegistry:
    def __init__(self, paths: AgentPaths):
        self.paths = paths
        self.manifest = self._load_manifest()
        self._skill_runner: Callable[..., dict] | None = None

    def _load_manifest(self) -> list[dict[str, Any]]:
        path = self.paths.tools / "manifest.json"
        if not path.exists():
            log.warning("No tools/manifest.json found; only built-ins are available.")
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read tools manifest: %s", exc)
            return []
        return data.get("tools", []) if isinstance(data, dict) else data

    def attach_skill_runner(self, fn: Callable[..., dict]) -> None:
        self._skill_runner = fn

    def _catalog(self) -> list[dict[str, Any]]:
        by_name: dict[str, dict[str, Any]] = {}
        for n, domain in TOOL_DOMAINS.items():
            by_name[n] = {
                "name": n,
                "description": f"{domain} tool: {n}",
                "domain": domain,
            }
        for row in self.manifest:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            name = str(row["name"])
            domain = TOOL_DOMAINS.get(name, str(row.get("domain") or "disk"))
            by_name[name] = {
                "name": name,
                "description": str(row.get("description") or "")[:160],
                "domain": domain,
            }
        if self._skill_runner is not None:
            by_name.setdefault(
                "run_skill",
                {
                    "name": "run_skill",
                    "description": "Run an approved skill by name.",
                    "domain": "skill",
                },
            )
        return list(by_name.values())

    def list_filtered(
        self,
        query: str,
        k: int = 8,
        *,
        beliefs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q = (query or "").lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._catalog():
            hay = f"{entry['name']} {entry.get('description', '')}".lower()
            domain = entry.get("domain") or "disk"
            score = 0.0
            if q:
                words = [w for w in q.split() if len(w) > 2]
                score += sum(1.0 for w in words if w in hay)
                if entry["name"] in q:
                    score += 3.0
            if any(h in q for h in _NETWORK_HINTS) and domain == "network":
                score += 2.0
            if any(h in q for h in _DISK_HINTS) and domain == "disk":
                score += 2.0
            if any(h in q for h in _PROCESS_HINTS) and domain == "process":
                score += 2.0
            if any(h in q for h in _BUILD_HINTS) and domain == "build":
                score += 2.0
            p = _belief_p(beliefs, entry["name"])
            row = {**entry, "p_success": round(p, 3)}
            scored.append((score + p, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:k]]

    def registered_names(self) -> frozenset[str]:
        names = set(TOOL_DOMAINS)
        for row in self.manifest:
            if isinstance(row, dict) and row.get("name"):
                names.add(str(row["name"]))
        return frozenset(names)

    def call(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        from core.boundary import contains_nul, encoded_size, reject
        from core.contracts import MAX_TOOL_ARG_CHARS, MAX_TOOL_NAME_CHARS

        if not isinstance(name, str) or not name or len(name) > MAX_TOOL_NAME_CHARS or contains_nul(name):
            return normalize_tool_result(reject("unregistered_tool", layer="agent"), name="unknown")
        if name not in self.registered_names():
            return normalize_tool_result(reject("unregistered_tool", layer="agent", detail=name), name=name)
        if not isinstance(args, dict):
            return normalize_tool_result(reject("args_must_be_object", layer="agent"), name=name)
        if contains_nul(args):
            return normalize_tool_result(reject("nul_byte", layer="agent"), name=name)
        size = encoded_size(args)
        if size < 0:
            return normalize_tool_result(reject("args_not_serializable", layer="agent"), name=name)
        if size > MAX_TOOL_ARG_CHARS:
            return normalize_tool_result(reject("args_too_large", layer="agent"), name=name)
        try:
            raw = self._dispatch(str(name or ""), args)
        except TypeError as exc:
            raw = {"ok": False, "error": f"bad arguments for {name}: {exc}"}
        except (PermissionError, FileNotFoundError, OSError, ValueError) as exc:
            raw = {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.exception("Tool %s raised", name)
            raw = {"ok": False, "error": str(exc)}
        if not isinstance(raw, dict):
            raw = {"ok": False, "error": f"tool {name!r} returned {type(raw).__name__}"}
        return normalize_tool_result(raw, name=str(name or "unknown"))

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "run_skill":
            if self._skill_runner is None:
                return {"ok": False, "error": "no skill runner attached"}
            return self._skill_runner(**args)
        if name == "list_dir":
            from tools._core.file_tools import list_dir

            return list_dir(self.paths, str(args.get("path") or "."), str(args.get("glob") or "*"))
        if name == "read_file":
            from tools._core.file_tools import read_file

            return read_file(
                self.paths,
                str(args.get("path") or ""),
                int(args.get("max_chars") or 20000),
            )
        if name == "write_file":
            from tools._core.file_tools import write_file

            return write_file(self.paths, str(args.get("path") or ""), str(args.get("content") or ""))
        if name == "web_search":
            from tools._core.web_tools import web_search

            return web_search(str(args.get("query") or ""), int(args.get("num_results") or 5))
        if name == "fetch_url":
            from tools._core.web_tools import fetch_url

            return fetch_url(str(args.get("url") or ""), int(args.get("max_chars") or 12000))
        if name == "run_python":
            from tools._core.code_runner import run_python

            kwargs: dict[str, Any] = {"code": str(args.get("code") or "")}
            if args.get("timeout_sec") is not None:
                kwargs["timeout_sec"] = int(args["timeout_sec"])
            return run_python(**kwargs)
        if name == "init_project":
            from tools._core.creatorcentral_tools import init_project

            return init_project(
                self.paths,
                str(args.get("name") or ""),
                str(args.get("owner") or ""),
                str(args.get("purpose") or ""),
            )
        if name == "project_status":
            from tools._core.creatorcentral_tools import project_status

            proj_name = args.get("name")
            return project_status(self.paths, str(proj_name) if proj_name else None)
        if name == "check_axiom":
            from tools._core.creatorcentral_tools import check_axiom

            return check_axiom(self.paths, str(args.get("name") or ""))
        if name == "advance_axiom":
            from tools._core.creatorcentral_tools import advance_axiom

            return advance_axiom(self.paths, str(args.get("name") or ""))
        if name == "log_evidence":
            from tools._core.creatorcentral_tools import log_evidence

            return log_evidence(
                self.paths,
                str(args.get("name") or ""),
                str(args.get("requirement") or ""),
                str(args.get("text") or ""),
            )
        if name == "close_project":
            from tools._core.creatorcentral_tools import close_project

            reason = args.get("reason")
            return close_project(self.paths, str(args.get("name") or ""), str(reason) if reason else None)
        return {"ok": False, "error": f"unknown tool: {name}"}


def _belief_p(beliefs: dict[str, Any] | None, name: str) -> float:
    """Prior 0.67 until this machine has a record for the tool."""
    if not beliefs:
        return 0.67
    raw = beliefs.get(f"tool:{name}") or beliefs.get(name)
    if raw is None:
        return 0.67
    if isinstance(raw, dict):
        try:
            return float(raw.get("p_success", 0.67))
        except (TypeError, ValueError):
            return 0.67
    return float(getattr(raw, "p_success", 0.67))
