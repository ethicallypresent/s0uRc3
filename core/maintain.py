"""REST: idle maintenance so the runtime does not rot.

Safe, bounded cleanup. Never deletes brain/core/skills/approved.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.paths import AgentPaths

log = logging.getLogger("agent.maintain")

_LOG_CAP_BYTES = 2 * 1024 * 1024
_LOG_KEEP_BYTES = 256 * 1024
_LOG_NAMES = (
    "last_run.log",
    "llama_server.err.log",
    "llama_server.out.log",
    "llama_server.functiongemma.err.log",
    "llama_server.functiongemma.out.log",
)


def _trim_file(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= _LOG_CAP_BYTES:
        return False
    try:
        data = path.read_bytes()[-_LOG_KEEP_BYTES:]
        path.write_bytes(b"...trimmed...\n" + data)
        return True
    except OSError as exc:
        log.warning("log trim failed %s: %s", path, exc)
        return False


def rest(paths: AgentPaths) -> dict[str, int]:
    """Run idle cleanup. Call after a run and during boot."""
    trimmed = 0
    workspace = paths.workspace
    if workspace.is_dir():
        for name in _LOG_NAMES:
            if _trim_file(workspace / name):
                trimmed += 1
        for extra in workspace.glob("*.log"):
            if extra.name not in _LOG_NAMES and _trim_file(extra):
                trimmed += 1
    return {"logs_trimmed": trimmed}
