"""Safety net: no test in this suite may leave the live project state touched.

Most tests build an isolated root with AgentPaths.discover(start=tmp_path/"agent"),
but several (test_creation.py, test_edge.py, test_permission.py, test_phase_b.py,
test_phase_c.py, test_refine_code.py, test_scaffold.py, test_standards.py) call
AgentLoop(AgentPaths.discover()) / AgentPaths.discover() with no start= override,
which resolves against this real repo. The root cause, confirmed: AgentLoop.__init__
always opens db/memory.sqlite, db/evolution.json, and db/beliefs.json from
self.paths.db regardless of which SkillManager a test later swaps in, so every
one of those tests was writing real events, refine-cycles, and belief scores into
production state (this is why db/evolution.json accumulated ~25 near-identical
"bump_text" refine cycles from pytest runs, and why tool:list_dir's belief score
got inflated purely from test/salvage traffic rather than real usage). Snapshot
the whole db/ directory before every test and restore it byte-for-byte after,
regardless of which test (or future test) touches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_CONFIG = _REPO_ROOT / "brain" / "reasoning_config.json"
_REAL_DB_DIR = _REPO_ROOT / "db"


def _snapshot_dir(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture(autouse=True)
def _protect_real_config():
    before = _REAL_CONFIG.read_bytes() if _REAL_CONFIG.exists() else None
    yield
    if before is None:
        return
    after = _REAL_CONFIG.read_bytes() if _REAL_CONFIG.exists() else None
    if after != before:
        _REAL_CONFIG.write_bytes(before)


@pytest.fixture(autouse=True)
def _protect_real_db():
    before = _snapshot_dir(_REAL_DB_DIR)
    yield
    after = _snapshot_dir(_REAL_DB_DIR)
    if after == before:
        return
    for rel, data in before.items():
        target = _REAL_DB_DIR / rel
        if not target.exists() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for rel in after.keys() - before.keys():
        stray = _REAL_DB_DIR / rel
        if stray.exists():
            stray.unlink()
