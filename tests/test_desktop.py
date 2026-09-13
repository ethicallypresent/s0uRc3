"""Desktop shortcut helper (no live Desktop writes)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.desktop import SHORTCUT_NAME, shortcut_script


def test_shortcut_script_points_at_exe(tmp_path: Path):
    target = tmp_path / "Kurama.exe"
    link = tmp_path / SHORTCUT_NAME
    script = shortcut_script(
        link_path=link,
        target=target,
        working_directory=tmp_path,
        arguments="",
    )
    assert "CreateShortcut" in script
    assert str(target) in script
    assert str(tmp_path) in script
    assert "WindowStyle = 1" in script
    assert "Kurama.exe,0" in script


def test_shortcut_script_quotes_apostrophes(tmp_path: Path):
    target = tmp_path / "O'Brien" / "Kurama.exe"
    script = shortcut_script(
        link_path=tmp_path / SHORTCUT_NAME,
        target=target,
        working_directory=target.parent,
    )
    assert "O''Brien" in script


def test_shortcut_script_always_sets_arguments_even_when_empty(tmp_path: Path):
    """A stale Arguments string from an earlier install_desktop_shortcut()
    call (e.g. a dev-mode shortcut pointing at "path/to/main.py") must be
    cleared when the shortcut is later refreshed to take no arguments —
    CreateShortcut loads the existing .lnk's properties, so skipping this
    line for a falsy value would leave the old value in place forever."""
    script = shortcut_script(
        link_path=tmp_path / SHORTCUT_NAME,
        target=tmp_path / "s0uRc3.exe",
        working_directory=tmp_path,
        arguments="",
    )
    assert "$sc.Arguments = ''" in script
