"""Desktop shortcut for the s0uRc3 console agent (Windows).

The .lnk is the cross-language handle: name s0uRc3, target the exe (or the
system Python + main.py when not frozen), working directory the agent
root. No extra packages — PowerShell COM only.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("agent.desktop")

SHORTCUT_NAME = "s0uRc3.lnk"


def desktop_dir() -> Path | None:
    """User Desktop, including OneDrive-redirected folders."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[Environment]::GetFolderPath('Desktop')",
            ],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
        path = Path(out.strip())
        if path.is_dir():
            return path
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass
    for candidate in (
        Path.home() / "Desktop",
        Path(os.environ.get("USERPROFILE", "")) / "Desktop",
        Path(os.environ.get("OneDrive", "")) / "Desktop",
    ):
        if candidate.is_dir():
            return candidate
    return None


def launch_target() -> tuple[Path, str, Path]:
    """Return (target_path, arguments, working_directory) for this process."""
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        return exe, "", exe.parent
    root = Path(__file__).resolve().parents[1]
    return Path(sys.executable).resolve(), str(root / "main.py"), root


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def shortcut_script(
    *,
    link_path: Path,
    target: Path,
    working_directory: Path,
    arguments: str = "",
    description: str = "s0uRc3 local Builder Agent",
) -> str:
    icon = str(target) if target.suffix.lower() == ".exe" else ""
    lines = [
        f"$sc = (New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(str(link_path))})",
        f"$sc.TargetPath = {_ps_quote(str(target))}",
        f"$sc.WorkingDirectory = {_ps_quote(str(working_directory))}",
        f"$sc.WindowStyle = 1",
        f"$sc.Description = {_ps_quote(description)}",
    ]
    if arguments:
        lines.append(f"$sc.Arguments = {_ps_quote(arguments)}")
    if icon:
        lines.append(f"$sc.IconLocation = {_ps_quote(icon + ',0')}")
    lines.append("$sc.Save()")
    return "; ".join(lines)


def install_desktop_shortcut(
    *,
    target: Path | None = None,
    working_directory: Path | None = None,
    arguments: str | None = None,
    desktop: Path | None = None,
) -> dict[str, Any]:
    """Create or refresh Desktop\\Kurama.lnk. Never raises to the caller."""
    if os.name != "nt":
        return {"ok": False, "error": "not_windows"}
    dest = desktop if desktop is not None else desktop_dir()
    if dest is None:
        return {"ok": False, "error": "desktop_not_found"}
    if target is None or working_directory is None:
        auto_target, auto_args, auto_cwd = launch_target()
        target = target or auto_target
        working_directory = working_directory or auto_cwd
        if arguments is None:
            arguments = auto_args
    arguments = arguments or ""
    link = dest / SHORTCUT_NAME
    script = shortcut_script(
        link_path=link,
        target=target,
        working_directory=working_directory,
        arguments=arguments,
    )
    try:
        subprocess.check_call(
            ["powershell", "-NoProfile", "-Command", script],
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        log.warning("Could not write desktop shortcut: %s", exc)
        return {"ok": False, "error": str(exc), "path": str(link)}
    return {"ok": True, "path": str(link), "target": str(target)}
