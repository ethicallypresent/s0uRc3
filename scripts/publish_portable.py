#!/usr/bin/env python3
"""Build dist/s0uRc3-Portable — double-click s0uRc3.exe opens the terminal GUI."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "s0uRc3-Portable"
PYI_DIST = ROOT / "dist" / "s0uRc3"
SPEC = ROOT / "s0urc3.spec"

COPY_DIRS = ("brain", "core", "tui", "db", "skills", "tools", "models", "workspace", "tests", "data", "creatorcentral")
COPY_FILES = ("main.py", "requirements.txt")


def find_llama_server() -> Path | None:
    local = ROOT / "tools" / "bin" / "llama-server.exe"
    if local.exists():
        return local
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget.exists():
        hits = list(winget.rglob("llama-server.exe"))
        if hits:
            return hits[0]
    return None


def _rm(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    print("==> Installing build deps (textual, pyinstaller)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "textual>=0.80.0", "pyinstaller>=6.0.0"])

    print("==> PyInstaller onedir s0uRc3.exe…")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(SPEC),
    ]
    subprocess.check_call(cmd, cwd=ROOT)

    exe = PYI_DIST / "s0uRc3.exe"
    if not exe.exists():
        # Linux/mac fallback
        alt = PYI_DIST / "s0uRc3"
        if not alt.exists():
            print(f"ERROR: missing {exe}", file=sys.stderr)
            return 1
        exe = alt

    _rm(DIST)
    DIST.mkdir(parents=True)
    (DIST / "tools" / "bin").mkdir(parents=True)

    # Copy the onedir payload (exe + _internal)
    for item in PYI_DIST.iterdir():
        dest = DIST / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"==> Copied {exe.name}")

    for name in COPY_DIRS:
        src = ROOT / name
        if src.exists():
            shutil.copytree(src, DIST / name, dirs_exist_ok=True)
            print(f"==> Copied {name}/")
    for name in COPY_FILES:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, DIST / name)

    llama = find_llama_server()
    if llama:
        llama_dir = llama.parent
        dest_bin = DIST / "tools" / "bin"
        shutil.copy2(llama_dir / llama.name, dest_bin / llama.name)
        for dll in llama_dir.glob("*.dll"):
            shutil.copy2(dll, dest_bin / dll.name)
        for item in dest_bin.iterdir():
            shutil.copy2(item, DIST / item.name)
        print(f"==> Bundled llama-server from {llama_dir}")
    else:
        print("WARNING: llama-server.exe not found", file=sys.stderr)

    (DIST / "README.txt").write_text(
        "\n".join(
            [
                "s0uRc3 Portable",
                "===============",
                "",
                "Double-click s0uRc3.exe, or the s0uRc3 shortcut on your Desktop.",
                "",
                "It will:",
                "  1) Place/refresh a Desktop shortcut named s0uRc3",
                "  2) Start llama-server with the GGUF in models/",
                "  3) Open a looping chat UI in this console",
                "  4) Stream thoughts, then the answer",
                "  5) Ask permission before every tool call",
                "",
                "Commands inside the UI:  /help   /models   /config   /quit",
                "",
                "CreatorCentral (creatorcentral/creatorcentral.py) is bundled",
                "alongside the exe as real source — the axiom-gate CLI works",
                "standalone too: python creatorcentral\\creatorcentral.py --help",
                "",
                "Requirements:",
                "  - Visual C++ Redistributable (usually already present)",
                "  - Python is bundled; nothing else on PATH is required",
                "",
                "One-shot (no GUI):  s0uRc3.exe --once \"list the workspace\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    portable_exe = DIST / exe.name
    if os.name == "nt" and portable_exe.exists():
        sys.path.insert(0, str(ROOT))
        from core.desktop import install_desktop_shortcut

        placed = install_desktop_shortcut(
            target=portable_exe,
            working_directory=DIST,
            arguments="",
        )
        if placed.get("ok"):
            print(f"==> Desktop shortcut: {placed.get('path')}")
        else:
            print(f"WARNING: desktop shortcut not created: {placed.get('error')}", file=sys.stderr)

    print(f"\nDONE: {DIST}")
    print(f"Launch: {portable_exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
