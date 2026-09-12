"""s0uRc3 — local Builder Agent, forked from Kurama.

Walks software projects through the CREATE axioms (INITIATION -> REST) and,
in time, tears them down through DECONSTRUCT. See creatorcentral/README.md
and creatorcentral/SELF_CONSTRUCTION.md.

Default (no args): looping terminal GUI.
One-shot (scripts/CI): python main.py --once "goal"   or   python main.py "goal"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _enable_windows_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001
        pass


def _select_model(paths, cfg, preset: str | None) -> dict:
    from core.model import apply_choice, catalog, choice_from_spec

    if preset:
        choice = choice_from_spec(preset, paths.root, cfg)
        if choice is None:
            return {"ok": False, "error": f"unknown model: {preset}"}
        return apply_choice(paths, choice, cfg)
    items = catalog(paths.root, cfg)
    if not sys.stdin.isatty():
        from core.llama_server import ensure_from_config

        return ensure_from_config(paths.root, cfg)
    print("Select a model (llama-server ids, then GGUFs on disk):", flush=True)
    if not items:
        print("  (none listed — type a GGUF path or model id)", flush=True)
    for i, item in enumerate(items, 1):
        print(f"  {i}) {item.label}", flush=True)
    raw = input("number or path> ").strip()
    if not raw:
        return {"ok": False, "error": "no model selected"}
    if raw.isdigit() and items:
        idx = int(raw) - 1
        if 0 <= idx < len(items):
            return apply_choice(paths, items[idx], cfg)
    choice = choice_from_spec(raw, paths.root, cfg)
    if choice is None:
        return {"ok": False, "error": f"unknown model: {raw}"}
    return apply_choice(paths, choice, cfg)


def _run_once(args: argparse.Namespace) -> int:
    from core.llama_server import stop as stop_llama
    from core.loop import AgentLoop
    from core.paths import AgentPaths

    paths = AgentPaths.discover()
    loop = AgentLoop(paths)
    stream = args.stream and not args.no_stream
    if stream:

        def _emit(tok: str) -> None:
            sys.stdout.write("[[stream]]" + tok.replace("\n", "[[nl]]") + "\n")
            sys.stdout.flush()

        loop.on_token = _emit
    if not args.dry_run:
        boot = _select_model(paths, loop.cfg, getattr(args, "model", None))
        if not boot.get("ok"):
            logging.getLogger("agent").error("model select failed: %s", boot.get("error"))
            print(json.dumps({"ok": False, "error": boot.get("error")}, indent=2))
            return 1
        print(f"model: {boot}", flush=True)
        loop.close()
        loop = AgentLoop(paths)
        loop.on_token = _emit if stream else None
    goal = (args.goal or "").strip()
    print(f"starting run: {goal!r} (dry_run={args.dry_run}, stream={stream})", flush=True)
    try:
        result = loop.run(goal, dry_run=args.dry_run, max_steps=args.max_steps)
    finally:
        loop.close()
        stop_llama()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


def prepare_runtime_paths() -> None:
    """Frozen exe: agent root is next to Kurama.exe (brain/db/models), not _internal."""
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        os.environ.setdefault("AGENT_ROOT", str(root))
    else:
        root = Path(__file__).resolve().parent
    inserted = str(root)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)


def main() -> int:
    prepare_runtime_paths()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="s0uRc3 local Builder Agent")
    ap.add_argument("goal", nargs="?", help="Goal for a one-shot run (omit to open the TUI)")
    ap.add_argument("--once", action="store_true", help="One-shot JSON run (do not open the TUI)")
    ap.add_argument("--tui", action="store_true", help="Force the looping terminal GUI")
    ap.add_argument(
        "--classic-tui",
        action="store_true",
        help="Use the original full TUI (model picker + config screens) instead of the simplified one",
    )
    ap.add_argument("--dry-run", action="store_true", help="Skip the LLM, use deterministic policy")
    ap.add_argument("--max-steps", type=int, default=None, help="Cap on reasoning steps")
    ap.add_argument("--stream", action="store_true", default=True, help="Stream think/text tokens to stdout")
    ap.add_argument("--no-stream", action="store_true", help="Disable token streaming (one-shot only)")
    ap.add_argument(
        "--model",
        default=None,
        help="GGUF path or llama-server model id (skips the startup picker)",
    )
    args = ap.parse_args()

    use_tui = args.tui or (not args.once and not args.goal)
    if use_tui:
        _enable_windows_vt()
        if getattr(sys, "frozen", False):
            from core.desktop import install_desktop_shortcut

            install_desktop_shortcut()
        if args.classic_tui:
            from tui.app import run_app

            return run_app(dry_run=args.dry_run, max_steps=args.max_steps, model=args.model)
        from tui.simple_app import run_simple_app

        return run_simple_app(dry_run=args.dry_run, max_steps=args.max_steps, model=args.model)

    goal = (args.goal or input("goal> ")).strip()
    if not goal:
        print("no goal given", file=sys.stderr)
        return 2
    args.goal = goal
    return _run_once(args)


if __name__ == "__main__":
    sys.exit(main())
