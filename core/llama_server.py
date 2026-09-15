"""Start/stop local llama-server (llama.cpp) for Kurama.

The WinGet llama-server.exe is a tiny stub; its WorkingDirectory must be the
folder that contains llama-server-impl.dll and friends.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("agent.llama_server")

# Keyed by port so more than one llama-server instance can run at once
# (e.g. the primary "pocket" reasoning model on 8080 plus a second, tiny
# tool-call formatter model on another port). Every existing zero-arg
# caller (main.py, core/model.py, the TUI) only ever managed the primary
# server, so every function here keeps port=8080 as its default — that
# preserves their exact old behavior without touching those call sites.
_DEFAULT_PORT = 8080
_servers: dict[int, dict[str, Any]] = {}


def find_binary(root: Path) -> Path | None:
    candidates = [
        root / "tools" / "bin" / "llama-server.exe",
        root / "bin" / "llama-server.exe",
        root / "llama-server.exe",
    ]
    which = os.environ.get("PATH", "")
    for part in which.split(os.pathsep):
        if part:
            candidates.append(Path(part) / "llama-server.exe")
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if local.exists():
        candidates.extend(list(local.rglob("llama-server.exe"))[:5])
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def is_healthy(base_url: str = "http://127.0.0.1:8080/v1", timeout: float = 2.0) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def current_model_path(port: int = _DEFAULT_PORT) -> str | None:
    entry = _servers.get(port)
    return entry.get("model_path") if entry else None


def _close_logs(port: int) -> None:
    entry = _servers.get(port)
    if not entry:
        return
    for handle in entry.get("log_handles") or []:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
    entry["log_handles"] = []


def _log_paths(root: Path, alias: str) -> tuple[Path, Path]:
    folder = root / "workspace"
    folder.mkdir(parents=True, exist_ok=True)
    # The primary server keeps its original, unsuffixed filenames — other
    # instances (core/maintain.py's _LOG_NAMES doesn't know about them) are
    # namespaced by alias so they can't collide or shadow the primary's log.
    if alias in ("", "pocket"):
        return folder / "llama_server.out.log", folder / "llama_server.err.log"
    return folder / f"llama_server.{alias}.out.log", folder / f"llama_server.{alias}.err.log"


def _read_tail(path: Path, *, limit: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:].strip()


def build_server_args(
    binary: Path,
    model: Path,
    *,
    alias: str,
    host: str,
    port: int,
    num_ctx: int,
    num_thread: int,
    num_batch: int,
    num_gpu: int,
    jinja: bool = True,
    top_k: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
    dry_multiplier: float | None = None,
    dry_base: float | None = None,
    dry_allowed_length: int | None = None,
    dry_penalty_last_n: int | None = None,
) -> list[str]:
    args = [
        str(binary),
        "-m",
        str(model.resolve()),
        "--alias",
        alias,
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(max(512, num_ctx)),
        "-t",
        str(max(1, num_thread)),
        "-b",
        str(max(32, num_batch)),
        "-ngl",
        str(max(0, num_gpu)),
        "--parallel",
        "1",
        # Kurama's own loop parses <think>...</think> + a JSON action out of
        # the raw completion text (core/loop.py THINK_RE / JSON_RE). Thinking
        # models (e.g. Qwen3-*-Thinking) trigger llama-server's default
        # --reasoning-format auto, which splits <think> into a separate
        # `reasoning_content` field and leaves `content` empty until the
        # model stops thinking — the loop never sees the think block and
        # often burns the whole token budget before any JSON arrives, which
        # looks like the server "won't connect". Keep thoughts inline.
        "--reasoning-format",
        "none",
        # core/memory.py tries /v1/embeddings first and falls back to a
        # crude offline hash embedding when it 404s/501s — llama-server
        # only exposes that endpoint with this flag. Confirmed a plain
        # causal model like pocket.gguf can still serve pooled-hidden-state
        # embeddings alongside normal chat completions from the same
        # server; this doesn't disable or compete with completions.
        "--embeddings",
        # pocket.gguf's default pooling is "none" (per-token hidden states),
        # which /v1/embeddings rejects outright ("not OAI compatible") since
        # that endpoint needs one vector per input. "mean" pools token
        # states into a single vector — the standard choice for a causal
        # model with no dedicated embedding head.
        "--pooling",
        "mean",
    ]
    if jinja:
        args.append("--jinja")
    # llama-server's own defaults are --repeat-penalty 1.00 (off) and
    # --dry-multiplier 0.00 (off) — nothing discourages a small quantized
    # model from looping the same sentence inside a long <think> block.
    # Config already carries repeat_penalty/top_k/top_p; wire them through as
    # real CLI flags instead of silently dropping them (they were only ever
    # forwarded for the ollama backend's request body, never llamacpp's).
    if top_k is not None:
        args.extend(["--top-k", str(int(top_k))])
    if top_p is not None:
        args.extend(["--top-p", str(float(top_p))])
    if repeat_penalty is not None:
        args.extend(["--repeat-penalty", str(float(repeat_penalty))])
    if repeat_last_n is not None:
        args.extend(["--repeat-last-n", str(int(repeat_last_n))])
    # DRY targets exact phrase/sentence repeats specifically (unlike
    # repeat-penalty, which only tracks individual token frequency) — this is
    # the direct fix for a thinking model looping the same reasoning verbatim.
    if dry_multiplier is not None:
        args.extend(["--dry-multiplier", str(float(dry_multiplier))])
        if dry_base is not None:
            args.extend(["--dry-base", str(float(dry_base))])
        if dry_allowed_length is not None:
            args.extend(["--dry-allowed-length", str(int(dry_allowed_length))])
        if dry_penalty_last_n is not None:
            args.extend(["--dry-penalty-last-n", str(int(dry_penalty_last_n))])
    return args


def list_models(base_url: str = "http://127.0.0.1:8080/v1", timeout: float = 3.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out: list[str] = []
        for row in data.get("data") or []:
            if isinstance(row, dict) and row.get("id"):
                out.append(str(row["id"]))
        return out
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return []


def _kill_listener(port: int) -> None:
    """Best-effort: free the llama-server port so a new GGUF can bind."""
    if os.name != "nt":
        return
    try:
        cmd = (
            f"Get-NetTCPConnection -LocalPort {int(port)} -EA SilentlyContinue | "
            "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -EA SilentlyContinue }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def ensure_running(
    root: Path,
    *,
    model_path: str = "models/pocket.gguf",
    host: str = "127.0.0.1",
    port: int = 8080,
    alias: str = "pocket",
    num_ctx: int = 4096,
    num_thread: int = 6,
    num_batch: int = 128,
    num_gpu: int = 0,
    startup_timeout_sec: int = 180,
    force_restart: bool = False,
    jinja: bool = True,
    top_k: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
    dry_multiplier: float | None = None,
    dry_base: float | None = None,
    dry_allowed_length: int | None = None,
    dry_penalty_last_n: int | None = None,
) -> dict[str, Any]:
    """Ensure llama-server is serving. Starts one if needed. Returns status dict."""
    from core.launcher import parse_port, validate_launch

    port_i, port_err = parse_port(port)
    if port_err:
        return {"ok": False, "error": port_err, "from": "launcher", "observed": {}}
    base = f"http://{host}:{port_i}/v1"
    live_ids = list_models(base) if is_healthy(base) else []
    wrong_model = bool(live_ids) and alias not in live_ids
    if force_restart or wrong_model:
        log.info("Restarting llama-server (force=%s wrong_model=%s live=%s want=%s)", force_restart, wrong_model, live_ids, alias)
        stop(port_i)
        _kill_listener(port_i)
        for _ in range(16):
            if not is_healthy(base):
                break
            time.sleep(0.25)
    elif is_healthy(base):
        log.info("llama-server already healthy at %s ids=%s", base, live_ids)
        return {"ok": True, "started": False, "base_url": base, "ids": live_ids}

    gate = validate_launch(
        root=root,
        model_path=model_path,
        port=port_i,
        alias=alias,
        host=host,
    )
    if not gate.get("ok"):
        log.error("llama-server launch refused: %s", gate.get("error"))
        return gate
    model = gate["model"]
    port = gate["port"]
    alias = gate["alias"]
    host = gate["host"]
    base = f"http://{host}:{port}/v1"

    binary = find_binary(root)
    if binary is None:
        return {"ok": False, "error": "llama-server.exe not found (tools/bin or PATH)"}

    bin_dir = binary.parent
    args = build_server_args(
        binary,
        model,
        alias=alias,
        host=host,
        port=int(port),
        num_ctx=num_ctx,
        num_thread=num_thread,
        num_batch=num_batch,
        num_gpu=num_gpu,
        jinja=jinja,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        dry_multiplier=dry_multiplier,
        dry_base=dry_base,
        dry_allowed_length=dry_allowed_length,
        dry_penalty_last_n=dry_penalty_last_n,
    )
    port_i = int(port)
    out_path, err_path = _log_paths(root, alias)
    log.info("Starting llama-server: %s (cwd=%s)", " ".join(args), bin_dir)
    try:
        # Keep as a child of this process (UI/CLI owns lifetime). Do not
        # DETACHED_PROCESS — that made the stub exit unpredictably on WinGet builds.
        # cwd MUST be the stub's folder so llama-server-impl.dll resolves.
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _close_logs(port_i)
        out_h = out_path.open("w", encoding="utf-8")
        err_h = err_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            args,
            cwd=str(bin_dir),
            stdout=out_h,
            stderr=err_h,
            creationflags=creation,
        )
        _servers[port_i] = {"proc": proc, "model_path": None, "log_handles": [out_h, err_h]}
    except OSError as exc:
        _close_logs(port_i)
        return {"ok": False, "error": f"failed to spawn llama-server: {exc}"}

    atexit.register(stop, port_i)
    deadline = time.time() + startup_timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _read_tail(err_path)
            log.error("llama-server exited early code=%s log=%s", proc.returncode, tail[-800:])
            return {
                "ok": False,
                "error": f"llama-server exited early code={proc.returncode}",
                "log": tail,
            }
        if is_healthy(base):
            ids = list_models(base)
            model_path_str = str(model.resolve())
            _servers[port_i]["model_path"] = model_path_str
            log.info("llama-server ready at %s pid=%s ids=%s", base, proc.pid, ids)
            return {
                "ok": True,
                "started": True,
                "base_url": base,
                "pid": proc.pid,
                "model_path": model_path_str,
                "ids": ids,
            }
        time.sleep(0.75)
    tail = _read_tail(err_path)
    stop(port_i)
    log.error("llama-server start timed out log=%s", tail[-800:])
    return {"ok": False, "error": "timed out waiting for llama-server", "log": tail}


def stop(port: int = _DEFAULT_PORT) -> None:
    entry = _servers.get(port)
    if not entry:
        _close_logs(port)
        return
    proc: subprocess.Popen[str] | None = entry.get("proc")
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:  # noqa: BLE001
        pass
    _close_logs(port)
    _servers.pop(port, None)


def sampling_kwargs(llm: dict[str, Any]) -> dict[str, Any]:
    """Extract anti-repetition / sampling knobs from llm config for ensure_running.

    Centralized so every ensure_running caller (config auto-start, TUI model
    picker, --model preset) launches the server with the same repeat_penalty
    / DRY settings instead of some paths silently reverting to llama-server's
    own repeat_penalty=1.0 / dry_multiplier=0.0 (both effectively "off").
    """
    return {
        "top_k": int(llm["top_k"]) if llm.get("top_k") is not None else None,
        "top_p": float(llm["top_p"]) if llm.get("top_p") is not None else None,
        "repeat_penalty": float(llm["repeat_penalty"]) if llm.get("repeat_penalty") is not None else None,
        "repeat_last_n": int(llm["repeat_last_n"]) if llm.get("repeat_last_n") is not None else None,
        "dry_multiplier": float(llm["dry_multiplier"]) if llm.get("dry_multiplier") is not None else None,
        "dry_base": float(llm["dry_base"]) if llm.get("dry_base") is not None else None,
        "dry_allowed_length": int(llm["dry_allowed_length"]) if llm.get("dry_allowed_length") is not None else None,
        "dry_penalty_last_n": int(llm["dry_penalty_last_n"]) if llm.get("dry_penalty_last_n") is not None else None,
    }


def ensure_from_config(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    llm = cfg.get("llm") or {}
    ls = cfg.get("llama_server") or {}
    if not ls.get("auto_start", True):
        result = {"ok": True, "started": False, "skipped": True}
    else:
        backend = str(llm.get("backend") or ls.get("backend") or "llamacpp").lower()
        if backend not in ("llamacpp", "llama.cpp", "llama-server"):
            result = {"ok": True, "started": False, "skipped": True, "backend": backend}
        else:
            result = ensure_running(
                root,
                model_path=str(ls.get("model_path") or "models/pocket.gguf"),
                host=str(ls.get("host") or "127.0.0.1"),
                port=int(ls.get("port") or 8080),
                alias=str(ls.get("alias") or llm.get("model") or "pocket"),
                num_ctx=int(llm.get("num_ctx") or 4096),
                num_thread=int(llm.get("num_thread") or 6),
                num_batch=int(llm.get("num_batch") or 128),
                num_gpu=max(0, int(llm.get("num_gpu") or 0)),
                force_restart=bool(ls.get("force_restart") or False),
                jinja=bool(ls.get("jinja", True)),
                **sampling_kwargs(llm),
            )
    # Second, independent instance for the tool-call arg formatter
    # (core/tool_formatter.py). Best-effort and never allowed to affect the
    # primary result above — a missing/failed formatter server just means
    # core/loop.py falls back to the reasoning model's own args, same as
    # before this existed.
    fc = cfg.get("tool_formatter") or {}
    if fc.get("enabled"):
        fls = cfg.get("llama_server_tool_formatter") or {}
        try:
            fc_result = ensure_running(
                root,
                model_path=str(fls.get("model_path") or "models/functiongemma.gguf"),
                host=str(fls.get("host") or "127.0.0.1"),
                port=int(fls.get("port") or 8081),
                alias=str(fls.get("alias") or fc.get("model") or "functiongemma"),
                num_ctx=int(fc.get("num_ctx") or 4096),
                num_thread=int(fc.get("num_thread") or llm.get("num_thread") or 6),
                num_batch=int(fc.get("num_batch") or 128),
                num_gpu=max(0, int(fc.get("num_gpu") or 0)),
                force_restart=bool(fls.get("force_restart") or False),
                jinja=bool(fls.get("jinja", True)),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("tool_formatter server failed to start: %s", exc)
            fc_result = {"ok": False, "error": str(exc)}
        result["tool_formatter"] = fc_result
    return result
