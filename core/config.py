"""Config is a versioned contract. Load a copy; save through backup + atomic replace."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from core.contracts import (
    BACKENDS,
    CONFIG_VERSION,
    KNOWN_LLM_DEFAULTS,
    NUM_CTX_MAX,
    NUM_CTX_MIN,
    PORT_MAX,
    PORT_MIN,
)
from core.paths import AgentPaths

log = logging.getLogger("agent.config")


class ConfigError(ValueError):
    """Hostile or malformed reasoning_config.json."""


def _parse_port(raw: Any) -> tuple[int | None, str | None]:
    """Allowlist: integer in [PORT_MIN, PORT_MAX]. Strings that are not digits fail."""
    if isinstance(raw, bool):
        return None, "invalid_port"
    if isinstance(raw, int):
        port = raw
    elif isinstance(raw, str) and raw.strip().isdigit():
        port = int(raw.strip())
    else:
        return None, "invalid_port"
    if port < PORT_MIN or port > PORT_MAX:
        return None, "invalid_port"
    return port, None


def _contains_nul(value: Any) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, bytes):
        return b"\x00" in value
    if isinstance(value, dict):
        return any(_contains_nul(k) or _contains_nul(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_nul(item) for item in value)
    return False


def backup_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.name + ".bak")


def validate(cfg: Any) -> list[str]:
    """Return issue codes. Empty list means the object is usable."""
    if not isinstance(cfg, dict):
        return ["config_not_object"]
    issues: list[str] = []
    llm = cfg.get("llm")
    if llm is None:
        llm = {}
    if not isinstance(llm, dict):
        issues.append("llm_not_object")
        return issues
    backend = str(llm.get("backend") or cfg.get("backend") or "llamacpp").lower()
    if backend not in BACKENDS:
        issues.append("unknown_backend")
    try:
        ctx = int(llm.get("num_ctx") or cfg.get("num_ctx") or NUM_CTX_MIN)
        if ctx < NUM_CTX_MIN or ctx > NUM_CTX_MAX:
            issues.append("num_ctx_out_of_range")
    except (TypeError, ValueError):
        issues.append("num_ctx_not_int")
    ls = cfg.get("llama_server")
    if ls is None:
        ls = {}
    if ls and not isinstance(ls, dict):
        issues.append("llama_server_not_object")
    elif isinstance(ls, dict) and ls.get("port") is not None:
        _port, err = _parse_port(ls["port"])
        if err:
            issues.append("invalid_port")
    model_path = str((ls or {}).get("model_path") or "")
    if model_path:
        posix = Path(model_path).as_posix()
        if _contains_nul(model_path) or ".." in Path(model_path).parts:
            issues.append("model_path_nul")
        elif not posix.endswith(".gguf") or not posix.startswith("models/"):
            issues.append("model_not_under_models")
    return issues


def load(paths: AgentPaths) -> dict[str, Any]:
    """Read config, merge defaults into a *copy*. Never writes."""
    path = paths.effective_config_path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config is not JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    issues = validate(raw)
    hostile = [i for i in issues if i not in ("unknown_backend",)]
    fatal = {
        "config_not_object",
        "llm_not_object",
        "model_path_nul",
        "model_not_under_models",
        "invalid_port",
    }
    if any(i in fatal for i in issues):
        raise ConfigError("; ".join(issues))
    if issues:
        log.warning("config issues (continuing with defaults): %s", issues)
    cfg = copy.deepcopy(raw)
    llm = dict(KNOWN_LLM_DEFAULTS)
    llm.update(cfg.get("llm") or {})
    cfg["llm"] = llm
    if not cfg.get("config_version"):
        cfg["config_version"] = CONFIG_VERSION
    return cfg


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def snapshot(paths: AgentPaths) -> bytes | None:
    path = paths.effective_config_path
    try:
        return path.read_bytes()
    except OSError:
        return None


def restore(paths: AgentPaths, previous: bytes | None) -> bool:
    """Roll config back to a snapshot. Designed before any save."""
    if previous is None:
        return False
    path = paths.effective_config_path
    try:
        _atomic_write(path, previous.decode("utf-8"))
        return True
    except OSError as exc:
        log.error("config restore failed: %s", exc)
        return False


def save(paths: AgentPaths, data: dict[str, Any]) -> None:
    """Backup current file, then atomic-replace. Increments config_revision."""
    if not isinstance(data, dict):
        raise ConfigError("cannot save non-object config")
    issues = validate(data)
    if "model_path_nul" in issues or "llm_not_object" in issues:
        raise ConfigError("; ".join(issues))
    path = paths.effective_config_path
    payload = copy.deepcopy(data)
    payload["config_version"] = str(payload.get("config_version") or CONFIG_VERSION)
    try:
        payload["config_revision"] = int(payload.get("config_revision") or 0) + 1
    except (TypeError, ValueError):
        payload["config_revision"] = 1
    bak = backup_path(path)
    if path.exists():
        try:
            bak.write_bytes(path.read_bytes())
        except OSError as exc:
            log.warning("could not write config backup: %s", exc)
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def merge_and_save(
    paths: AgentPaths,
    *,
    top: dict[str, Any] | None = None,
    llm: dict[str, Any] | None = None,
    llama_server: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy-on-write merge, then save. Returns the new on-disk object."""
    try:
        data = json.loads(paths.effective_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data = copy.deepcopy(data)
    if top:
        data.update(top)
    if llm:
        block = dict(data.get("llm") or {})
        block.update(llm)
        data["llm"] = block
        if llm.get("model"):
            data["model"] = llm["model"]
        if llm.get("fallback_model"):
            data["fallback_model"] = llm["fallback_model"]
    if llama_server:
        block = dict(data.get("llama_server") or {})
        block.update(llama_server)
        data["llama_server"] = block
    save(paths, data)
    return data
