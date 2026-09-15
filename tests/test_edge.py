"""Attack-first edge tests. Hostile input must fail closed at the boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.launcher import parse_port, validate_launch
from core.loop import AgentLoop
from core.model import choice_from_spec
from core.paths import AgentPaths
from core.permission import inspect_request
from core.tool_registry import ToolRegistry


def _mini_root(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    for sub in ("brain", "core", "db", "skills", "tools", "workspace", "models"):
        (root / sub).mkdir(parents=True)
    (root / "brain" / "constitution.md").write_text("# law\n")
    (root / "brain" / "system_prompt.md").write_text("# identity\n")
    (root / "brain" / "reasoning_config.json").write_text(
        json.dumps({"model": "test", "max_steps": 5, "llm": {"model": "test", "num_ctx": 4096}})
    )
    (root / "models" / "ok.gguf").write_bytes(b"gguf")
    return root


def _tools(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(AgentPaths.discover(start=_mini_root(tmp_path)))


# --- NUL bytes ---


def test_nul_in_tool_arg_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("write_file", {"path": "workspace/x.txt", "content": "ok\x00evil"})
    assert out["ok"] is False
    assert out["error"] == "nul_byte"


def test_nul_in_path_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("read_file", {"path": "workspace/\x00secret"})
    assert out["ok"] is False
    assert out["error"] == "nul_byte"


def test_nul_in_permission_payload_rejected_before_approval():
    called = []

    def on_perm(req):
        called.append(req)
        return "allow"

    loop = AgentLoop(AgentPaths.discover())
    loop.on_permission = on_perm
    try:
        ok, _req, err = inspect_request(
            {"kind": "use_tool", "name": "write_file", "args": {"content": "a\x00b"}}
        )
        assert ok is False
        assert err == "nul_byte"
        decision = loop._ask_permission(
            {"kind": "use_tool", "name": "write_file", "args": {"content": "a\x00b"}}
        )
        assert decision == "deny"
        assert called == []
    finally:
        loop.close()


# --- size cap (no truncate) ---


def test_5mb_string_arg_rejected(tmp_path: Path):
    blob = "x" * (5 * 1024 * 1024)
    out = _tools(tmp_path).call("write_file", {"path": "workspace/x.txt", "content": blob})
    assert out["ok"] is False
    assert out["error"] == "args_too_large"


# --- type strictness: args must be a JSON object ---


def test_args_array_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("list_dir", ["workspace"])  # type: ignore[arg-type]
    assert out["ok"] is False
    assert out["error"] == "args_must_be_object"


def test_args_string_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("list_dir", "workspace")  # type: ignore[arg-type]
    assert out["ok"] is False
    assert out["error"] == "args_must_be_object"


def test_args_number_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("list_dir", 42)  # type: ignore[arg-type]
    assert out["ok"] is False
    assert out["error"] == "args_must_be_object"


def test_args_null_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("list_dir", None)
    assert out["ok"] is False
    assert out["error"] == "args_must_be_object"


# --- tool name allowlist ---


def test_empty_tool_name_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("", {"path": "."})
    assert out["ok"] is False
    assert out["error"] == "unregistered_tool"


def test_path_traversal_tool_name_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("../../evil", {"path": "."})
    assert out["ok"] is False
    assert out["error"] == "unregistered_tool"


def test_shell_tool_name_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("rm -rf", {"path": "/"})
    assert out["ok"] is False
    assert out["error"] == "unregistered_tool"


def test_unregistered_tool_name_rejected(tmp_path: Path):
    out = _tools(tmp_path).call("not_a_tool", {})
    assert out["ok"] is False
    assert out["error"] == "unregistered_tool"


# --- permission oversized blob never reaches approval ---


def test_oversized_permission_code_blob_never_reaches_approval():
    called = []

    def on_perm(req):
        called.append(req)
        return "allow"

    loop = AgentLoop(AgentPaths.discover())
    loop.on_permission = on_perm
    try:
        blob = {"kind": "create_skill", "name": "x", "args": {"code": "a" * (5 * 1024 * 1024)}}
        ok, _req, err = inspect_request(blob)
        assert ok is False
        assert err == "args_too_large"
        assert loop._ask_permission(blob) == "deny"
        assert called == []
    finally:
        loop.close()


# --- model specs / ports ---


def test_model_spec_evil_exe_refused(tmp_path: Path):
    root = _mini_root(tmp_path)
    (root / "models" / "evil.exe").write_bytes(b"mz")
    assert choice_from_spec("evil.exe", root, {}) is None
    gate = validate_launch(root=root, model_path="models/evil.exe", port=8080)
    assert gate["ok"] is False
    assert gate["error"] == "not_a_gguf"


def test_model_spec_parent_escape_refused(tmp_path: Path):
    root = _mini_root(tmp_path)
    outside = tmp_path / "escape.gguf"
    outside.write_bytes(b"gguf")
    assert choice_from_spec("../escape.gguf", root, {}) is None
    gate = validate_launch(root=root, model_path="../escape.gguf", port=8080)
    assert gate["ok"] is False
    assert gate["error"] == "model_not_under_models"


def test_model_spec_empty_refused(tmp_path: Path):
    root = _mini_root(tmp_path)
    assert choice_from_spec("", root, {}) is None
    gate = validate_launch(root=root, model_path="", port=8080)
    assert gate["ok"] is False


def test_port_abc_refused():
    port, err = parse_port("abc")
    assert port is None and err == "invalid_port"
    gate = validate_launch(root=Path("."), model_path="models/ok.gguf", port="abc")
    assert gate["ok"] is False
    assert gate["error"] == "invalid_port"


def test_port_negative_refused():
    port, err = parse_port(-1)
    assert port is None and err == "invalid_port"


def test_port_99999_refused():
    port, err = parse_port(99999)
    assert port is None and err == "invalid_port"


def test_read_file_on_directory_gives_clear_error_not_raw_oserror(tmp_path: Path):
    """Confirmed live: the model called read_file on the workspace directory
    itself (no filename), and target.read_text() on a directory raised a raw
    OSError — PermissionError errno 13 on Windows specifically — which reads
    like a permissions bug and gives the model nothing to correct from."""
    tools = _tools(tmp_path)
    result = tools.call("read_file", {"path": "workspace"})
    assert result["ok"] is False
    assert "directory" in result["error"].lower()
    assert "list_dir" in result["error"]
