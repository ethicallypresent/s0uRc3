import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.brain import CONSTITUTION_MARK, load_system_prompt
from core.loop import AgentLoop, parse_model_output
from core.paths import AgentPaths, PathGuard
from core.skill_manager import SkillManager
from tools._core.code_runner import run_python


def test_parse_action():
    raw = """<think>ok</think>
```json
{"action": "finish", "finish": {"status": "success", "summary": "done"}}
```"""
    think, action = parse_model_output(raw)
    assert think == "ok"
    assert action["action"] == "finish"


def test_think_block_is_lowercased_but_finish_summary_is_not():
    """brain/system_prompt.md's "thinking" rule applies only to the
    reasoning trace, never to user-facing text — the code-level backstop
    must match that scope exactly, not lowercase everything."""
    raw = """<think>Intent: Do The Thing. Risk: NONE.</think>
```json
{"action": "finish", "finish": {"status": "success", "summary": "Done. The Thing Worked."}}
```"""
    think, action = parse_model_output(raw)
    assert think == "intent: do the thing. risk: none."
    assert action["finish"]["summary"] == "Done. The Thing Worked."


def test_path_guard_blocks_core():
    paths = AgentPaths.discover()
    guard = PathGuard(paths)
    try:
        guard.resolve_writable("core/loop.py")
        raise AssertionError("should have blocked")
    except PermissionError:
        pass


def test_code_runner():
    out = run_python("print(2+2)")
    assert out["ok"] is True
    assert "4" in out["stdout"]


def test_skill_roundtrip(tmp_name="echo_skill"):
    paths = AgentPaths.discover()
    sm = SkillManager(paths)
    code = '''"""
SKILL MANIFEST
name: echo_skill
description: echo kwargs
inputs: {}
outputs: {"ok": "bool"}
dependencies: []
author: self
version: 1
"""

def run(**kwargs):
    return {"ok": True, "echo": kwargs}
'''
    created = sm.create_draft(tmp_name, code)
    assert created["ok"] is True
    ran = sm.run(tmp_name, {"x": 1})
    assert ran["ok"] is True
    assert ran["echo"]["x"] == 1


def test_dry_run_loop():
    loop = AgentLoop(AgentPaths.discover())
    result = loop.run("List workspace", dry_run=True)
    assert result["ok"] is True
    assert result["steps"] >= 1
    assert result.get("run_state", {}).get("identity_conserved") is True
    loop.memory.close()


def test_brain_is_organ():
    text = load_system_prompt(AgentPaths.discover())
    assert CONSTITUTION_MARK in text
    assert "Constitution (binding)" in text


def test_self_improves_when_capability_missing():
    loop = AgentLoop(AgentPaths.discover())
    result = loop.run("Extend your capabilities with a text normalizer skill", dry_run=True)
    assert result["ok"] is True
    approved = AgentPaths.discover().skills / "approved" / "normalize_text.py"
    assert approved.exists(), result


if __name__ == "__main__":
    test_parse_action()
    test_path_guard_blocks_core()
    test_code_runner()
    test_skill_roundtrip()
    test_dry_run_loop()
    test_brain_is_organ()
    test_self_improves_when_capability_missing()
    print("all tests passed")
