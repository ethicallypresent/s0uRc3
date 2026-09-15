"""Skill lifecycle manager: drafts -> test -> promote.

Skills live as single .py files with a SKILL MANIFEST docstring and a
run(**kwargs) -> dict entry point:

    \"\"\"
    SKILL MANIFEST
    name: normalize_text
    description: Trim and lowercase text
    inputs: {"text": "str"}
    outputs: {"ok": "bool", "text": "str"}
    dependencies: []
    author: self
    version: 1
    \"\"\"

    TESTS = [{"args": {"text": "  Hi "}, "expect_ok": True}]

    def run(text: str = "") -> dict:
        return {"ok": True, "text": text.strip().lower()}

New skills are written to skills/drafts/, AST-checked, executed against
their TESTS, and promoted to skills/approved/ only if every test passes.
Not a security boundary (see tools/_core/code_runner.py header) — the AST
check is a guardrail against accidents, not adversaries.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from core.memory import cosine, get_embedding_tagged

log = logging.getLogger("agent.skills")

# Weight of semantic similarity in list_skills' ranking, in units of
# "keyword hits" (see list_skills): a near-perfect semantic match (~1.0)
# outranks a couple of incidental word overlaps but never fully drowns out
# an exact keyword hit. Kept small deliberately — the embedding score is a
# tie-breaker on top of keyword overlap, not a replacement for it.
SEMANTIC_SIM_WEIGHT = 3.0

SKILL_KINDS = ("transform", "retrieve", "verify", "notify")
TRANSFORM_KEYS = ("text", "data", "output", "value", "result", "echo", "content")
RETRIEVE_KEYS = ("items", "data", "results", "hits", "records")
NOTIFY_KEYS = ("delivered", "sent", "notified")

# Imports a skill may never use. This is accident-prevention, not a jail:
# a determined author can still reach these via getattr tricks, which is why
# truly untrusted skills belong in a container (see code_runner.py).
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "urllib", "http", "ftplib",
    "smtplib", "shutil", "pathlib", "ctypes", "importlib", "multiprocessing",
    "pty", "signal",
}
BLOCKED_NAMES = {"__import__", "eval", "exec", "compile", "open"}

MANIFEST_RE = re.compile(r"SKILL MANIFEST\s*\n(.*?)\n\s*\"\"\"", re.DOTALL)


def _skills_root(paths: Any) -> Path:
    root = getattr(paths, "skills", None)
    if root is not None:
        return Path(root)
    base = getattr(paths, "root", None) or getattr(paths, "base", None) or Path(".")
    return Path(base) / "skills"


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name or "").strip("_")
    if not clean or clean.startswith("_"):
        raise ValueError(f"invalid skill name {name!r}")
    return clean[:64]


def ast_check(code: str) -> list[str]:
    """Return a list of violations; empty means the code passes."""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_IMPORTS:
                    violations.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in BLOCKED_IMPORTS:
                violations.append(f"blocked from-import: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            violations.append(f"blocked builtin use: {node.id}")
    return violations


def infer_skill_kind(name: str, description: str = "") -> str:
    hay = f"{name} {description}".lower()
    if any(w in hay for w in ("search", "find", "get", "list", "fetch", "recall", "retrieve")):
        return "retrieve"
    if any(w in hay for w in ("check", "test", "valid", "verify", "lint")):
        return "verify"
    if any(w in hay for w in ("send", "notify", "alert", "email", "message")):
        return "notify"
    return "transform"


def kind_contract_errors(kind: str, output: dict[str, Any]) -> list[str]:
    """Seed after its kind: a promoted skill must honor its family's contract."""
    if kind not in SKILL_KINDS:
        return [f"unknown skill kind {kind!r}; must be one of {SKILL_KINDS}"]
    if not isinstance(output, dict):
        return ["run() must return a dict"]
    if "ok" not in output:
        return ["run() must return an 'ok' field"]
    if kind == "transform" and not any(k in output for k in TRANSFORM_KEYS):
        return ["transform skills must return text/data/output/value/result/echo/content"]
    if kind == "retrieve" and not any(k in output for k in RETRIEVE_KEYS):
        return ["retrieve skills must return items/data/results/hits/records"]
    if kind == "notify" and not any(k in output for k in NOTIFY_KEYS):
        return ["notify skills must return delivered/sent/notified"]
    return []


def parse_manifest(code: str) -> dict[str, Any]:
    """Extract the SKILL MANIFEST block from the module docstring."""
    manifest: dict[str, Any] = {}
    m = MANIFEST_RE.search(code)
    if not m:
        # fall back: whole docstring
        try:
            doc = ast.get_docstring(ast.parse(code)) or ""
        except SyntaxError:
            doc = ""
        block = doc
    else:
        block = m.group(1)
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not key or not val:
            continue
        try:
            manifest[key] = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            manifest[key] = val
    return manifest


def _load_skill_module(code: str, name: str) -> dict[str, Any]:
    """Compile and exec skill code in a restricted namespace. Returns namespace."""
    namespace: dict[str, Any] = {"__name__": f"skill_{name}"}
    compiled = compile(code, f"<skill:{name}>", "exec")
    exec(compiled, namespace)  # noqa: S102 - AST-checked above; not a sandbox
    return namespace


def _extract_tests(namespace: dict[str, Any]) -> list[dict[str, Any]]:
    tests = namespace.get("TESTS", [])
    return tests if isinstance(tests, list) else []


class SkillManager:
    def __init__(self, paths: Any):
        self.paths = paths
        self.root = _skills_root(paths)
        self.drafts = self.root / "drafts"
        self.approved = self.root / "approved"
        self.core = self.root / "_core"
        self.archived = self.root / "archived"
        for d in (self.drafts, self.approved, self.core, self.archived):
            d.mkdir(parents=True, exist_ok=True)
        # Skill file path -> (vector, space, mtime). Embedding a description
        # is a network round-trip to the local embedding server; skills
        # rarely change within a run, so caching here avoids re-embedding
        # the whole library on every list_skills() call (gather_context
        # calls it once per turn — see core/context.py).
        self._embed_cache: dict[str, tuple[list[float], str, float]] = {}
        # Single-slot memo for the goal's own embedding: list_skills() runs
        # twice per turn on the same goal text (once directly from the loop,
        # once via gather_context), so without this the goal gets embedded
        # twice for no reason every step.
        self._last_goal_embed: tuple[str, list[float], str] | None = None

    # -- discovery ------------------------------------------------------

    def _skill_files(self, include_drafts: bool) -> list[Path]:
        dirs = [self.approved, self.core]
        if include_drafts:
            dirs.append(self.drafts)
        files: list[Path] = []
        for d in dirs:
            files.extend(sorted(d.glob("*.py")))
        return files

    def _goal_embedding(self, goal: str) -> tuple[list[float], str]:
        """Embed `goal`, memoized against the single most recent call."""
        cached = self._last_goal_embed
        if cached is not None and cached[0] == goal:
            return cached[1], cached[2]
        vec, space = get_embedding_tagged(goal)
        self._last_goal_embed = (goal, vec, space)
        return vec, space

    def _cached_embedding(self, path: Path, text: str) -> tuple[list[float], str] | None:
        """Embedding for `text`, cached per-path and invalidated on mtime change."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = str(path)
        cached = self._embed_cache.get(key)
        if cached is not None and cached[2] == mtime:
            return cached[0], cached[1]
        vec, space = get_embedding_tagged(text)
        self._embed_cache[key] = (vec, space, mtime)
        return vec, space

    def list_skills(self, goal: str = "", k: int = 8, include_drafts: bool = True) -> list[dict[str, Any]]:
        """Return skill manifests, ranked by keyword overlap plus semantic
        similarity to `goal`. Keyword overlap alone misses a skill whose
        description is worded differently from the goal (e.g. "clean up
        this text" vs. a skill described as "trim and lowercase") — the
        semantic term catches that the same way memory recall already does
        (see core/memory.py's embedding-based retrieve)."""
        goal_words = set(re.findall(r"[a-z]+", goal.lower()))
        goal_vec: list[float] | None = None
        goal_space = ""
        if goal.strip():
            goal_vec, goal_space = self._goal_embedding(goal)
        scored: list[tuple[float, dict[str, Any]]] = []
        for path in self._skill_files(include_drafts):
            try:
                code = path.read_text(encoding="utf-8")
            except OSError:
                continue
            manifest = parse_manifest(code)
            name = manifest.get("name") or path.stem
            description = str(manifest.get("description", ""))
            haystack = f"{name} {description}".lower()
            score: float = sum(1 for w in goal_words if w and w in haystack)
            if goal_vec is not None and description:
                embedded = self._cached_embedding(path, f"{name}: {description}")
                if embedded is not None:
                    vec, space = embedded
                    if space == goal_space:
                        score += SEMANTIC_SIM_WEIGHT * cosine(goal_vec, vec)
            # approved skills outrank drafts on ties
            if path.parent == self.approved:
                score += 1
            kind = manifest.get("kind") or infer_skill_kind(str(name), description)
            scored.append((score, {
                "name": name,
                "description": description,
                "kind": kind,
                "inputs": manifest.get("inputs", {}),
                "outputs": manifest.get("outputs", {}),
                "version": manifest.get("version", 1),
                "status": "draft" if path.parent == self.drafts else "approved",
                "path": str(path),
            }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def get_code(self, name: str) -> tuple[Path, str] | tuple[None, None]:
        safe = _sanitize_name(name)
        for d in (self.approved, self.core, self.drafts):
            path = d / f"{safe}.py"
            if path.exists():
                return path, path.read_text(encoding="utf-8")
        return None, None

    # -- execution ------------------------------------------------------

    def run(self, name: str, args: dict[str, Any] | None = None, allow_draft: bool = False) -> dict[str, Any]:
        safe = _sanitize_name(name)
        search = [self.approved, self.core] + ([self.drafts] if allow_draft else [])
        path = next((d / f"{safe}.py" for d in search if (d / f"{safe}.py").exists()), None)
        if path is None:
            return {"ok": False, "error": f"skill not found: {safe!r}"}
        try:
            code = path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"could not read skill: {exc}"}
        violations = ast_check(code)
        if violations:
            return {"ok": False, "error": f"skill failed safety check: {violations}"}
        try:
            namespace = _load_skill_module(code, safe)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"skill load failed: {exc}"}
        run_fn = namespace.get("run")
        if not callable(run_fn):
            return {"ok": False, "error": "skill has no run() entry point"}
        try:
            out = run_fn(**(args or {}))
        except TypeError as exc:
            return {"ok": False, "error": f"bad args for run(): {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"skill raised: {exc}"}
        if not isinstance(out, dict):
            return {"ok": False, "error": "skill run() must return a dict"}
        out.setdefault("ok", True)
        return out

    # -- authoring ------------------------------------------------------

    def run_skill_tests(self, name: str, code: str, *, kind: str | None = None) -> dict[str, Any]:
        """AST-check, load, run TESTS. Does not write files. Pass/fail is harness-only."""
        try:
            safe = _sanitize_name(name)
        except ValueError as exc:
            return {"ok": False, "name": name, "error": str(exc), "failures": [], "test_results": []}

        violations = ast_check(code)
        if violations:
            failure = {"test": "ast", "passed": False, "error": f"safety check failed: {violations}"}
            return {
                "ok": False,
                "name": safe,
                "error": failure["error"],
                "failures": [failure],
                "test_results": [failure],
            }

        try:
            namespace = _load_skill_module(code, safe)
        except Exception as exc:  # noqa: BLE001
            failure = {"test": "load", "passed": False, "error": f"draft failed to load: {exc}"}
            return {
                "ok": False,
                "name": safe,
                "error": failure["error"],
                "failures": [failure],
                "test_results": [failure],
            }

        run_fn = namespace.get("run")
        if not callable(run_fn):
            failure = {"test": "load", "passed": False, "error": "draft has no run() entry point"}
            return {
                "ok": False,
                "name": safe,
                "error": failure["error"],
                "failures": [failure],
                "test_results": [failure],
            }

        tests = _extract_tests(namespace)
        manifest = parse_manifest(code)
        skill_kind = str(kind or manifest.get("kind") or infer_skill_kind(safe, str(manifest.get("description") or "")))
        if skill_kind not in SKILL_KINDS:
            failure = {
                "test": "kind",
                "passed": False,
                "error": f"skill kind {skill_kind!r} is not after a known kind {SKILL_KINDS}",
            }
            return {
                "ok": False,
                "name": safe,
                "kind": skill_kind,
                "error": failure["error"],
                "failures": [failure],
                "test_results": [failure],
            }
        results: list[dict[str, Any]] = []
        all_pass = True
        for i, test in enumerate(tests):
            targs = test.get("args", {}) if isinstance(test, dict) else {}
            expect_ok = test.get("expect_ok", True) if isinstance(test, dict) else True
            out: Any = None
            try:
                out = run_fn(**targs)
                passed = isinstance(out, dict) and bool(out.get("ok")) == bool(expect_ok)
                results.append({"test": i, "passed": passed, "output": out})
            except Exception as exc:  # noqa: BLE001
                passed = False
                results.append({"test": i, "passed": False, "error": str(exc)})
            if passed and isinstance(out, dict):
                contract = kind_contract_errors(skill_kind, out)
                if contract:
                    passed = False
                    results[-1]["passed"] = False
                    results[-1]["error"] = "; ".join(contract)
            all_pass = all_pass and passed

        if not tests:
            try:
                out = run_fn()
                smoke_ok = isinstance(out, dict)
                contract = kind_contract_errors(skill_kind, out) if smoke_ok else ["smoke did not return a dict"]
                if contract:
                    smoke_ok = False
                row: dict[str, Any] = {"test": "smoke", "passed": smoke_ok, "output": out}
                if contract:
                    row["error"] = "; ".join(contract)
                results.append(row)
                all_pass = smoke_ok
            except Exception as exc:  # noqa: BLE001
                results.append({"test": "smoke", "passed": False, "error": str(exc)})
                all_pass = False

        failures = [r for r in results if not r.get("passed")]
        return {
            "ok": all_pass,
            "name": safe,
            "kind": skill_kind,
            "test_results": results,
            "failures": failures,
            "error": None if all_pass else "tests failed; draft kept for revision",
        }

    def create_draft(self, name: str, code: str, *, kind: str | None = None) -> dict[str, Any]:
        """Write a draft, AST-check it, run its TESTS, promote if all pass."""
        try:
            safe = _sanitize_name(name)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "promoted": False}

        violations = ast_check(code)
        if violations:
            return {"ok": False, "error": f"safety check failed: {violations}", "promoted": False}

        draft_path = self.drafts / f"{safe}.py"
        draft_path.write_text(code, encoding="utf-8")
        log.info("Draft skill written: %s", draft_path)

        tested = self.run_skill_tests(safe, code, kind=kind)
        results = tested.get("test_results") or []
        skill_kind = tested.get("kind")
        if tested.get("ok"):
            dest = self.approved / f"{safe}.py"
            if dest.exists():
                self._archive_previous_version(safe, dest)
            shutil.move(str(draft_path), str(dest))
            log.info("Skill promoted: %s -> %s", safe, dest)
            return {
                "ok": True,
                "name": safe,
                "promoted": True,
                "kind": skill_kind,
                "path": str(dest),
                "test_results": results,
            }

        if tested.get("error") and not results:
            return {
                "ok": False,
                "error": tested["error"],
                "promoted": False,
                "path": str(draft_path),
                "test_results": results,
            }

        log.info(
            "Skill %s stayed in drafts (%d/%d tests passed)",
            safe,
            sum(1 for r in results if r.get("passed")),
            len(results),
        )
        return {
            "ok": False,
            "name": safe,
            "promoted": False,
            "path": str(draft_path),
            "test_results": results,
            "error": tested.get("error") or "tests failed; draft kept for revision",
        }

    def _archive_previous_version(self, safe: str, dest: Path) -> None:
        """Keep the skill being replaced instead of silently losing it.

        create_draft() promotes a new draft over an existing approved skill
        of the same name as soon as the new code passes its own tests — that
        only proves the new version isn't broken, not that it's better than
        what it's replacing. Without this, a regression that still passes
        its own TESTS permanently overwrites the last known-good version
        with no way back (Voyager's skill library keeps a `NameV2` alongside
        the original for the same reason)."""
        existing = 0
        version_re = re.compile(rf"^{re.escape(safe)}\.v(\d+)\.py$")
        for f in self.archived.glob(f"{safe}.v*.py"):
            m = version_re.match(f.name)
            if m:
                existing = max(existing, int(m.group(1)))
        try:
            shutil.copy2(str(dest), str(self.archived / f"{safe}.v{existing + 1}.py"))
        except OSError as exc:
            log.warning("Could not archive previous version of %s before overwrite: %s", safe, exc)

    def archive(self, name: str) -> dict[str, Any]:
        safe = _sanitize_name(name)
        for d in (self.approved, self.drafts):
            path = d / f"{safe}.py"
            if path.exists():
                dest = self.archived / f"{safe}.py"
                shutil.move(str(path), str(dest))
                return {"ok": True, "archived": str(dest)}
        return {"ok": False, "error": f"skill not found: {safe!r}"}
