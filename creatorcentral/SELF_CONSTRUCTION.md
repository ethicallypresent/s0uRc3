# Self-construction: s0uRc3 walks itself through CREATE

Per the handoff brief's Part 3: before building anything for Nathanael,
s0uRc3 satisfies each CREATE axiom for its own creation. This wasn't
written as prose first and formatted after — it's the record of an actual
run of the CLI this document describes:

```bash
python creatorcentral.py init s0urc3-v1 --owner Nathanael --purpose "..."
python creatorcentral.py log s0urc3-v1 <requirement> "<evidence>"   # x20
python creatorcentral.py advance s0urc3-v1                          # x10
python creatorcentral.py close s0urc3-v1
```

Full source of truth: [`projects/s0urc3-v1.json`](projects/s0urc3-v1.json)
— every entry below is quoted from its `log`, not reconstructed from
memory. Final state: `status: done`, `axiom: REST` — the gate was never
bypassed; every `advance` call in the log succeeded because the
requirement before it was actually met.

## 1. INITIATION
**What:** s0uRc3 v1: a local Builder Agent forked from Kurama, plus
`creatorcentral/` — a gated CLI that walks any software build through the
ten CREATE axioms with enforced evidence.
**Why:** Nathanael's brief called for a software-building agent
constructed from the CREATE/DECONSTRUCT axiom system itself, dogfooding
the framework on its own build before using it on anything else.
**Owner:** Nathanael (final authority); built by s0uRc3 / Claude Sonnet 5
in this session.

## 2. IMMUTABILITY
**Where the history lives:** a local git repo at
`C:\Users\natha\OneDrive\Documents\Naruto\s0uRc3`, cloned with full commit
history from `kurama-local-agent` (commit `1d5ce4f`) so nothing before
this fork is lost.
**First commit:** inherited `1d5ce4f "Initial commit: Kurama local
agent"`; this build's own work lands as new commits in this repo's own
history.

## 3. NAMING
**Name:** s0uRc3 — chosen by Nathanael directly (stylized; not "Source" or
"S0urce").
**Glossary:** *axiom* = one of the ten CREATE stages. *requirement* = a
specific piece of evidence an axiom needs. *evidence* = a logged,
timestamped text answer to a requirement. *gate* = `advance_axiom`'s
refusal to proceed without evidence. *close-out* = `done` (reached REST)
vs. `closed_early` (stopped before REST, with a reason).

## 4. ORDERING
**Layout:** `creatorcentral/` (axioms.json, creatorcentral.py, projects/,
README.md, this file) holds the gate itself, independent of the LLM loop.
`core/tool_registry.py` + `tools/_core/creatorcentral_tools.py` wire it
into the conversational engine as six tools. `brain/` carries the s0uRc3
identity. `tui/`, `skills/`, `models/` are inherited from Kurama, dormant.
**Build order:** (1) `axioms.json` data, (2) `creatorcentral.py` CLI +
tests — foundation, usable standalone with zero LLM, (3) wire as agent
tools (`tool_registry.py`, `manifest.json`, the loop's dry-run demo),
(4) identity swap (`brain/`, `main.py`), (5) this document, produced last
using the tool it describes.

## 5. ADVERSITY
**Failure modes:** (a) hallucinated APIs — e.g. claiming `advance_axiom`
succeeded when it refused; (b) untested code handed over; (c) skipped
gates under pressure — closing a project "done" when it isn't at REST;
(d) scope creep — building UNBUILD/adapters/multi-project before v1 is
solid.
**Mitigations:** (a) `advance_axiom`/`close_project` return structured
`ok`/`error` the caller must check, never silently assumed; (b)
`tests/test_creatorcentral.py` (9 tests) run before any claim of
"working"; (c) `close_project`'s own code branches on
`status == "at_rest"` and cannot be told to lie — it always records
`closed_early` otherwise; (d) `creatorcentral/README.md`'s backlog section
holds scope-creep items out of v1 explicitly.

## 6. FIDELITY
**Spec source:** the handoff brief ("HANDOFF BRIEF — The Builder Agent")
plus Nathanael's own words in this conversation (fork target, name,
file-disposition, and v1-scope decisions) — his words win when they
conflict with the brief's suggested defaults.
**Drift check:** checked against Part 4/6 of the brief:
`creatorcentral.py`'s CLI (init/status/check/advance/log/close) matches
exactly; `axioms.json` matches Part 1's ten axioms; the advance gate is
tested and enforced as specified; UNBUILD/adapters/lifecycle-closer are
explicitly deferred per Nathanael's chosen v1 scope, not silently
dropped.

## 7. MIGRATION
**Inventory:** the full Kurama codebase (`core/`, `tui/`, `brain/`,
`models/`, `skills/`, `tools/_core/`, `db/`, `workspace/`) and its git
history; Nathanael's constraints (zero budget, local-first, stdlib-first,
Windows).
**Migration plan:** Kurama's engine kept in place and reused as the
conversational shell. `core/tool_registry.py`, `core/loop.py`,
`core/reflection.py`, `main.py`, and `brain/{system_prompt,
constitution}.md` rewritten for the new identity and tool set. Everything
else carried forward untouched and dormant, per Nathanael's explicit
choice, not deleted.

## 8. RECOVERY
**Backup:** the original `kurama-local-agent` working directory
(`C:\Users\natha\OneDrive\Documents\Naruto\agent`) is untouched and
remains a complete, working backup; this fork's own history is a git
repo, so every step here is a commit that can be reverted.
**Rollback:** `git revert`/`reset` within the s0uRc3 repo undoes any
single change; in the worst case, deleting the s0uRc3 folder loses
nothing, since the original Kurama repo it was cloned from was never
modified.

## 9. PROTOCOL
**Interfaces:** Nathanael talks to s0uRc3 in chat (brief + plain
instructions). s0uRc3 talks to CreatorCentral via six tools dispatched
through `core/tool_registry.py`, which call
`tools/_core/creatorcentral_tools.py`, which calls
`creatorcentral/creatorcentral.py` — the same functions the standalone
CLI uses, so the CLI and the agent loop can never disagree about what the
gate allows.
**Handoff doc:** `creatorcentral/README.md` (CLI reference + backlog) and
this file are the two docs a stranger, or a future agent, would read to
pick this up.

## 10. REST
**Definition of done for v1:**
1. `creatorcentral.py`'s six operations exist and are gate-enforced —
   `advance` refuses incomplete axioms, `close` is honest about early vs.
   done.
2. `tests/test_creatorcentral.py` passes.
3. The six operations are wired into the agent loop as tools.
4. This self-construction document exists and was produced by walking
   the CLI itself through all ten axioms.

Assignments 3–4 from the brief (UNBUILD, the lifecycle-loop closer) are
explicitly **out of v1's REST** and live in `creatorcentral/README.md`'s
backlog — not silently dropped, not silently included.

**Evidence of rest:** `tests/test_creatorcentral.py`: 9/9 passed. This
project's own log (`projects/s0urc3-v1.json`) is the record of walking
every axiom above via the real `creatorcentral.py` CLI commands, not
hand-written prose. `status s0urc3-v1` reports `"status": "done"`,
`"axiom": "REST"`.
