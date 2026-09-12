# CreatorCentral

A gated CLI for walking a build through the ten CREATE axioms:
INITIATION -> IMMUTABILITY -> NAMING -> ORDERING -> ADVERSITY -> FIDELITY ->
MIGRATION -> RECOVERY -> PROTOCOL -> REST.

Stdlib only (`argparse`, `json`, `pathlib`, `datetime`). One JSON file per
project under `projects/<name>.json`. The `advance` command is the gate: it
will not move a project past an axiom whose requirements lack evidence, and
it always names exactly what's missing instead of failing silently.

## Usage

```bash
python creatorcentral.py init myproj --owner alice --purpose "what this is"
python creatorcentral.py log myproj what "One sentence describing the build"
python creatorcentral.py log myproj why "Why it needs to exist"
python creatorcentral.py log myproj owner "alice"
python creatorcentral.py check myproj      # read-only: met vs missing
python creatorcentral.py advance myproj    # gate: refuses if anything missing
python creatorcentral.py status myproj     # full state + log
python creatorcentral.py status            # list all projects
python creatorcentral.py close myproj --reason "why it stopped here"
```

`close` is honest: it only records `done` if the project actually reached
REST (advanced past axiom 10). Anything closed before that is recorded as
`closed_early` with the given reason — never silently reclassified as done.

## Requirements per axiom

See [`axioms.json`](axioms.json) for the full data (id, name, description,
requirement keys). `log <project> <requirement-key> <text>` attaches
evidence to one requirement of the *current* axiom; `advance` checks all of
that axiom's requirement keys have evidence before moving on.

## Files

| Path | What |
| --- | --- |
| `creatorcentral.py` | The CLI + the library functions it wraps (`init_project`, `log_evidence`, `check_project`, `advance_project`, `close_project`, `status_one`/`status_all`) |
| `axioms.json` | The ten axioms as data — edit the runbook without touching code |
| `projects/*.json` | One state file per tracked project, committed as evidence (IMMUTABILITY) |
| `SELF_CONSTRUCTION.md` | s0uRc3's own walk through these ten axioms, for its own v1 build |

## Backlog (not built in v1)

These are the brief's assignments 3-4 — deliberately deferred so v1 stays
small and solid:

- **Multi-project selection** — a `--project` default / active-project
  pointer so commands don't require the name every time.
- **Adapters** — `git` (auto-log a commit hash as evidence for IMMUTABILITY),
  `test` (auto-log a passing test run as evidence for ADVERSITY/FIDELITY),
  `backup` (auto-log a backup location for RECOVERY) — so evidence gets
  attached automatically instead of typed by hand where a tool already
  knows the answer.
- **UNBUILD** — the DECONSTRUCT mirror: ten teardown operations
  (declare the ending, close the record, revoke identity, unwrap
  dependencies in reverse, test the absence, sweep for zombies, migrate
  dependents before removal, preserve the archive, terminate contracts,
  reach stable rest) as a gated CLI with the same shape as this one —
  `unbuild.py` + `deconstruct_operations.json` + `cases/<name>.json`.
- **Lifecycle loop closer** — a single command that takes a CreatorCentral
  project that reached REST and opens it as an UNBUILD case file, so a
  build's full life (birth to rest to eventual teardown) is one system
  instead of two disconnected tools.
