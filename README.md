# s0uRc3

A local Builder Agent, forked from [Kurama](https://github.com/ethicallypresent/kurama-local-agent). Same self-contained engine — it owns its prompt, memory, tools, and skill lifecycle, with a **looping terminal GUI** (`python main.py`) — repurposed around one rule: every real build moves through the ten CREATE axioms, gated, with evidence, tracked by [`creatorcentral/`](creatorcentral/README.md). s0uRc3's own construction is documented the same way: [`creatorcentral/SELF_CONSTRUCTION.md`](creatorcentral/SELF_CONSTRUCTION.md).

Kurama's TUI, skill lifecycle, and Windows portable-build pipeline are carried over as-is (dormant where unused) rather than deleted — nothing vanishes without a trace.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Double-click `dist/Kurama-Portable/Kurama.exe` after a portable build (name inherited from the fork; unrenamed). The console stays open: you type a goal, s0uRc3 streams **thinking**, then the **answer**, and **asks before every tool**.

| Command | What |
| --- | --- |
| `python main.py` | Terminal GUI loop (asks which llama-server / GGUF model to use) |
| `python main.py --dry-run` | GUI with deterministic policy (no LLM) |
| `python main.py --model path-or-id` | Skip the picker and use that GGUF or server id |
| `python main.py --once "List the workspace"` | One-shot JSON run (scripts/CI) |
| `python main.py --once --dry-run "List the workspace"` | One-shot dry-run |
| `python tests/test_scaffold.py` | Stdlib scaffold tests |

Slash commands inside the GUI: `/help` `/models` `/config` `/preset` `/dry-run` `/reward` `/clear` `/quit`. F2 models, F3 config, Esc stops a run. Permission dialog: `y` allow, `n` deny, `a` allow this tool for the rest of the run.

## Tools

Alongside Kurama's file/web/skill tools, s0uRc3 adds the CreatorCentral gate as first-class tools — permission-gated like every other action:

| Tool | What it does |
| --- | --- |
| `init_project` | Start a new project at axiom 1 (INITIATION) |
| `project_status` | Show one project's axiom/evidence/log, or list all projects |
| `check_axiom` | Read-only: which requirements of the current axiom are met vs missing |
| `advance_axiom` | The gate — refuses to move on if requirements are missing, and says which |
| `log_evidence` | Attach evidence text to a requirement of the current axiom |
| `close_project` | Close a project — honestly: `done` only if it reached REST, else `closed_early` |

See [`creatorcentral/README.md`](creatorcentral/README.md) for the standalone CLI these tools wrap.

## Portable exe

```bash
python scripts/publish_portable.py
```

Writes `dist/Kurama-Portable/Kurama.exe` (console, Python bundled) plus `brain/`, `models/`, `llama-server`, and the rest of the runtime data. Python on PATH is not required to *run* the portable folder.

On Windows the publisher also drops **Kurama.lnk** on the Desktop (working directory = the portable folder). The frozen exe refreshes that shortcut the first time you open the GUI, so copying the folder to another machine still gets a one-click launch.

## Layout

```
s0uRc3/
  creatorcentral/ the gated CREATE-axiom project tracker (new)
  brain/          system prompt, constitution, reasoning config
  core/           loop, memory, tools, skills
  tui/            terminal GUI (Textual)
  tools/_core/    read-only tools (+ creatorcentral_tools.py bridge)
  skills/         drafts → approved → archived
  db/             SQLite + chat history
  workspace/      writable area besides skills/drafts
  tests/
  main.py
```

## Loop contract

Every model turn must emit:

1. `<think>…</think>`
2. One JSON action: `use_tool | create_skill | update_plan | save_memory | request_confirmation | finish`

The TUI streams the think block live, hides the JSON action, and shows `finish.summary` as the answer. `use_tool` and `create_skill` pause for a human decision. `request_confirmation` uses the same dialog and **continues the loop** instead of exiting.

The runtime injects: goal, last result, working memory, top-k tools, top-k skills, recalled memories.

## Skill protocol

New capability → write draft with `SKILL MANIFEST` + `def run(...) -> dict` + optional `TESTS` → static + unit gate → promote on pass. Cap: 3 drafts per run.

## Honest limits

- Embeddings are hash-bags unless you plug in a real model and pgvector.
- Web search needs `TAVILY_API_KEY`.
- Skill execution loads Python in-process after an AST gate. That is convenience, not isolation.
- `code_runner` and skill import are not a security boundary against a hostile tenant.
