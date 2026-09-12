# s0uRc3

A local Builder Agent, forked from [Kurama](https://github.com/ethicallypresent/kurama-local-agent). Same self-contained engine — it owns its prompt, memory, tools, and skill lifecycle, with a **looping terminal GUI** (`python main.py`) — repurposed around one rule: every real build moves through the ten CREATE axioms, gated, with evidence, tracked by [`creatorcentral/`](creatorcentral/README.md). s0uRc3's own construction is documented the same way: [`creatorcentral/SELF_CONSTRUCTION.md`](creatorcentral/SELF_CONSTRUCTION.md).

Kurama's TUI, skill lifecycle, and Windows portable-build pipeline are carried over as-is (dormant where unused) rather than deleted — nothing vanishes without a trace.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Double-click `dist/Kurama-Portable/Kurama.exe` after a portable build (name inherited from the fork; unrenamed). The console stays open: you type a goal, s0uRc3 streams **thinking**, then the **answer**, and **asks before every tool**.

### Two TUIs

`python main.py` opens the **simplified TUI** (`tui/simple_app.py`) by default — no in-app model picker or config screens; pick the model with `--model` or the `brain/reasoning_config.json` default, switch later with `/model`. It exists because the original TUI redraws on *every token* (a full widget rebuild + a full-buffer rescan + a DOM query + a scroll, per token, with no batching — visibly stutters on a fast local model). The simplified one buffers tokens off the UI thread and redraws at a fixed ~20Hz tick no matter how fast the model streams, so the terminal repaints smoothly instead of flooding.

The original full TUI (`tui/app.py` — in-app model picker, `/models`/`/config` screens) is still there, dormant by default: `python main.py --classic-tui`.

| Command | What |
| --- | --- |
| `python main.py` | Simplified TUI (asks nothing; uses `--model` or the config default) |
| `python main.py --classic-tui` | Original TUI (asks which llama-server / GGUF model to use) |
| `python main.py --dry-run` | GUI with deterministic policy (no LLM) |
| `python main.py --model path-or-id` | Use that GGUF path or server id instead of the config default |
| `python main.py --once "List the workspace"` | One-shot JSON run (scripts/CI) |
| `python main.py --once --dry-run "List the workspace"` | One-shot dry-run |
| `python tests/test_scaffold.py` | Stdlib scaffold tests |

Simplified-TUI slash commands: `/help` `/model [spec]` `/dry-run` `/reward` `/clear` `/quit`. Classic TUI adds `/models` `/config` `/preset` `/load` `/stop-server`, plus F2/F3. Esc stops a run in either. Permission dialog: `y` allow, `n` deny, `a` allow this tool for the rest of the run.

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
