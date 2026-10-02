# s0uRc3

A minimal local chat agent. It talks to a local llama-server (llama.cpp),
started automatically from `brain/reasoning_config.json`, and recalls/saves
long-term memory in `db/memory.sqlite` — nothing else. No tools, no TUI, no
JSON action protocol, no skill system. Plain conversation, on purpose: the
smaller and more direct the loop, the less there is for a small local model
to trip over.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Drop your GGUF model at `models/pocket.gguf` (or point `llama_server.model_path`
in `brain/reasoning_config.json` elsewhere first). `main.py` starts
`llama-server` for you if it isn't already running, then opens a plain
`you>` / `agent>` prompt. Say "remember that ..." to save a fact to memory;
`exit` or Ctrl+C to quit.

## Layout

```
s0uRc3/
  brain/   reasoning_config.json (llm + llama-server settings)
  core/    config, paths, memory, llama-server lifecycle
  db/      SQLite long-term memory
  models/  GGUF model(s) + Modelfile
  main.py  the agent
```

## Memory

Every turn, the top few relevant memories (`fact`/`preference` kind) are
pulled into context — a cheap embedding/keyword lookup, not an extra LLM
call. Nothing is saved automatically; only when you ask for it ("remember
that ...", "note that ...", "don't forget ...").

## Honest limits

- Embeddings are hash-bags unless you plug in a real embedding model.
- This is a conversational loop only — it cannot read/write files, browse
  the web, or run code on your machine.
