# Training data

Two batches, kept side by side rather than merged — each is a real record of
a specific session, not a synthetic demo set (IMMUTABILITY: nothing gets
overwritten to make room for the next one).

## Kurama batch (original debugging/coding session — loop fixes, llama.cpp, safety, skills)

| File | Purpose |
|------|---------|
| `trajectories.jsonl` | Goal → assistant think+JSON actions (coding-focused) |
| `kurama_sft.jsonl` | Chat SFT rows (system + state packet + assistant) for fine-tuning `pocket.gguf` |
| `session_lessons.json` | Lessons already seeded into `db/memory.sqlite` |

Regenerate/reapply: `python scripts/train_from_session.py`

## s0uRc3 batch (forking Kurama into the Builder Agent, building creatorcentral/)

| File | Purpose |
|------|---------|
| `s0urc3_coding_trajectories.jsonl` | 8 goal → think+action examples distilled from *this session's real work*: reading an existing tool's signature convention before writing a new one, diagnosing and fixing the check_axiom/advance_axiom ok-vs-gate_ok bug, excluding creatorcentral/ from the PyInstaller bundle because of its `__file__`-relative data lookup, and an honest closed_early close-out |
| `s0urc3_coding_sft.jsonl` | Same 8 examples as ShareGPT-style `messages` rows, system turn built from the live `brain/system_prompt.md` + `constitution.md` |
| `s0urc3_session_lessons.json` | 9 lessons — the ok/gate_ok distinction, mirroring tool conventions, the PyInstaller `__file__` constraint, writing the regression test right after the fix, a goal-wording keyword collision that silently redirected the loop, and more |

Regenerate/reapply: `python scripts/train_from_s0urc3_coding_session.py`

## Already applied (agent learning layer — this is what "training" means without a fine-tune binary)

Both scripts do the same two things:

```bash
python scripts/train_from_session.py                  # Kurama batch
python scripts/train_from_s0urc3_coding_session.py     # s0uRc3 batch
```

1. Seed each batch's lessons into long-term memory (`db/memory.sqlite`, `verified: true`).
2. Boost Bayesian action/tool beliefs in `db/beliefs.json` (e.g. `tool:check_axiom`,
   `tool:advance_axiom`) and reward a synthetic trace matching the batch's real arc,
   so the loop's tool ranking and steering are shaped by what actually happened,
   not just by what's in the SFT file.

This is real and live right now — it changes which tools the loop favors and what
lessons get recalled into context. It does **not** touch `pocket.gguf`'s weights.

## Fine-tune the GGUF later (needs infrastructure this environment doesn't have)

Checked and confirmed missing here: no `llama-finetune` binary (not in `tools/bin/`,
not in the WinGet llama.cpp package), no `torch`/`transformers`/`peft`/`unsloth`
installed, no GPU (`nvidia-smi` not found), and no unquantized base weights for
`pocket` — only the already-quantized `.gguf` exists, which standard LoRA tooling
can't fine-tune directly.

To actually fine-tune later, you'd need:

1. The original unquantized instruct model `pocket.gguf` was quantized from (HF
   `safetensors` format), or accept training a difference base model instead.
2. A training stack — Unsloth (needs a CUDA GPU) or plain `transformers` + `peft`
   (CPU works but is very slow for a 4B-class model).
3. `s0urc3_coding_sft.jsonl` (or the Kurama batch, or both concatenated) as the
   dataset — already ShareGPT-style `messages`, ready to load as-is.
4. LoRA on top of that base model, then re-quantize/export back to GGUF and
   replace `models/pocket.gguf`.

Until then, behavioral training via memory + beliefs (above) is the live mechanism.
