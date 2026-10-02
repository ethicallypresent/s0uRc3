# Voice agent (Windows, CPU-only, 8GB RAM)

Built in stages, each independently testable before being wired together.

| Stage | Piece | What it does | Test script |
| --- | --- | --- | --- |
| 1 | EARS | speech-to-text (faster-whisper) | `stage1_ears_test.py` |
| 2 | MOUTH | text-to-speech (Kokoro-82M) | `stage2_mouth_test.py` |
| 3 | BRAIN | local LLM reply (Qwen2.5 GGUF via llama-cpp-python) | `stage3_brain_test.py` |
| 4 | LOOP | mic -> transcribe -> LLM -> speak, continuous | `loop.py` |
| 5 | ALIVE | voice activity detection + streamed TTS for low latency | (built into `loop.py`) |

All pieces are CPU-only — no CUDA, no GPU-only packages. Everything here
is self-contained under `voice_agent/`; it does not depend on the rest of
this repo.

## Stage 1 — EARS

```bash
pip install -r requirements.txt
python stage1_ears_test.py
```

`faster-whisper` is a CPU/GPU-optimized reimplementation of OpenAI's
Whisper speech-recognition model (built on the CTranslate2 inference
engine — int8 quantization + optimized kernels, which is what makes it
fast enough on an ordinary CPU). You give it a raw audio waveform; it
gives you the words. `stage1_ears_test.py` records 5 seconds from your
default mic and prints what it heard, using the `tiny.en` model
(~75MB, downloads once to `voice_agent/models/whisper/` on first run).

Flags:
- `--list-devices` — see available microphones, if the default one is wrong
- `--device N` — record from a specific device index
- `--model base.en` — slightly larger/slower/more accurate than `tiny.en`
- `--seconds N` — change the recording length

Don't move on to stage 2 until this reliably transcribes what you say.

## Stage 3 — BRAIN

```bash
pip install -r requirements.txt
python stage3_brain_test.py
```

`llama-cpp-python` is a Python binding for `llama.cpp`, the C++ engine
that runs quantized ("GGUF") LLMs efficiently on ordinary CPUs — no GPU,
no CUDA. "Quantized" means the model's weights are compressed from
16-bit floats down to ~4 bits each (here, `q4_k_m`), trading a small
amount of accuracy for a ~4x smaller file and much faster CPU inference.

We're using Qwen2.5-Instruct: small enough (0.5B or 1.5B parameters) to
reply in well under a second per turn on CPU, and instruction-tuned so it
actually follows a "keep replies short" system prompt instead of
rambling. `stage3_brain_test.py` is a plain type-and-reply chat loop
against it — no memory, no voice yet, just proving the model itself
works and responds sensibly.

First run downloads the GGUF (~470MB for 0.5B, ~1.1GB for 1.5B) to
`voice_agent/models/brain/`.

Flags:
- `--size 1.5b` — use the larger model if 0.5B's replies feel too thin
  (still comfortably fits in 8GB RAM; slower per-token but more coherent)
- `--threads N` — CPU threads for inference (default 4; try your core
  count if replies feel slow)
- `--ctx N` — context window in tokens (default 2048, plenty for a
  voice conversation with no long documents involved)

**Note on `llama-cpp-python` install:** PyPI only hosts the source
package, which needs a C++ compiler to build. `requirements.txt` points
pip at the maintainer's own prebuilt-wheel index instead
(`--extra-index-url` line at the top), so `pip install -r
requirements.txt` gets a ready-to-use Windows wheel with no compiler
needed.
