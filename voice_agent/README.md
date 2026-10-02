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
