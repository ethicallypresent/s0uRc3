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

## Stage 2 — MOUTH

```bash
pip install -r requirements.txt
python stage2_mouth_test.py --audition
```

Kokoro-82M is a small (82M parameter) text-to-speech model trained to
sound natural rather than robotic — small by TTS standards, which is
exactly why it's CPU-friendly. We run it through `kokoro-onnx`, which
loads it as an ONNX model (a portable, runtime-optimized model format)
via `onnxruntime` on CPU. A "voice" isn't a separate model — it's a
different style vector fed into the same network, which is why switching
voices is instant and free (no extra model to load).

**Default precision is fp32, not int8 — this was flipped after hardware
testing.** The original plan was int8-quantized (~110MB vs. ~300MB fp32),
same idea as stage 3's quantized LLM: smaller, and normally faster on
CPU. But int8 ONNX ops are only fast with CPU VNNI support (AVX512-VNNI
or AVX-VNNI — Intel Rocket Lake+/Alder Lake+, AMD Zen4+). Without it,
onnxruntime's int8 path is *slower* than fp32, not faster — measured
**~23s vs ~2.5s per sentence (9x)** on a Ryzen 5 7520U (Zen2, no VNNI),
which is exactly the "says a sentence, pauses ~30s" symptom. fp32 is the
safe default across unknown target hardware; pass `--precision int8`
(stage 2) / `--mouth-precision int8` (`loop.py`) if you've confirmed your
CPU has VNNI and want the smaller footprint.

`--audition` plays three candidate voices (`af_heart`, `am_michael`,
`bf_emma`) and lets you pick one, saving your choice to
`voice_agent/models/mouth/selected_voice.txt` so stages 4/5 pick it up
automatically. Run `--list-voices` to see all 54 available voices (en/es/fr/
hi/it/ja/pt/zh accents) if none of the three defaults suit you.

First run downloads `kokoro-v1.0.onnx` (~300MB fp32, or `kokoro-v1.0.int8.onnx`
~110MB with `--precision int8`) and `voices-v1.0.bin` (~27MB, all 54 voices
bundled together) from the `kokoro-onnx` project's GitHub releases into
`voice_agent/models/mouth/`.

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

**Troubleshooting: crashes with `OSError: [WinError -1073741795]
0xc000001d` (STATUS_ILLEGAL_INSTRUCTION) on model load.** The
maintainer's prebuilt Windows wheel is compiled with AVX512 instructions.
Most CPUs support AVX2 but not AVX512 — in particular every mobile/laptop
AMD Ryzen chip up through Zen3 (this includes the 7020/7030/7040-series
"U" chips, which are Zen2/Zen3 despite the Ryzen-7000-era model number)
and most pre-2021 Intel CPUs. If `python -c "import llama_cpp;
print(llama_cpp.llama_cpp.llama_print_system_info().decode())"` prints
`AVX512 = 1`, that's the cause — the binary has no runtime fallback, it
just executes an instruction your CPU doesn't have.

Fix: rebuild `llama-cpp-python` from source with AVX512 off and
AVX2/FMA on, using a local C++ compiler (Visual Studio Build Tools) and
CMake instead of the prebuilt wheel. Run `rebuild_llama_cpp_no_avx512.bat`
(needs CMake and the MSVC Build Tools installed — `winget install
Kitware.CMake` and `winget install Microsoft.VisualStudio.2022.BuildTools`
if you don't have them) after the normal `pip install -r
requirements.txt`, or by hand:

```bat
call "<path to>\VC\Auxiliary\Build\vcvarsall.bat" x64
set CMAKE_ARGS=-DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX=ON -DGGML_F16C=ON
.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.30 --no-binary llama-cpp-python --force-reinstall --no-cache-dir
```

Re-run the system-info check above afterward to confirm `AVX512` is gone
from the line and `AVX2`/`FMA` are still `1`.

## Stage 4 — THE LOOP

```bash
pip install -r requirements.txt
python loop.py
```

Wires all three pieces into one conversation: press Enter, speak, press
Enter again, EARS transcribes it, BRAIN replies, MOUTH speaks the reply —
repeat until you say the stop word ("goodbye" by default, change with
`--stop-word`). Still push-to-talk on purpose: get the pipeline itself
solid before stage 5 layers automatic turn-taking and streaming on top of
it — same three pieces, just smarter about when to listen and faster to
start talking.

Uses your stage 2 voice pick automatically (reads
`models/mouth/selected_voice.txt`); override with `--voice`.
`--brain-size 1.5b` swaps in the larger model if you upgraded in stage 3.

Verified in this sandbox with the mic/speaker calls mocked out and the
transcription stepped through canned input: the LLM reply, the stop-word
exit, and every call into EARS/BRAIN/MOUTH executed correctly end to end.
The one thing that needs your actual hardware is hearing it.

## Stage 5 — ALIVE

```bash
pip install -r requirements.txt
python loop.py
```

Two changes on top of stage 4's exact same pipeline — nothing about
EARS/BRAIN/MOUTH themselves changes, only when they're triggered:

**Voice activity detection** (`vad.py`) replaces the Enter key. It uses
`webrtcvad` — WebRTC's voice activity detector (the same real-time-audio
project behind Chrome/Firefox calls), a lightweight signal-based speech
classifier, not a neural network, so it costs almost nothing on CPU.
`loop.py` now listens continuously: it waits for you to start speaking,
keeps recording while you talk, and stops automatically ~800ms after you
stop (configurable with `--vad-silence-ms`). Test it on its own first:

```bash
python vad.py
```

This just prints "captured N.NNs of speech" each time you talk and
pause — confirm it reliably starts/stops on your actual voice and room
noise before trusting it inside the full loop. `--aggressiveness 3` is
stricter about what counts as speech (use it in a noisy room);
`--silence-ms 500` ends your turn faster if 800ms feels laggy.
`--push-to-talk` on `loop.py` falls back to stage 4's Enter-key behavior
if VAD isn't working well for you.

**Streamed TTS** (`stream_reply_and_speak` in `loop.py`) is the actual
latency fix. Previously the agent waited for the *entire* reply to
generate, then synthesized and played all of it. Now it watches the
LLM's token stream for a complete sentence (ends in `.`/`!`/`?`), hands
that sentence to a background thread to synthesize and play immediately,
and keeps generating the next sentence while the current one is already
playing. You hear the first sentence as soon as it's ready instead of
waiting for the whole answer — this is most of what makes it feel like a
conversation instead of a request/response tool.

Once this works reliably, that's the whole agent: `loop.py` is the one
thing you run day to day.
