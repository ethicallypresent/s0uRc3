"""Stage 4 — THE LOOP: wire EARS + BRAIN + MOUTH into one conversation.

mic -> faster-whisper (transcribe) -> Qwen2.5 (reply) -> Kokoro (speak) -> speaker,
on repeat, until you say the stop word ("goodbye" by default).

This stage is deliberately still push-to-talk (press Enter, speak, press
Enter again) — the same turn-taking as stage 1's test, just with the
other two pieces now wired on. Stage 5 replaces push-to-talk with voice
activity detection (it knows when you've stopped talking on its own) and
streams the reply so it starts speaking before the full reply is
generated. Get this stage working reliably first; stage 5 is a latency
optimization on top of the same pipeline, not a different one.

Usage:
  python loop.py
  python loop.py --brain-size 1.5b --voice am_michael --stop-word "that's all"
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from llama_cpp import Llama

WHISPER_SAMPLE_RATE = 16000
WHISPER_CACHE_DIR = Path(__file__).resolve().parent / "models" / "whisper"
BRAIN_CACHE_DIR = Path(__file__).resolve().parent / "models" / "brain"
MOUTH_DIR = Path(__file__).resolve().parent / "models" / "mouth"
MOUTH_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
MOUTH_MODEL_FILENAME = "kokoro-v1.0.int8.onnx"
MOUTH_VOICES_FILENAME = "voices-v1.0.bin"
VOICE_CHOICE_FILE = MOUTH_DIR / "selected_voice.txt"
DEFAULT_VOICE = "af_heart"

BRAIN_SIZES = {
    "0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
}

SYSTEM_PROMPT = (
    "You are a friendly, direct voice assistant. Keep replies short and "
    "conversational — one or two sentences unless asked for more detail. "
    "You are talking out loud, so never use emoji, markdown, or bullet points."
)
MAX_TURNS_KEPT = 20  # user+assistant pairs


def _download_mouth_file(filename: str) -> Path:
    dest = MOUTH_DIR / filename
    if dest.exists():
        return dest
    MOUTH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(f"{MOUTH_RELEASE_BASE}/{filename}", dest)
    return dest


def load_ears(model_size: str) -> WhisperModel:
    print(f"Loading EARS (faster-whisper {model_size}, CPU int8)...")
    WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=str(WHISPER_CACHE_DIR))


def load_mouth() -> Kokoro:
    print("Loading MOUTH (Kokoro-82M, CPU int8)...")
    model_path = _download_mouth_file(MOUTH_MODEL_FILENAME)
    voices_path = _download_mouth_file(MOUTH_VOICES_FILENAME)
    return Kokoro(str(model_path), str(voices_path))


def load_brain(size: str, n_ctx: int, n_threads: int) -> Llama:
    repo_id, filename = BRAIN_SIZES[size]
    print(f"Loading BRAIN (Qwen2.5 {size}, CPU, {n_threads} threads)...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(BRAIN_CACHE_DIR))
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def resolve_voice(explicit: str | None) -> str:
    if explicit:
        return explicit
    if VOICE_CHOICE_FILE.exists():
        return VOICE_CHOICE_FILE.read_text().strip()
    return DEFAULT_VOICE


def record_until_enter(device: int | None) -> np.ndarray:
    import queue

    chunks: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):  # noqa: ANN001 — sounddevice callback signature
        chunks.put(indata.copy())

    print("[listening... press Enter when you're done speaking]")
    with sd.InputStream(
        samplerate=WHISPER_SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=_callback
    ):
        input()

    frames = []
    while not chunks.empty():
        frames.append(chunks.get_nowait())
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).reshape(-1)


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    segments, _info = model.transcribe(audio, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def reply(llm: Llama, messages: list[dict[str, str]], max_tokens: int) -> str:
    out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.7)
    return out["choices"][0]["message"]["content"].strip() or "I don't have anything to add to that."


def speak(kokoro: Kokoro, text: str, voice: str, speed: float) -> None:
    if not text:
        return
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    sd.play(samples, sample_rate)
    sd.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ears-model", default="tiny.en", help="faster-whisper model size (default: tiny.en)")
    parser.add_argument("--brain-size", choices=sorted(BRAIN_SIZES), default="0.5b", help="Qwen2.5 size (default: 0.5b)")
    parser.add_argument("--brain-threads", type=int, default=4, help="CPU threads for the LLM")
    parser.add_argument("--brain-ctx", type=int, default=2048, help="LLM context window in tokens")
    parser.add_argument("--brain-max-tokens", type=int, default=150, help="max tokens per LLM reply")
    parser.add_argument("--voice", default=None, help="Kokoro voice (default: your stage 2 audition pick, else af_heart)")
    parser.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--stop-word", default="goodbye", help="say this to end the conversation (default: goodbye)")
    args = parser.parse_args()

    ears = load_ears(args.ears_model)
    mouth = load_mouth()
    brain = load_brain(args.brain_size, args.brain_ctx, args.brain_threads)
    voice = resolve_voice(args.voice)
    stop_word = args.stop_word.lower()
    print(f"\nAll three pieces loaded. Voice: {voice}. Say '{args.stop_word}' to end the conversation.\n")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        while True:
            try:
                input("[press Enter to talk] ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            audio = record_until_enter(args.device)
            heard = transcribe(ears, audio)
            if not heard:
                print("[heard nothing]\n")
                continue
            print(f"you> {heard}")

            if stop_word in heard.lower():
                farewell = "Goodbye!"
                print(f"agent> {farewell}\n")
                speak(mouth, farewell, voice, args.speed)
                break

            messages.append({"role": "user", "content": heard})
            text = reply(brain, messages, args.brain_max_tokens)
            print(f"agent> {text}\n")
            messages.append({"role": "assistant", "content": text})
            speak(mouth, text, voice, args.speed)

            if len(messages) > 1 + MAX_TURNS_KEPT * 2:
                messages[:] = [messages[0]] + messages[-MAX_TURNS_KEPT * 2 :]
    finally:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
