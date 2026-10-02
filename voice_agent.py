"""Voice wrapper around the same local llama-server s0uRc3's other agents use.

Fully local, push-to-talk, no tools, no memory, no action protocol — just:
  press Enter -> speak -> press Enter -> hear the reply.

  mic --(sounddevice)--> faster-whisper (STT) --> llama-server chat --> pyttsx3 (TTS) --> speakers

Everything runs on-device: no API keys, no network calls besides the local
llama-server this project already starts. This is deliberately simpler than
chat_agent.py (which still has memory recall/save) — a voice turn is already
slower and noisier than typing, so there's no budget for anything that isn't
the conversation itself.

Usage:
  python voice_agent.py
  python voice_agent.py --whisper-model small --input-device 2
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from core.config import load as load_config
from core.llama_server import ensure_from_config
from core.paths import AgentPaths

SYSTEM_PROMPT = "You are a friendly, direct voice assistant. Keep replies short and conversational — one or two sentences unless asked for more."
MAX_TURNS_KEPT = 20  # user+assistant pairs; trims the oldest once exceeded
SAMPLE_RATE = 16000


def _chat_once(
    base_url: str, api_key: str, model: str, messages: list[dict[str, str]], timeout: int
) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 300}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _record_until_enter(device: int | None) -> "object":
    """Record from the mic until the user presses Enter again. Returns a
    numpy float32 array at SAMPLE_RATE, mono."""
    import queue

    import numpy as np
    import sounddevice as sd

    chunks: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):  # noqa: ANN001 — sounddevice callback signature
        chunks.put(indata.copy())

    print("[recording... press Enter to stop]")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=_callback
    ):
        input()

    frames = []
    while not chunks.empty():
        frames.append(chunks.get_nowait())
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).reshape(-1)


def _transcribe(whisper_model, audio) -> str:
    import numpy as np

    if audio.size == 0:
        return ""
    segments, _info = whisper_model.transcribe(audio, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def _speak(tts_engine, text: str) -> None:
    if not text:
        return
    tts_engine.say(text)
    tts_engine.runAndWait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="faster-whisper model size/name (default: base.en)",
    )
    parser.add_argument(
        "--input-device", type=int, default=None, help="sounddevice input device index"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="skip TTS playback and just print replies (useful without speakers)",
    )
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Missing dependency: pip install faster-whisper", file=sys.stderr)
        return 1
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        print("Missing dependency: pip install sounddevice", file=sys.stderr)
        return 1

    tts_engine = None
    if not args.text_only:
        try:
            import pyttsx3

            tts_engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 — TTS is optional, never fatal
            print(f"[no TTS available ({exc}); continuing in text-only mode]")

    paths = AgentPaths.discover()
    cfg = load_config(paths)
    boot = ensure_from_config(paths.root, cfg)
    if not boot.get("ok"):
        print(f"Could not start/reach the model server: {boot.get('error')}", file=sys.stderr)
        return 1

    llm = cfg.get("llm") or {}
    base_url = str(llm.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")
    api_key = str(llm.get("api_key", "sk-no-key-needed"))
    model = str(llm.get("model", "pocket"))
    timeout = int(llm.get("timeout_sec", 300))

    print(f"Loading whisper model '{args.whisper_model}' (CPU)...")
    whisper_model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Voice agent ready. Press Enter to start talking, Ctrl+C to quit.\n")

    try:
        while True:
            try:
                input("[press Enter to talk] ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            audio = _record_until_enter(args.input_device)
            user_input = _transcribe(whisper_model, audio).strip()
            if not user_input:
                print("[heard nothing]\n")
                continue
            print(f"you> {user_input}")

            call_messages = messages + [{"role": "user", "content": user_input}]
            try:
                reply = _chat_once(base_url, api_key, model, call_messages, timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"[error: {exc}]\n")
                continue

            reply = reply.strip() or "(no reply)"
            print(f"agent> {reply}\n")
            if tts_engine is not None:
                _speak(tts_engine, reply)

            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": reply})
            if len(messages) > 1 + MAX_TURNS_KEPT * 2:
                messages[:] = [messages[0]] + messages[-MAX_TURNS_KEPT * 2 :]
    finally:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
