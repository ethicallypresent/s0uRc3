"""Stage 1 — EARS: speech-to-text, standalone test.

Records 5 seconds from your default microphone, runs it through a local
faster-whisper model (CPU, int8), and prints the transcription. Nothing
else in this project depends on this file running correctly first, but you
should not move on to stage 2 until this reliably prints back what you say.

First run downloads the model (~75MB for tiny.en) to
voice_agent/models/whisper/ and caches it there.

Usage:
  python stage1_ears_test.py
  python stage1_ears_test.py --model base.en --seconds 7 --device 2
  python stage1_ears_test.py --list-devices
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # whisper models expect 16kHz mono
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models" / "whisper"


def record(seconds: float, device: int | None) -> np.ndarray:
    print(f"Recording {seconds:.0f}s from device {device if device is not None else '(default)'}... speak now.")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device
    )
    sd.wait()
    print("Done recording.")
    return audio.reshape(-1)


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    segments, info = model.transcribe(audio, language="en")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"(detected language: {info.language}, confidence {info.language_probability:.2f})")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny.en", help="faster-whisper model size (default: tiny.en)")
    parser.add_argument("--seconds", type=float, default=5.0, help="recording length in seconds")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    print(f"Loading faster-whisper model '{args.model}' (CPU, int8)...")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        args.model, device="cpu", compute_type="int8", download_root=str(MODEL_CACHE_DIR)
    )
    print("Model ready.\n")

    try:
        audio = record(args.seconds, args.device)
    except sd.PortAudioError as exc:
        print(f"Could not record audio: {exc}", file=sys.stderr)
        print("Run with --list-devices to see available input devices.", file=sys.stderr)
        return 1

    text = transcribe(model, audio)
    print(f"\nTranscription: {text!r}")
    if not text:
        print("(empty — try speaking louder/closer to the mic, or a longer --seconds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
