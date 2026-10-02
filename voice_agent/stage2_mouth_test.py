"""Stage 2 — MOUTH: text-to-speech, standalone test.

You type a sentence, it speaks it through your speakers — local
Kokoro-82M (an 82-million-parameter TTS model, small by TTS standards but
trained to sound natural rather than robotic), running as a CPU-friendly
ONNX model via the `kokoro-onnx` package. No cloud calls, no API keys.

First run downloads two files from the kokoro-onnx project's GitHub
releases into voice_agent/models/mouth/:
  - kokoro-v1.0.onnx (~300MB, default) or kokoro-v1.0.int8.onnx (~110MB,
    `--precision int8`) — see the precision note below for why fp32 is
    the default despite being 3x larger.
  - voices-v1.0.bin (~27MB) — a bundle of all 54 voice "style" embeddings.
    A voice here isn't a separate model, just a different style vector fed
    into the same network.

**Note on precision — fp32 is the default, not int8:** int8 quantized
ONNX ops are only fast on CPUs with VNNI (AVX512-VNNI or AVX-VNNI,
Intel Rocket Lake+/Alder Lake+, AMD Zen4+). Without it, onnxruntime falls
back to an unoptimized int8 path that's *slower* than fp32 — measured 9x
slower (~23s vs ~2.5s per sentence) on a Ryzen 5 7520U (Zen2, no VNNI).
If your CPU does have VNNI, `--precision int8` is worth trying for the
smaller download/memory footprint — time a few sentences with both and
keep whichever is actually faster on your hardware.

Usage:
  python stage2_mouth_test.py                  # type sentences, hear af_heart (default)
  python stage2_mouth_test.py --audition        # cycle a handful of voices, pick one
  python stage2_mouth_test.py --voice am_michael
  python stage2_mouth_test.py --list-voices
  python stage2_mouth_test.py --precision int8  # try the quantized model instead
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import sounddevice as sd
from kokoro_onnx import Kokoro

MODEL_DIR = Path(__file__).resolve().parent / "models" / "mouth"
RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
MODEL_FILENAMES = {"fp32": "kokoro-v1.0.onnx", "int8": "kokoro-v1.0.int8.onnx"}
VOICES_FILENAME = "voices-v1.0.bin"
VOICE_CHOICE_FILE = MODEL_DIR / "selected_voice.txt"

DEFAULT_VOICE = "af_heart"
AUDITION_VOICES = ["af_heart", "am_michael", "bf_emma"]


def _download(filename: str) -> Path:
    dest = MODEL_DIR / filename
    if dest.exists():
        return dest
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE_BASE}/{filename}"
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_kokoro(precision: str = "fp32") -> Kokoro:
    model_path = _download(MODEL_FILENAMES[precision])
    voices_path = _download(VOICES_FILENAME)
    return Kokoro(str(model_path), str(voices_path))


def speak(kokoro: Kokoro, text: str, voice: str, speed: float = 1.0) -> None:
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    sd.play(samples, sample_rate)
    sd.wait()


def list_voices(kokoro: Kokoro) -> None:
    print(", ".join(sorted(kokoro.voices.files)))


def audition(kokoro: Kokoro, voices: list[str]) -> str | None:
    sample_text = "Hi, I'm your voice assistant. This is what I sound like."
    print("Auditioning voices — listening to each in turn.\n")
    for voice in voices:
        print(f"-- {voice} --")
        speak(kokoro, sample_text, voice)
    while True:
        choice = input(f"\nPick a voice ({', '.join(voices)}), or type any other voice name: ").strip()
        if not choice:
            continue
        if choice in kokoro.voices.files:
            VOICE_CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
            VOICE_CHOICE_FILE.write_text(choice)
            print(f"Saved '{choice}' as the default voice (voice_agent/models/mouth/selected_voice.txt).")
            return choice
        print(f"Unknown voice {choice!r}. Run with --list-voices to see all options.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=None, help=f"voice name (default: {DEFAULT_VOICE}, or last audition pick)")
    parser.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier")
    parser.add_argument("--audition", action="store_true", help="play a few candidate voices and pick one")
    parser.add_argument("--list-voices", action="store_true", help="print all 54 available voices and exit")
    parser.add_argument(
        "--precision",
        choices=sorted(MODEL_FILENAMES),
        default="fp32",
        help="fp32 (default, fastest without VNNI) or int8 (smaller, faster only with CPU VNNI support)",
    )
    args = parser.parse_args()

    print(f"Loading Kokoro-82M (CPU, {args.precision})...")
    kokoro = load_kokoro(args.precision)
    print("Model ready.\n")

    if args.list_voices:
        list_voices(kokoro)
        return 0

    if args.audition:
        audition(kokoro, AUDITION_VOICES)
        return 0

    voice = args.voice
    if voice is None:
        voice = VOICE_CHOICE_FILE.read_text().strip() if VOICE_CHOICE_FILE.exists() else DEFAULT_VOICE
    print(f"Using voice: {voice}\nType a sentence to hear it spoken, 'exit' to quit.\n")

    try:
        while True:
            try:
                text = input("text> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break
            speak(kokoro, text, voice, args.speed)
    finally:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
