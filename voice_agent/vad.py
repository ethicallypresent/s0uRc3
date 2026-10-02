"""Voice activity detection — knows when you've started and stopped
talking, so the loop doesn't need a push-to-talk key.

Uses `webrtcvad` (the voice activity detector from WebRTC, the same
real-time-audio codebase used in Chrome and Firefox): for each small
chunk of audio, it classifies "speech" or "not speech" using signal
energy/spectrum characteristics — a classical signal-processing
classifier, not a neural network, which is exactly why it's nearly free
on CPU (a hard requirement here, since the LLM and TTS already want most
of the CPU budget). Silero VAD is more accurate in noisy rooms but pulls
in PyTorch as a dependency, which is a lot of weight for a step that only
has to answer "talking or not" — not worth it on an 8GB CPU-only box.

Installed as `webrtcvad-wheels` (not the plain `webrtcvad` package):
the original has no Windows wheels on PyPI and needs a C compiler to
build from source; this is the same library prebuilt for Windows.

Usage (standalone test — no STT/LLM/TTS involved, just proves VAD
triggers and releases correctly):
  python vad.py
  python vad.py --aggressiveness 3 --silence-ms 500
"""
from __future__ import annotations

import argparse
import collections
import queue
import sys

import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples/frame at 16kHz


class VadRecorder:
    """Blocks until speech starts, then returns once speech has stopped
    (silence_ms of continuous non-speech) or max_seconds is hit."""

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_ms: int = 800,
        max_seconds: float = 15.0,
        preroll_ms: int = 300,
        device: int | None = None,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames_needed = max(1, silence_ms // FRAME_MS)
        self.max_frames = int(max_seconds * 1000 // FRAME_MS)
        self.preroll_frames = max(1, preroll_ms // FRAME_MS)
        self.device = device

    def listen(self) -> np.ndarray:
        frame_queue: queue.Queue = queue.Queue()

        def _callback(indata, frames, time_info, status):  # noqa: ANN001 — sounddevice callback signature
            frame_queue.put(indata.copy())

        preroll: collections.deque = collections.deque(maxlen=self.preroll_frames)
        speech_frames: list[np.ndarray] = []
        triggered = False
        silence_run = 0
        total_frames = 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self.device,
            callback=_callback,
        ):
            while True:
                frame = frame_queue.get().reshape(-1)
                if len(frame) != FRAME_SAMPLES:
                    continue  # partial frame (e.g. stream startup); webrtcvad needs exact sizes
                is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

                if not triggered:
                    preroll.append(frame)
                    if is_speech:
                        triggered = True
                        speech_frames.extend(preroll)
                        preroll.clear()
                        silence_run = 0
                else:
                    speech_frames.append(frame)
                    if is_speech:
                        silence_run = 0
                    else:
                        silence_run += 1
                        if silence_run >= self.silence_frames_needed:
                            break

                total_frames += 1
                if total_frames >= self.max_frames:
                    break

        if not speech_frames:
            return np.zeros(0, dtype=np.float32)
        audio_i16 = np.concatenate(speech_frames)
        return (audio_i16.astype(np.float32) / 32768.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3], help="0=lenient..3=strict about what counts as speech")
    parser.add_argument("--silence-ms", type=int, default=800, help="pause length that ends an utterance")
    parser.add_argument("--max-seconds", type=float, default=15.0, help="hard cap per utterance")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    args = parser.parse_args()

    recorder = VadRecorder(
        aggressiveness=args.aggressiveness,
        silence_ms=args.silence_ms,
        max_seconds=args.max_seconds,
        device=args.device,
    )
    print("Listening continuously — speak, then pause. Ctrl+C to quit.\n")
    try:
        while True:
            print("[waiting for speech...]")
            audio = recorder.listen()
            seconds = len(audio) / SAMPLE_RATE
            print(f"[captured {seconds:.2f}s of speech]\n")
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
