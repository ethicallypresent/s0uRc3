"""Stage 3 — BRAIN: local LLM reply, standalone test.

You type text, it replies — a plain terminal chat loop against a local
Qwen2.5-Instruct model running as a quantized GGUF through
llama-cpp-python, 100% on CPU. No network calls after the model is
downloaded once.

Why Qwen2.5 0.5B/1.5B: small enough to load and respond in well under a
second per turn on an ordinary CPU, instruction-tuned (follows "keep it
short" style system prompts reasonably well), and ships official GGUF
quantizations from Qwen themselves.

First run downloads the model (~470MB for 0.5B at q4_k_m, ~1.1GB for
1.5B) to voice_agent/models/brain/ and caches it there.

Usage:
  python stage3_brain_test.py
  python stage3_brain_test.py --size 1.5b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models" / "brain"

SYSTEM_PROMPT = (
    "You are a friendly, direct voice assistant. Keep replies short and "
    "conversational — one or two sentences unless asked for more detail."
)

_SIZES = {
    "0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
}


def load_model(size: str, n_ctx: int, n_threads: int) -> Llama:
    repo_id, filename = _SIZES[size]
    print(f"Fetching {filename} (cached after first run)...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODEL_CACHE_DIR))
    print(f"Loading {filename} (CPU, {n_threads} threads, ctx={n_ctx})...")
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(_SIZES), default="0.5b", help="model size (default: 0.5b)")
    parser.add_argument("--ctx", type=int, default=2048, help="context window in tokens")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads to use")
    parser.add_argument("--max-tokens", type=int, default=200, help="max tokens per reply")
    args = parser.parse_args()

    llm = load_model(args.size, args.ctx, args.threads)
    print("Model ready. Type a message, 'exit' to quit.\n")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            messages.append({"role": "user", "content": user_input})
            out = llm.create_chat_completion(messages=messages, max_tokens=args.max_tokens, temperature=0.7)
            reply = out["choices"][0]["message"]["content"].strip() or "(no reply)"
            print(f"agent> {reply}\n")
            messages.append({"role": "assistant", "content": reply})
    finally:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
