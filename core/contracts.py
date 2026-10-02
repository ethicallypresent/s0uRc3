"""Shared contracts for the LLM/config layer. Change only with a
config_version bump; do not mutate at runtime."""

from __future__ import annotations

from types import MappingProxyType

CONFIG_VERSION = "2.0"

BACKENDS = frozenset({"llamacpp", "llama.cpp", "llama-server", "openai", "ollama", "openai-compatible"})

NUM_CTX_MIN, NUM_CTX_MAX = 512, 262_144
PORT_MIN, PORT_MAX = 1, 65535

KNOWN_LLM_DEFAULTS = MappingProxyType(
    {
        "backend": "llamacpp",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "pocket",
        "api_key": "sk-no-key-needed",
        "temperature": 0.7,
        "max_tokens": 500,
        "timeout_sec": 300,
        "dry_run_on_no_server": True,
        "warmup": True,
        "keep_alive": "10m",
        "embedding_model": "",
        "embedding_url": "",
        "num_ctx": 4096,
        "num_batch": 512,
        "num_gpu": 0,
        "num_thread": 0,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "mirostat": 0,
        "load_preset": "cpu_optimal",
    }
)

__all__ = (
    "BACKENDS",
    "CONFIG_VERSION",
    "KNOWN_LLM_DEFAULTS",
    "NUM_CTX_MAX",
    "NUM_CTX_MIN",
    "PORT_MAX",
    "PORT_MIN",
)
