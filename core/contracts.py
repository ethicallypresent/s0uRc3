"""Shared contracts. Named and frozen before any component talks.

Change these only with a config_version bump. Do not mutate at runtime.
"""

from __future__ import annotations

from types import MappingProxyType

CONFIG_VERSION = "1.1"

ACTIONS = frozenset(
    {
        "use_tool",
        "create_skill",
        "update_plan",
        "save_memory",
        "propose_long_term_memory",
        "request_confirmation",
        "refine_code",
        "finish",
    }
)

STEER_VALUES = frozenset({"continue", "replan", "stop_spin", "rest"})
MODEL_KINDS = frozenset({"server", "gguf"})
MEMORY_KINDS = frozenset({"fact", "lesson", "preference", "failure", "constraint"})
BACKENDS = frozenset({"llamacpp", "llama.cpp", "llama-server", "openai", "ollama", "openai-compatible"})

MAX_GOAL_CHARS = 8000
MAX_TOOL_ARG_CHARS = 200_000
MAX_MODEL_SPEC_CHARS = 1024
MAX_TOOL_NAME_CHARS = 64
NUM_CTX_MIN, NUM_CTX_MAX = 512, 262_144
PORT_MIN, PORT_MAX = 1, 65535

KNOWN_LLM_DEFAULTS = MappingProxyType(
    {
        "backend": "llamacpp",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "pocket",
        "api_key": "sk-no-key-needed",
        "temperature": 0.2,
        "max_tokens": 800,
        # Thinking-budget tiers (core/router.py): chosen by classify_turn()
        # before the request is sent, not one flat cap for everything.
        # "max_tokens" above stays as deliberate's fallback for configs that
        # only ever set the old single value.
        "act_max_tokens": 120,
        "glance_max_tokens": 250,
        "deliberate_max_tokens": 800,
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
    "ACTIONS",
    "BACKENDS",
    "CONFIG_VERSION",
    "KNOWN_LLM_DEFAULTS",
    "MAX_GOAL_CHARS",
    "MAX_MODEL_SPEC_CHARS",
    "MAX_TOOL_ARG_CHARS",
    "MAX_TOOL_NAME_CHARS",
    "MEMORY_KINDS",
    "MODEL_KINDS",
    "NUM_CTX_MAX",
    "NUM_CTX_MIN",
    "PORT_MAX",
    "PORT_MIN",
    "STEER_VALUES",
)
