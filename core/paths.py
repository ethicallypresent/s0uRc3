"""Filesystem layout for the agent: where its config, code, and database live."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.paths")

# Subdirectories that must exist for a directory to be recognized as an
# agent root during auto-discovery (source tree).
REQUIRED_SUBDIRS = ("brain", "core", "db")


@dataclass(frozen=True)
class AgentPaths:
    root: Path
    # Optional override: if set, load_config reads this file instead of
    # brain/reasoning_config.json.
    config_path: Path | None = None

    # --- derived paths (properties so they're always in sync with root) ---

    @property
    def brain(self) -> Path:
        return self.root / "brain"

    @property
    def core(self) -> Path:
        return self.root / "core"

    @property
    def db(self) -> Path:
        return self.root / "db"

    @property
    def effective_config_path(self) -> Path:
        """The config file that should be read — override if set, else default."""
        return self.config_path or (self.brain / "reasoning_config.json")

    @classmethod
    def discover(cls, start: Path | None = None, config_path: Path | None = None) -> "AgentPaths":
        """Find the agent root and return an AgentPaths for it.

        Resolution order:
        1. AGENT_ROOT environment variable (explicit override, useful in CI/containers)
        2. Walk upward from `start` (defaults to this file's location) until a
           directory is found that contains all REQUIRED_SUBDIRS.
        3. Raise FileNotFoundError if nothing matches.
        """
        env_root = os.environ.get("AGENT_ROOT")
        if env_root:
            root = Path(env_root).resolve()
            log.debug("Using AGENT_ROOT=%s", root)
            return cls(root=root, config_path=config_path)
        root = cls._find_root(start or Path(__file__).resolve())
        return cls(root=root, config_path=config_path)

    @staticmethod
    def _find_root(start: Path) -> Path:
        for candidate in [start, *start.parents]:
            if candidate.is_dir() and all((candidate / s).is_dir() for s in REQUIRED_SUBDIRS):
                return candidate
        raise FileNotFoundError(
            f"Could not locate agent root above {start} "
            f"(looked for {', '.join(REQUIRED_SUBDIRS)}/). "
            "Set AGENT_ROOT to specify it explicitly."
        )
