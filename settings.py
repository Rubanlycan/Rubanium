"""Small helpers for environment-backed project configuration."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_env(env_file: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell environment values."""
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def configured_path(name: str, default: Path, base_dir: Path) -> Path:
    """Resolve an environment path, using the project directory for relative values."""
    configured = os.environ.get(name)
    path = Path(configured).expanduser() if configured else default
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
