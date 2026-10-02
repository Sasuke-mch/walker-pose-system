"""Discover commands without importing optional numerical/model dependencies."""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def commands() -> dict[str, str]:
    """Return command ID -> historical filename (no acceptance classification)."""
    return json.loads((PACKAGE_ROOT / "commands.json").read_text(encoding="utf-8"))


def command_path(name: str) -> Path:
    """Resolve an explicit domain.name or an unambiguous historical filename."""
    entries = commands()
    if name not in entries:
        filename = name if name.endswith(".py") else name + ".py"
        matches = [key for key, value in entries.items() if value == filename]
        if len(matches) != 1:
            raise ValueError(f"Unknown command: {name}. Use --list to see available commands.")
        name = matches[0]
    domain, module = name.split(".", 1)
    return PACKAGE_ROOT / domain / (module + ".py")
