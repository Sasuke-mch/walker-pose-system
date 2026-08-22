from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_host_path(raw: str | Path, project_root: Path, pipeline_root: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if p.is_absolute():
        return p.resolve()
    # Project-relative paths are the default. Paths starting with pipeline: are
    # resolved relative to this program.
    text = str(raw)
    if text.startswith("pipeline:"):
        return (pipeline_root / text[len("pipeline:"):]).resolve()
    return (project_root / p).resolve()


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def parse_key_value(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid KEY=VALUE: {item}")
        result[key] = value
    return result
