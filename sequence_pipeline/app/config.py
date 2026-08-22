from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import deep_merge, read_json, resolve_host_path


@dataclass
class LoadedConfig:
    pipeline_root: Path
    project_root: Path
    data: dict[str, Any]
    default_path: Path
    local_path: Path

    @property
    def defaults(self) -> dict[str, Any]:
        return self.data.get("defaults", {})

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return self.data.get("models", {})

    @property
    def model_order(self) -> list[str]:
        return list(self.data.get("model_order", []))

    def model(self, key: str) -> dict[str, Any]:
        if key not in self.models:
            raise KeyError(f"Unknown model: {key}")
        return self.models[key]

    def resolve(self, raw: str | Path) -> Path:
        return resolve_host_path(raw, self.project_root, self.pipeline_root)


def load_config(pipeline_root: Path, local_path: Path | None = None) -> LoadedConfig:
    default_path = pipeline_root / "configs" / "default.json"
    if local_path is None:
        local_path = pipeline_root / "configs" / "local.json"
    elif not local_path.is_absolute():
        local_path = (Path.cwd() / local_path).resolve()

    base = read_json(default_path)
    local = read_json(local_path) if local_path.exists() else {"project_root": "..", "overrides": {}}
    merged = deep_merge(base, local.get("overrides", {}))

    raw_root = local.get("project_root", "..")
    root_path = Path(raw_root)
    if root_path.is_absolute():
        project_root = root_path.resolve()
    else:
        project_root = (pipeline_root / root_path).resolve()

    return LoadedConfig(
        pipeline_root=pipeline_root,
        project_root=project_root,
        data=merged,
        default_path=default_path,
        local_path=local_path,
    )
