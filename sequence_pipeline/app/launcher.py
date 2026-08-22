from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LoadedConfig
from .docker import Mount, build_command, run_streaming
from .utils import resolve_host_path


@dataclass
class StageResult:
    key: str
    status: str
    message: str
    command: list[str]
    expected_outputs: list[Path]


class ModelLauncher:
    def __init__(self, config: LoadedConfig, model_key: str, context: dict[str, Any], weight_override: str | None = None):
        self.config = config
        self.key = model_key
        self.spec = config.model(model_key)
        self.context = context
        self.weight_override = weight_override

    def _weight(self) -> tuple[Path | None, str]:
        spec = self.spec.get("weight", {})
        raw = self.weight_override if self.weight_override is not None else spec.get("host_path", "")
        if not raw:
            return None, ""
        host = self.config.resolve(raw)
        fixed = spec.get("container_file_name")
        filename = fixed or host.name
        return host, f"{spec['container_dir'].rstrip('/')}/{filename}"

    def _values(self, weight_container: str, weight_exists: bool) -> dict[str, Any]:
        values = dict(self.config.defaults)
        values.update(self.context)
        values["weight_container"] = weight_container
        values["weight_exists"] = weight_exists
        values["not_save_vis"] = not bool(values.get("save_vis"))
        return values

    @staticmethod
    def _condition(name: str, values: dict[str, Any]) -> bool:
        return bool(values.get(name, False))

    def _command(self, values: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for token in self.spec.get("command", []):
            if isinstance(token, str):
                result.append(token.format(**values))
                continue
            if not isinstance(token, dict):
                continue
            if token.get("optional_unsupported"):
                # Existing helper scripts do not all support a no-visualization
                # flag. Keep their current tested interface instead of guessing.
                continue
            if not self._condition(token.get("when", ""), values):
                continue
            result.append(token["flag"].format(**values))
            if "value" in token:
                result.append(str(token["value"]).format(**values))
        return result

    def build(self) -> tuple[list[str], list[Path]]:
        run_dir = Path(self.context["run_dir"])
        input_dir = Path(self.context["input_dir"])
        pipeline_root = self.config.pipeline_root
        weight_host, weight_container = self._weight()
        values = self._values(weight_container, bool(weight_host and weight_host.exists()))

        mounts = [
            Mount(pipeline_root, "/workspace/pipeline", "ro"),
            Mount(input_dir, "/workspace/input", "ro"),
            Mount(run_dir, "/workspace/output", "rw"),
        ]

        repo = self.spec.get("repo")
        if repo:
            mounts.append(Mount(self.config.resolve(repo["host_path"]), repo["container_path"], repo.get("mode", "rw")))

        if weight_host:
            container_dir = self.spec["weight"]["container_dir"]
            mounts.append(Mount(weight_host.parent, container_dir, "ro"))

        for item in self.spec.get("mounts", []):
            mounts.append(Mount(self.config.resolve(item["host_path"]), item["container_path"], item.get("mode", "rw")))

        inner = self._command(values)
        command = build_command(
            image=self.spec["docker_image"],
            workdir=self.spec["workdir"],
            mounts=mounts,
            env={k: str(v).format(**values) for k, v in self.spec.get("environment", {}).items()},
            inner=inner,
            shm_size=str(self.config.defaults.get("docker_shm_size", "8g")),
        )
        expected = [run_dir / p for p in self.spec.get("expected_outputs", [])]
        return command, expected

    def run(self) -> StageResult:
        command, expected = self.build()
        if self.context.get("skip_existing") and expected and all(p.exists() for p in expected):
            return StageResult(self.key, "skipped", "Expected outputs already exist.", command, expected)
        if self.context.get("overwrite"):
            for p in expected:
                if p.is_file():
                    p.unlink()
        log_path = Path(self.context["run_dir"]) / "logs" / f"{self.key}.log"
        rc = run_streaming(command, log_path, dry_run=bool(self.context.get("dry_run")))
        if rc != 0:
            return StageResult(self.key, "failed", f"Docker exited with code {rc}. See {log_path}", command, expected)
        if self.context.get("dry_run"):
            return StageResult(self.key, "dry_run", "Command generated.", command, expected)
        missing = [p for p in expected if not p.exists()]
        if missing:
            return StageResult(self.key, "failed", "Missing outputs: " + ", ".join(str(p) for p in missing), command, expected)
        return StageResult(self.key, "success", "Expected outputs created.", command, expected)
