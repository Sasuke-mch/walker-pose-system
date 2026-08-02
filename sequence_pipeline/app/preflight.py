from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import LoadedConfig
from .utils import resolve_host_path


@dataclass
class CheckItem:
    level: str
    subject: str
    message: str


@dataclass
class CheckReport:
    items: list[CheckItem]

    @property
    def errors(self) -> list[CheckItem]:
        return [x for x in self.items if x.level == "ERROR"]

    def print(self) -> None:
        for item in self.items:
            print(f"[{item.level}] {item.subject}: {item.message}")
        print(f"\nErrors: {len(self.errors)}")


def _path_check(items: list[CheckItem], subject: str, path: Path, kind: str, warning_only: bool = False):
    ok = path.is_file() if kind == "file" else path.is_dir() if kind == "dir" else path.exists()
    if ok:
        items.append(CheckItem("OK", subject, str(path)))
    else:
        items.append(CheckItem("WARN" if warning_only else "ERROR", subject, f"Missing {kind}: {path}"))


def run_preflight(config: LoadedConfig, selected: list[str], weight_overrides: dict[str, str], check_docker: bool = True, input_dir: Path | None = None) -> CheckReport:
    items: list[CheckItem] = []
    items.append(CheckItem("OK" if sys.version_info >= (3, 10) else "ERROR", "Python", sys.version.split()[0]))
    _path_check(items, "Project root", config.project_root, "dir")
    if input_dir is not None:
        _path_check(items, "Input directory", input_dir, "dir")

    if check_docker:
        docker = shutil.which("docker")
        if not docker:
            items.append(CheckItem("ERROR", "Docker", "docker command not found"))
        else:
            try:
                cp = subprocess.run([docker, "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=20)
                if cp.returncode == 0:
                    items.append(CheckItem("OK", "Docker daemon", cp.stdout.strip() or "available"))
                else:
                    items.append(CheckItem("ERROR", "Docker daemon", cp.stderr.strip() or cp.stdout.strip()))
            except Exception as exc:
                items.append(CheckItem("ERROR", "Docker daemon", str(exc)))

    stages = list(selected)
    if any(config.model(k).get("requires_detector") for k in selected):
        stages = ["yolo26_detector"] + stages

    seen_images = set()
    for key in stages:
        model = config.model(key)
        if not model.get("enabled", True):
            items.append(CheckItem("ERROR", key, "disabled in configuration"))
            continue
        for req in model.get("required_paths", []):
            path = config.resolve(req["path"])
            _path_check(items, f"{key} required path", path, req.get("type", "exists"), req.get("warning_only", False))
        repo = model.get("repo")
        if repo:
            _path_check(items, f"{key} repository", config.resolve(repo["host_path"]), "dir")

        weight = model.get("weight", {})
        raw = weight_overrides.get(key, weight.get("host_path", ""))
        if raw:
            weight_path = config.resolve(raw)
            _path_check(items, f"{key} weight", weight_path, "file", not weight.get("required", False))
            fixed = weight.get("container_file_name")
            if fixed and not weight.get("supports_arbitrary_name", False) and weight_path.name != fixed:
                items.append(CheckItem("ERROR", f"{key} weight name", f"Expected filename {fixed}, got {weight_path.name}"))
        elif weight.get("required", False):
            items.append(CheckItem("ERROR", f"{key} weight", "No weight path configured"))
        else:
            items.append(CheckItem("WARN", f"{key} weight", weight.get("note", "Optional weight not configured")))

        if check_docker:
            image = str(model.get("docker_image", "")).strip()
            if image and image not in seen_images:
                seen_images.add(image)

                if not docker:
                    items.append(
                        CheckItem(
                            "ERROR",
                            f"Docker image {image}",
                            "docker executable was not found",
                        )
                    )
                else:
                    try:
                        cp = subprocess.run(
                            [
                                docker,
                                "image",
                                "ls",
                                "--quiet",
                                "--no-trunc",
                                image,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

                        image_id = cp.stdout.strip()

                        if cp.returncode == 0 and image_id:
                            items.append(
                                CheckItem(
                                    "OK",
                                    f"Docker image {image}",
                                    image_id.splitlines()[0],
                                )
                            )
                        else:
                            details = cp.stderr.strip() or cp.stdout.strip()
                            items.append(
                                CheckItem(
                                    "ERROR",
                                    f"Docker image {image}",
                                    details or "image tag was not listed locally",
                                )
                            )
                    except Exception as exc:
                        items.append(
                            CheckItem(
                                "ERROR",
                                f"Docker image {image}",
                                str(exc),
                            )
                        )


    return CheckReport(items)
