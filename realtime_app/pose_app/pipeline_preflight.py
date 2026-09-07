"""Hardware-free configuration checks for the live stereo lower-limb command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _path_check(name: str, path: Path, *, directory: bool) -> PreflightCheck:
    exists = path.is_dir() if directory else path.is_file()
    kind = "directory" if directory else "file"
    return PreflightCheck(
        name,
        "pass" if exists else "error",
        f"{kind} {'exists' if exists else 'is missing'}: {path}",
    )


def configured_model_checks(config: AppConfig, model: str) -> list[PreflightCheck]:
    """Check only local paths and service-port configuration, never the services."""

    if model not in {"pmpose", "yolo26x_pose"}:
        raise ValueError(f"unsupported model for preflight: {model}")
    checks: list[PreflightCheck] = []
    if model == "pmpose":
        checks.extend(
            [
                _path_check("detector_repository", config.detector.repo, directory=True),
                _path_check("detector_weight", config.detector.weight, directory=False),
                _path_check("pmpose_repository", config.pmpose.repo, directory=True),
                _path_check("pmpose_cache", config.pmpose.cache, directory=True),
            ]
        )
        ports = {config.detector.host_port, config.pmpose.host_port}
        checks.append(
            PreflightCheck(
                "pmpose_service_ports",
                "pass" if len(ports) == 2 else "error",
                (
                    f"detector={config.detector.host_port}, pmpose={config.pmpose.host_port}"
                    if len(ports) == 2
                    else "detector and PMPose use the same host port"
                ),
            )
        )
    else:
        checks.extend(
            [
                _path_check("pose_repository", config.model.repo, directory=True),
                _path_check("pose_weight", config.model.weight, directory=False),
            ]
        )
    checks.append(
        PreflightCheck(
            "camera_config",
            "pass"
            if (
                config.camera.width > 0
                and config.camera.height > 0
                and config.camera.fps > 0
                and config.camera.backend in {"auto", "msmf", "dshow"}
            )
            else "error",
            f"{config.camera.width}x{config.camera.height} @ {config.camera.fps} FPS, backend={config.camera.backend}",
        )
    )
    return checks


def recommended_live_command(
    *,
    config_path: Path,
    calibration_path: Path,
    registry_path: Path,
    model: str,
    coordinate_transform_path: Path | None,
    camera_backend: str = "auto",
) -> list[str]:
    """Return an argument list; callers may render it, but this function never runs it."""

    command = [
        "python",
        "run_stereo.py",
        "--config",
        str(config_path),
        "--calibration",
        str(calibration_path),
        "--camera-registry",
        str(registry_path),
        "--camera-backend",
        camera_backend,
        "--model",
        model,
        "--left-model-rotation",
        "ccw90",
        "--right-model-rotation",
        "cw90",
        "--stereo-subject-mode",
        "single",
        "--enable-lower-limb-pipeline",
    ]
    if coordinate_transform_path is not None:
        command.extend(["--coordinate-transform", str(coordinate_transform_path)])
    return command


def preflight_status(checks: list[PreflightCheck]) -> str:
    return "software_ready" if all(check.status == "pass" for check in checks) else "software_blocked"
