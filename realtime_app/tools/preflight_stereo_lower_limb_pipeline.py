#!/usr/bin/env python3
"""Report hardware-free readiness for the live stereo lower-limb pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.camera_registry import load_camera_registry  # noqa: E402
from pose_app.config import load_config  # noqa: E402
from pose_app.fixed_coordinate import load_coordinate_transform  # noqa: E402
from pose_app.pipeline_preflight import (  # noqa: E402
    PreflightCheck,
    configured_model_checks,
    preflight_status,
    recommended_live_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-registry", type=Path, required=True)
    parser.add_argument("--model", choices=["pmpose", "yolo26x_pose"], default="pmpose")
    parser.add_argument("--coordinate-transform", type=Path)
    parser.add_argument("--allow-test-coordinate-transform", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.allow_test_coordinate_transform and args.coordinate_transform is None:
        raise ValueError("--allow-test-coordinate-transform requires --coordinate-transform")
    checks: list[PreflightCheck] = []
    config_path = args.config.resolve()
    calibration_path = args.calibration.resolve()
    registry_path = args.camera_registry.resolve()
    transform_path = args.coordinate_transform.resolve() if args.coordinate_transform else None

    camera_backend = "auto"
    try:
        config = load_config(config_path)
        camera_backend = config.camera.backend
        checks.append(PreflightCheck("app_config", "pass", f"loaded: {config_path}"))
        checks.extend(configured_model_checks(config, args.model))
    except Exception as exc:
        checks.append(PreflightCheck("app_config", "error", str(exc)))

    try:
        calibration = StereoCalibration.load(calibration_path)
        checks.append(
            PreflightCheck(
                "stereo_calibration",
                "pass",
                f"model={calibration.camera_model}, unit={calibration.length_unit}, baseline={calibration.baseline}",
            )
        )
    except Exception as exc:
        checks.append(PreflightCheck("stereo_calibration", "error", str(exc)))

    try:
        left, right = load_camera_registry(registry_path)
        checks.append(
            PreflightCheck(
                "camera_registry_schema",
                "pass",
                f"cam0={left.role}, cam1={right.role}; physical devices are intentionally not enumerated",
            )
        )
    except Exception as exc:
        checks.append(PreflightCheck("camera_registry_schema", "error", str(exc)))

    if transform_path is None:
        checks.append(
            PreflightCheck(
                "coordinate_transform",
                "pending_physical",
                "not supplied; T3 will remain not_configured until a measured_locked transform is captured",
            )
        )
    else:
        try:
            transform = load_coordinate_transform(
                transform_path,
                allow_test_transform=args.allow_test_coordinate_transform,
            )
            checks.append(
                PreflightCheck(
                    "coordinate_transform",
                    "pass" if transform.status == "measured_locked" else "test_only",
                    f"status={transform.status}, target_frame={transform.target_frame}",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("coordinate_transform", "error", str(exc)))

    software_checks = [check for check in checks if check.status != "pending_physical"]
    report = {
        "status": preflight_status(software_checks),
        "checks": [check.to_dict() for check in checks],
        "recommended_live_command": recommended_live_command(
            config_path=config_path,
            calibration_path=calibration_path,
            registry_path=registry_path,
            model=args.model,
            coordinate_transform_path=(
                transform_path
                if any(check.name == "coordinate_transform" and check.status == "pass" for check in checks)
                else None
            ),
            camera_backend=camera_backend,
        ),
        "not_performed": [
            "camera enumeration or camera open",
            "camera pairing or image capture",
            "Docker image/container/service check",
            "GPU check",
            "pose inference or triangulation",
            "physical coordinate or static-reference validation",
        ],
        "interpretation": (
            "This is a configuration-only preflight. software_ready means the inspected local "
            "configuration is self-consistent; it is not a live-camera, service, performance, "
            "or physical-coordinate acceptance result."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing preflight report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "software_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
