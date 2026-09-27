"""Local hand-model deployment contract for the SMPL mainline.

InterWild and WiLoR are optional hand-side providers.  They do not replace the
current 6,890-vertex SMPL + 17x6890 COCO observation regressor.  This module
keeps their large dependencies out of the normal import path and exposes only
asset checks, command construction, and a common result schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import inspect
from pathlib import Path
import pickle
import subprocess
from typing import Any, Sequence

import numpy as np

from .project_paths import project_path


INTERWILD_ROOT = project_path("third_party", "InterWild")
WILOR_ROOT = project_path("third_party", "WiLoR")


@dataclass(frozen=True)
class HandModelAssets:
    """Resolved local assets for one optional backend."""

    name: str
    source_root: Path
    checkpoint: Path
    mano_files: tuple[Path, ...]
    detector: Path | None = None


@dataclass
class HandObservation:
    """Backend-neutral hand result; absent fields remain unavailable.

    Pixel coordinates must be mapped back to the original fisheye image before
    any stereo operation.  3D fields are model/camera-local hypotheses until
    a project calibration and cross-view quality gate accept them.
    """

    backend: str
    side: str
    bbox_xyxy: Any = None
    keypoints_2d: Any = None
    keypoints_2d_confidence: Any = None
    keypoints_3d_local: Any = None
    vertices_3d_local: Any = None
    mano_pose: Any = None
    mano_shape: Any = None
    camera_translation: Any = None
    coordinate_frame: str = "model_local_unaccepted"
    metadata: dict[str, Any] = field(default_factory=dict)


def local_assets() -> dict[str, HandModelAssets]:
    """Return expected local paths without checking or loading large files."""

    return {
        "wilor": HandModelAssets(
            name="wilor",
            source_root=WILOR_ROOT,
            checkpoint=WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt",
            detector=WILOR_ROOT / "pretrained_models" / "detector.pt",
            mano_files=(WILOR_ROOT / "mano_data" / "MANO_RIGHT.pkl",),
        ),
        "interwild": HandModelAssets(
            name="interwild",
            source_root=INTERWILD_ROOT,
            checkpoint=INTERWILD_ROOT / "demo" / "snapshot_6.pth",
            mano_files=(
                INTERWILD_ROOT / "common" / "utils" / "human_model_files" / "mano" / "MANO_RIGHT.pkl",
                INTERWILD_ROOT / "common" / "utils" / "human_model_files" / "mano" / "MANO_LEFT.pkl",
            ),
        ),
    }


def _file_status(path: Path, minimum_bytes: int = 1) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "size_ok": path.is_file() and path.stat().st_size >= minimum_bytes,
    }


def _mano_status(path: Path) -> dict[str, Any]:
    result = _file_status(path, 1_000_000)
    if not result["exists"]:
        result["keys_ok"] = False
        return result
    try:
        # MANO pickles were produced with old chumpy/numpy APIs.  Keep this
        # compatibility local to the asset preflight; do not alter the SMPL
        # application's global runtime behavior.
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
        for name, value in {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "str": str,
            "unicode": str,
        }.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        with path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        required = {"f", "J_regressor", "hands_components", "shapedirs"}
        result["keys_ok"] = required.issubset(data)
        result["missing_keys"] = sorted(required.difference(data))
    except Exception as exc:  # pragma: no cover - depends on local MANO pickle
        result["keys_ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def preflight_backend(name: str) -> dict[str, Any]:
    """Check local assets and optional Python modules without model inference."""

    assets = local_assets()[name]
    checkpoint_minimum = 100_000_000
    report: dict[str, Any] = {
        "backend": name,
        "source_root": str(assets.source_root),
        "source_present": assets.source_root.is_dir(),
        "checkpoint": _file_status(assets.checkpoint, checkpoint_minimum),
        "mano": [_mano_status(path) for path in assets.mano_files],
        "detector": None if assets.detector is None else _file_status(assets.detector, 10_000_000),
    }
    modules = {
        "torch": importlib.util.find_spec("torch") is not None,
        "smplx": importlib.util.find_spec("smplx") is not None,
    }
    if name == "wilor":
        modules.update({
            "ultralytics": importlib.util.find_spec("ultralytics") is not None,
            "torchvision": importlib.util.find_spec("torchvision") is not None,
            "timm": importlib.util.find_spec("timm") is not None,
            "einops": importlib.util.find_spec("einops") is not None,
        })
    else:
        modules.update({
            "torchvision": importlib.util.find_spec("torchvision") is not None,
            "yacs": importlib.util.find_spec("yacs") is not None,
        })
    report["python_modules"] = modules
    report["assets_ready"] = (
        report["checkpoint"]["size_ok"]
        and all(item["size_ok"] and item.get("keys_ok", False) for item in report["mano"])
        and (report["detector"] is None or report["detector"]["size_ok"])
    )
    report["inference_dependencies_ready"] = report["assets_ready"] and all(modules.values())
    return report


def build_wilor_command(
    python_executable: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    fast: bool = False,
) -> tuple[list[str], Path]:
    """Build the official WiLoR demo command and its required working directory."""

    args = [str(python_executable), "demo.py", "--img_folder", str(Path(image_dir).resolve()),
            "--out_folder", str(Path(output_dir).resolve()), "--save_mesh"]
    if fast:
        args.append("--fast")
    return args, WILOR_ROOT


def build_interwild_command(
    python_executable: str | Path,
    *,
    gpu: str = "0",
) -> tuple[list[str], Path]:
    """Build the official InterWild demo command.

    The upstream demo reads ``demo/images`` and writes ``demo/boxes``,
    ``demo/meshes``, ``demo/params`` and ``demo/renders``.  A future project
    adapter should stage audited hand/body crops there and copy the outputs to
    a research record; it must not silently feed raw experiment frames.
    """

    return [str(python_executable), "demo.py", "--gpu", str(gpu)], INTERWILD_ROOT / "demo"


def run_checked(command: Sequence[str], cwd: str | Path, *, dry_run: bool = True) -> subprocess.CompletedProcess[str] | list[str]:
    """Run an explicitly supplied backend command, defaulting to dry-run."""

    if dry_run:
        return list(command)
    return subprocess.run(command, cwd=str(cwd), check=True, text=True, capture_output=True)


__all__ = [
    "HandModelAssets",
    "HandObservation",
    "INTERWILD_ROOT",
    "WILOR_ROOT",
    "local_assets",
    "preflight_backend",
    "build_wilor_command",
    "build_interwild_command",
    "run_checked",
]
