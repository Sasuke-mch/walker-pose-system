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
RAW_FISHEYE_PIXEL_FRAME = "raw_fisheye"
MODEL_LOCAL_3D_FRAME = "model_local_unaccepted"


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
    coordinate_frame: str = MODEL_LOCAL_3D_FRAME
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_raw_fisheye_pixels(
    points: Any,
    image_size: tuple[int, int],
    *,
    allow_out_of_bounds: bool = False,
) -> Any:
    """Validate Nx2 pixels in the original, unrotated fisheye image frame.

    ``image_size`` is ``(width, height)``.  The function deliberately does not
    clip points: clipping would hide crop/rotation mistakes before a stereo
    quality gate can reject them.
    """

    import numpy as np

    pixels = np.asarray(points, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"expected Nx2 pixel array, got {pixels.shape}")
    if not np.isfinite(pixels).all():
        raise ValueError("pixel array contains NaN or Inf")
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image_size}")
    if not allow_out_of_bounds and (
        (pixels[:, 0] < 0).any()
        or (pixels[:, 0] >= width).any()
        or (pixels[:, 1] < 0).any()
        or (pixels[:, 1] >= height).any()
    ):
        raise ValueError("pixel array is outside the original fisheye image")
    return pixels


def project_wilor_vertices_to_image(
    vertices_3d_local: Any,
    camera_translation: Any,
    focal_length: float,
    image_size: tuple[int, int],
) -> Any:
    """Reproduce WiLoR's full-image projection in the raw-image frame.

    ``vertices_3d_local`` must already contain WiLoR's handedness correction;
    this function does not guess left/right from a crop or image orientation.
    """

    import numpy as np

    vertices = np.asarray(vertices_3d_local, dtype=np.float64)
    translation = np.asarray(camera_translation, dtype=np.float64).reshape(3)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected Nx3 vertices, got {vertices.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(translation).all():
        raise ValueError("WiLoR 3D output contains NaN or Inf")
    width, height = (float(image_size[0]), float(image_size[1]))
    points = vertices + translation[None, :]
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("WiLoR camera-translated points must have positive depth")
    pixels = points[:, :2] / points[:, 2:3]
    pixels[:, 0] = pixels[:, 0] * float(focal_length) + width / 2.0
    pixels[:, 1] = pixels[:, 1] * float(focal_length) + height / 2.0
    return pixels


def make_wilor_observation(
    *,
    side: str,
    bbox_xyxy: Any,
    keypoints_2d: Any,
    image_size: tuple[int, int],
    keypoints_3d_local: Any = None,
    vertices_3d_local: Any = None,
    mano_pose: Any = None,
    mano_shape: Any = None,
    camera_translation: Any = None,
    confidence: Any = None,
) -> HandObservation:
    """Adapt WiLoR's full-image outputs without accepting model-local 3D.

    The caller must pass points produced from the same unrotated raw fisheye
    image that was given to WiLoR.  The 3D fields are retained for audit only;
    downstream stereo code must check ``coordinate_frame`` before consuming
    them.
    """

    pixels = validate_raw_fisheye_pixels(keypoints_2d, image_size)
    return HandObservation(
        backend="wilor",
        side=str(side),
        bbox_xyxy=bbox_xyxy,
        keypoints_2d=pixels,
        keypoints_2d_confidence=confidence,
        keypoints_3d_local=keypoints_3d_local,
        vertices_3d_local=vertices_3d_local,
        mano_pose=mano_pose,
        mano_shape=mano_shape,
        camera_translation=camera_translation,
        coordinate_frame=MODEL_LOCAL_3D_FRAME,
        metadata={
            "pixel_frame": RAW_FISHEYE_PIXEL_FRAME,
            "input_image_transform": "identity_raw_fisheye",
            "three_d_acceptance": "blocked_until_calibrated_cross_view_gate",
        },
    )


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
            "kornia": importlib.util.find_spec("kornia") is not None,
            "pytorch3d": importlib.util.find_spec("pytorch3d") is not None,
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
    "RAW_FISHEYE_PIXEL_FRAME",
    "MODEL_LOCAL_3D_FRAME",
    "local_assets",
    "preflight_backend",
    "validate_raw_fisheye_pixels",
    "project_wilor_vertices_to_image",
    "make_wilor_observation",
    "build_wilor_command",
    "build_interwild_command",
    "run_checked",
]
