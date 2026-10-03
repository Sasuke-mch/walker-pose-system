"""SMPL mesh to MS COCO-17 observation interface.

This module deliberately targets the 6,890-vertex SMPL topology.  It does not
accept SMPL-X vertices and it does not map COCO names directly to SMPL's
internal kinematic joints.  Instead, a fixed 17 x 6890 regressor produces the
model-side COCO observation points from the posed surface mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect

import numpy as np


COCO17_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SMPL_VERTEX_COUNT = 6890
COCO17_COUNT = len(COCO17_NAMES)
MALE_MODEL_FILENAMES = (
    "SMPL_MALE.pkl",
    "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "basicModel_m_lbs_10_207_0_v1.0.0.pkl",
)


def resolve_smpl_male_model(model_path: str | Path) -> Path | None:
    """Resolve an official male SMPL pickle without accepting SMPL-X files."""
    root = Path(model_path).expanduser().resolve()
    if root.is_file() and root.name in MALE_MODEL_FILENAMES:
        return root
    candidates: list[Path] = []
    for name in MALE_MODEL_FILENAMES:
        candidates.extend((root / name, root / "smpl" / name))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_coco17_regressor(regressor: np.ndarray) -> np.ndarray:
    """Validate and return a float64 17 x 6890 observation regressor."""
    matrix = np.asarray(regressor, dtype=np.float64)
    expected = (COCO17_COUNT, SMPL_VERTEX_COUNT)
    if matrix.shape != expected:
        raise ValueError(f"COCO regressor must have shape {expected}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("COCO regressor contains NaN or Inf")
    if np.any(matrix < 0.0):
        raise ValueError("COCO regressor must use non-negative weights")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"COCO regressor rows must sum to one, got {row_sums}")
    return matrix


def load_coco17_regressor(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing COCO regressor: {source}")
    return validate_coco17_regressor(np.load(source, allow_pickle=False))


def regress_coco17_numpy(vertices: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    """Compute model-side COCO-17 points from (..., 6890, 3) vertices."""
    mesh = np.asarray(vertices)
    if mesh.shape[-2:] != (SMPL_VERTEX_COUNT, 3):
        raise ValueError(
            "SMPL vertices must end in (6890, 3); "
            f"received {mesh.shape}"
        )
    matrix = validate_coco17_regressor(regressor)
    return np.einsum("jv,...vc->...jc", matrix, mesh)


def regress_coco17_torch(vertices: "Any", regressor: np.ndarray | "Any") -> "Any":
    """Differentiable model-side COCO-17 points for torch SMPL vertices."""
    import torch

    if tuple(vertices.shape[-2:]) != (SMPL_VERTEX_COUNT, 3):
        raise ValueError(
            "SMPL vertices must end in (6890, 3); "
            f"received {tuple(vertices.shape)}"
        )
    if torch.is_tensor(regressor):
        matrix_np = regressor.detach().cpu().numpy()
    else:
        matrix_np = np.asarray(regressor)
    validate_coco17_regressor(matrix_np)
    matrix = torch.as_tensor(regressor, dtype=vertices.dtype, device=vertices.device)
    return torch.einsum("jv,...vc->...jc", matrix, vertices)


def inspect_smpl_coco_assets(
    model_path: str | Path,
    regressor_path: str | Path,
) -> dict[str, Any]:
    """Return a non-mutating preflight report for the clean SMPL mainline."""
    model = resolve_smpl_male_model(model_path)
    regressor_source = Path(regressor_path).expanduser().resolve()
    regressor_status = "missing"
    regressor_error: str | None = None
    if regressor_source.is_file():
        try:
            load_coco17_regressor(regressor_source)
            regressor_status = "valid_17x6890"
        except (OSError, ValueError) as exc:
            regressor_status = "invalid"
            regressor_error = str(exc)
    missing: list[str] = []
    if model is None:
        missing.append("licensed_SMPL_MALE_6890_model")
    if regressor_status != "valid_17x6890":
        missing.append("Pose2Mesh_J_regressor_coco_17x6890")
    return {
        "male_model": None if model is None else str(model),
        "regressor": str(regressor_source),
        "regressor_status": regressor_status,
        "regressor_error": regressor_error,
        "ready": not missing,
        "missing": missing,
        "required_model_topology": "SMPL_6890_vertices",
        "required_observation_order": list(COCO17_NAMES),
        "forbidden_model_topology": "SMPL-X_10475_vertices",
    }


def _install_legacy_smpl_pickle_compatibility() -> None:
    """Provide removed Python/NumPy names needed only to unpickle SMPL v1.0.

    The official Python model predates Python 3.12 and NumPy 2.  Chumpy is
    imported during pickle loading even though the resulting arrays are then
    converted to torch tensors.  Keep the compatibility shim local to our
    loader rather than modifying site-packages or the licensed model file.
    """
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": np.bool_,
        "int": np.int64,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "unicode": np.str_,
        "str": np.str_,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def load_smpl_male(
    model_path: str | Path,
    *,
    batch_size: int = 1,
    device: str = "cpu",
    dtype: "Any" = None,
) -> "Any":
    """Load the official 6,890-vertex male SMPL model with PyTorch."""
    import torch
    import smplx

    resolved = resolve_smpl_male_model(model_path)
    if resolved is None:
        raise FileNotFoundError(f"missing official male SMPL model under {model_path}")
    _install_legacy_smpl_pickle_compatibility()
    model = smplx.SMPL(
        str(resolved),
        gender="male",
        batch_size=batch_size,
        dtype=torch.float32 if dtype is None else dtype,
    ).to(device)
    if model.get_num_verts() != SMPL_VERTEX_COUNT:
        raise ValueError(
            f"expected {SMPL_VERTEX_COUNT} SMPL vertices, got {model.get_num_verts()}"
        )
    return model


__all__ = [
    "COCO17_NAMES",
    "SMPL_VERTEX_COUNT",
    "MALE_MODEL_FILENAMES",
    "resolve_smpl_male_model",
    "validate_coco17_regressor",
    "load_coco17_regressor",
    "regress_coco17_numpy",
    "regress_coco17_torch",
    "inspect_smpl_coco_assets",
    "load_smpl_male",
]
