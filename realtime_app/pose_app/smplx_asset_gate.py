from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


MODEL_FILENAMES = ("SMPLX_MALE.npz", "SMPLX_MALE.pkl")


def resolve_male_model(model_path: str | Path) -> Path | None:
    root = Path(model_path).expanduser().resolve()
    if root.is_file() and root.name in MODEL_FILENAMES:
        return root
    candidates = [root / name for name in MODEL_FILENAMES]
    candidates.extend(root / "smplx" / name for name in MODEL_FILENAMES)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def inspect_smplx_assets(
    model_path: str | Path,
    vposer_path: str | Path | None = None,
    *,
    smplx_library_available: bool | None = None,
    body_prior_library_available: bool | None = None,
) -> dict[str, Any]:
    smplx_available = (
        importlib.util.find_spec("smplx") is not None
        if smplx_library_available is None else smplx_library_available
    )
    body_prior_available = (
        importlib.util.find_spec("human_body_prior") is not None
        if body_prior_library_available is None else body_prior_library_available
    )
    model_file = resolve_male_model(model_path)
    vposer = None if vposer_path is None else Path(vposer_path).expanduser().resolve()
    vposer_exists = bool(vposer and vposer.exists())
    missing = []
    if not smplx_available:
        missing.append("python_package_smplx")
    if model_file is None:
        missing.append("licensed_SMPLX_MALE_model")
    if not body_prior_available:
        missing.append("python_package_human_body_prior")
    if not vposer_exists:
        missing.append("licensed_VPoser_checkpoint")
    return {
        "smplx_library_available": smplx_available,
        "male_model": None if model_file is None else str(model_file),
        "human_body_prior_available": body_prior_available,
        "vposer_path": None if vposer is None else str(vposer),
        "vposer_exists": vposer_exists,
        "joint_only_debug_ready": bool(smplx_available and model_file),
        "temporal_pose_prior_ready": bool(
            smplx_available and model_file and body_prior_available and vposer_exists
        ),
        "missing": missing,
        "policy": (
            "Model and VPoser files must be obtained by the user from their official licensed sources; "
            "the project does not download or redistribute them."
        ),
    }
