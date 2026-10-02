#!/usr/bin/env python3
"""Read-only audit of refinement drift and VPoser parameter provenance.

Encoder/decoder reconstruction is a diagnostic, not a physical validity gate
or an exact distance to the VPoser decoder manifold. No optimization is run.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "realtime_app"))
NAMES = ("left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
         "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
         "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist")


def matrices(axis_angle):
    a = np.asarray(axis_angle)
    if a.ndim != 2 or a.shape[1] != 63 or not np.isfinite(a).all():
        raise ValueError("expected finite [frames,63] SMPL-H body pose")
    return Rotation.from_rotvec(a.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)


def rotation_degrees(a, b):
    if a.shape != b.shape or a.shape[-2:] != (3, 3):
        raise ValueError("rotation shape mismatch")
    relative = np.swapaxes(a, -1, -2) @ b
    return np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(a.shape[:-2])


def stats(x):
    x = np.asarray(x)
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="original VPoser result, not an intermediate refinement")
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--vposer-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse existing audit")
    import torch
    from pose_app.smplx_fitting import load_vposer_explicit
    torch.set_num_threads(2)
    vp, _, checkpoint = load_vposer_explicit(args.vposer_dir, "cpu")
    if vp.training or any(p.requires_grad for p in vp.parameters()):
        raise RuntimeError("VPoser audit requires frozen eval model")
    source = np.load(args.source, allow_pickle=False)
    baseline = matrices(source["body_pose"])
    with torch.no_grad():
        decoded = vp.decode(torch.as_tensor(source["vposer_latent"], dtype=torch.float32))["pose_body"].reshape(-1, 63).numpy()
    direct_error = rotation_degrees(baseline, matrices(decoded))
    if direct_error.max() > .01:
        raise ValueError("source latent does not reproduce original body pose")
    records = []
    for path in [args.source] + args.results:
        z = np.load(path, allow_pickle=False)
        ids = np.asarray(z["pair_id"], dtype=int) if "pair_id" in z.files else np.arange(len(z["body_pose"]))
        if len(ids) != len(z["body_pose"]) or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(baseline)) or np.any(np.diff(ids) != 1):
            raise ValueError("audit requires explicit contiguous source frame identity")
        rot = matrices(z["body_pose"])
        with torch.no_grad():
            encoded = vp.encode(torch.as_tensor(z["body_pose"], dtype=torch.float32)).mean
            reconstructed = vp.decode(encoded)["pose_body"].reshape(-1, 63).numpy()
        reconstruction = rotation_degrees(rot, matrices(reconstructed))
        drift = rotation_degrees(baseline[ids], rot)
        absolute = np.rad2deg(Rotation.from_matrix(rot.reshape(-1, 3, 3)).magnitude()).reshape(-1, 21)
        temporal = rotation_degrees(rot[:-1], rot[1:])
        metadata_path = path.parent / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf8")) if metadata_path.exists() else None
        records.append({"path": str(path.resolve()), "has_vposer_latent": "vposer_latent" in z.files,
            "run_config": metadata["config"] if metadata else None,
            "joint_drift_from_original_deg": {n: stats(drift[:, j]) for j, n in enumerate(NAMES)},
            "joint_step_rotation_deg": {n: stats(temporal[:, j]) for j, n in enumerate(NAMES)},
            "local_wrist_rotation_deg": {n: stats(absolute[:, j]) for j, n in enumerate(NAMES) if j >= 19},
            "encoder_mean_reconstruction_rotation_deg": stats(reconstruction),
            "encoder_mean_squared": float(encoded.square().mean()),
            "frame_records": [{"pair_id": int(ids[i]), "drift_deg": drift[i].tolist(),
                               "reconstruction_deg": reconstruction[i].tolist()} for i in range(len(rot))]})
    audit = {"status": "logic_audit_not_physical_truth", "source": str(args.source.resolve()),
             "checkpoint": str(checkpoint), "source_saved_latent_decode_error_deg": stats(direct_error),
             "encoder_reconstruction_limit": "not nearest decoder pose, not likelihood, not an anatomical validity test",
             "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"source_replay": stats(direct_error), "results": [{k: r[k] for k in
        ("path", "has_vposer_latent", "encoder_mean_reconstruction_rotation_deg", "local_wrist_rotation_deg")} for r in records]}, indent=2))


if __name__ == "__main__":
    main()
