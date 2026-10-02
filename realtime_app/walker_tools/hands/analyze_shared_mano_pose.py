"""Compare same-input per-frame and shared PCA hand fits (engineering only)."""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    a = np.asarray(values)
    if not a.size:
        return {"count": 0, "median": None, "p95": None}
    return {"count": int(a.size), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95))}


def analyze(perframe: Path, shared: Path):
    with np.load(perframe / "result.npz") as a, np.load(shared / "result.npz") as b:
        sa = json.loads((perframe / "fit_summary.json").read_text(encoding="utf-8"))
        sb = json.loads((shared / "fit_summary.json").read_text(encoding="utf-8"))
        if sa.get("shared_hand_pose") or not sb.get("shared_hand_pose"):
            raise ValueError("expected per-frame control and shared candidate")
        if sa["frames"] != sb["frames"]:
            raise ValueError("different_frame_ranges")
        for key in ("model", "hand_pca_components", "hand_pca_prior_weight",
                    "hand_temporal_weight", "contact_enabled", "steps"):
            if sa[key] != sb[key]:
                raise ValueError(f"comparison_protocol_mismatch:{key}")
        for key in ("weight", "initialization", "hand_2d_weight"):
            if sa["native_mano_prior"][key] != sb["native_mano_prior"][key]:
                raise ValueError(f"native_prior_protocol_mismatch:{key}")
        for key in ("raw_triangulated_points", "body_accepted", "body_quality"):
            if not np.array_equal(a[key], b[key], equal_nan=True):
                raise ValueError(f"input_mismatch:{key}")
        report = {"frames": sa["frames"], "status": "engineering_comparison",
                  "body_parameters_max_abs_difference": {}, "hands": {}}
        for key in ("betas", "body_pose", "global_orient", "transl"):
            report["body_parameters_max_abs_difference"][key] = float(np.max(np.abs(a[key]-b[key])))
        for side in ("left", "right"):
            targets = a[f"mano_{side}_target_rotations"]
            weights = a[f"mano_{side}_view_weights"]
            if not np.array_equal(targets, b[f"mano_{side}_target_rotations"]) or not np.array_equal(weights,b[f"mano_{side}_view_weights"]):
                raise ValueError("native_pose_input_mismatch")
            hand = {"covered_frames": int((weights.sum(1)>0).sum()),
                    "unavailable_frames": np.flatnonzero(weights.sum(1)==0).tolist()}
            for label, result in (("perframe", a), ("shared", b)):
                pose = result[f"{side}_hand_pose"]
                if not np.isfinite(pose).all():
                    raise ValueError("nonfinite_hand_pose")
                rotations = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(-1,15,3,3)
                residual = rotations[:,None].swapaxes(-1,-2) @ targets
                angle = np.degrees(Rotation.from_matrix(residual.reshape(-1,3,3)).magnitude()).reshape(weights.shape+(15,))
                hand[f"{label}_native_rotation_residual_deg"] = stats(angle[weights>0])
                hand[f"{label}_pca_temporal_range_max"] = float(np.ptp(result[f"{side}_hand_pca"],axis=0).max())
            if hand["shared_pca_temporal_range_max"] != 0:
                raise ValueError("shared_hand_pose_is_not_constant")
            hand["shared_pca_coefficients"] = b[f"{side}_hand_pca"][0].tolist()
            p, q = a[f"hand_points_{side}"], b[f"hand_points_{side}"]
            covered = weights.sum(1)>0
            hand["wrist_change_mm"] = stats(np.linalg.norm(p[:,0]-q[:,0],axis=-1)*1000)
            tips = [4,8,12,16,20]
            hand["tip_change_mm_on_covered_frames"] = stats(np.linalg.norm(p[covered][:,tips]-q[covered][:,tips],axis=-1)*1000)
            relative_p, relative_q = p-p[:,:1], q-q[:,:1]
            hand["wrist_relative_tip_change_mm_on_covered_frames"] = stats(np.linalg.norm(relative_p[covered][:,tips]-relative_q[covered][:,tips],axis=-1)*1000)
            report["hands"][side] = hand
        audit = json.loads((shared/"wilor_mano_parameter_audit.json").read_text(encoding="utf-8"))
        report["candidate_reason_counts"] = {camera: dict(Counter(r["reason"] for r in audit[camera])) for camera in ("left_camera","right_camera")}
        report["cross_view_local_rotation_disagreement"] = audit["cross_view_local_rotation_disagreement"]
        report["limits"] = "fixed local articulation only; no walker-relative SE3 constraint, external hand truth or grip validation"
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perframe", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse_existing_output")
    result = analyze(args.perframe, args.shared)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
