"""Compare complete same-input pixel-tracking replays without hiding failures."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def stats(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return {"finite": len(finite), "median": None if not len(finite) else float(np.median(finite)),
            "p95": None if not len(finite) else float(np.percentile(finite, 95)),
            "max": None if not len(finite) else float(finite.max())}


def analyze(root):
    results, curves = {}, {}
    beta = None
    time = None
    for name in ("ekf_sigma15", "map_sigma15", "map_sigma075"):
        directory = root/name
        rows = [json.loads(line) for line in (directory/"trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        p = json.loads((directory/"run_parameters.json").read_text(encoding="utf-8"))
        assert [row["frame_index"] for row in rows] == list(range(448))
        errors = np.asarray([row["reprojection_px"] for row in rows], float)
        with np.load(directory/"result.npz", allow_pickle=False) as data:
            if beta is None:
                beta = data["betas"].copy()
                time = data["timestamp_s"].copy()
            else:
                np.testing.assert_array_equal(beta, data["betas"])
                np.testing.assert_array_equal(time, data["timestamp_s"])
            latent_norm = np.linalg.norm(data["vposer_latent"], axis=1)
            knee_local = np.full((448, 2, 3), np.nan)
            available = data["pose_available"]
            knee_local[available] = Rotation.from_matrix(data["body_rotations"][available][:, [3, 4]].reshape(-1, 3, 3)).as_rotvec().reshape(-1, 2, 3)*180/np.pi
        result = {"frames": len(rows), "all_raw_view_slots": errors.size,
            "raw_error_px": stats(errors), "fit_le25px_count": int(np.sum(errors <= 25)),
            "fit_le25px_rate_whole10752": float(np.sum(errors <= 25)/errors.size),
            "pose_available_frames": summary["pose_available_frames"],
            "derivative_available_samples": summary["derivative_available_joint_samples"],
            "derivative_denominator": 448*12, "model_status_counts": summary["frame_status_counts"],
            "optimizer_status_counts": summary["iterated_update_status_counts"],
            "optimizer_termination_counts": dict(Counter(row["iterated_update"].get("termination", row["iterated_update"]["status"])
                for row in rows if row["iterated_update"] is not None)),
            "prediction_out_of_domain_frames": [row["frame_index"] for row in rows if row["prediction_rejection"] is not None],
            "latent_norm": stats(latent_norm), "step_ms": summary["step_ms_model_geometry_state_update_and_derivative_decode"],
            "whole_vposer_sigma": p["configuration"]["latent_prior_sigma"],
            "knee_local_rotvec_diagnostic_not_clinical": {"left_x": stats(knee_local[:, 0, 0]), "right_x": stats(knee_local[:, 1, 0]),
                "left_abs_y": stats(abs(knee_local[:, 0, 1])), "right_abs_y": stats(abs(knee_local[:, 1, 1])),
                "left_x_below_minus5_degrees_frames": int(np.sum(knee_local[:, 0, 0] < -5)),
                "right_x_below_minus5_degrees_frames": int(np.sum(knee_local[:, 1, 0] < -5))},
            "per_joint": {joint: {"all_views": stats(errors[:, :, j]), "left": stats(errors[:, 0, j]), "right": stats(errors[:, 1, j]),
                "fit_le25px_count": int(np.sum(errors[:, :, j] <= 25)), "whole_view_slots": 896} for j, joint in enumerate(NAMES)}}
        results[name] = result
        curves[name] = (np.nanmedian(errors, axis=(1, 2)), np.nanmedian(errors[:, :, 6:8], axis=(1, 2)), latent_norm)
    output = {"comparison": results, "fixed_beta": beta.tolist(), "fixed_timestamps": True,
        "observation_scope": "same whole448 actual tracker inputs;frame0 manual substitutions identical across variants;no3D observation term",
        "limits": "Raw residual statistics require coverage;local knee axis angles are model diagnostics,not clinical angles;no independent real3D or derivative truth"}
    (root/"comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"ekf_sigma15": ("0.5", "--", "EKF / VPoser x1"),
              "map_sigma15": ("black", "-", "Iterated MAP / VPoser x1"),
              "map_sigma075": ("0.25", ":", "Iterated MAP / VPoser x4")}
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for name, values in curves.items():
        color, style, label = styles[name]
        for axis, value in zip(axes, values):
            axis.plot(time, value, color=color, linestyle=style, linewidth=1., label=label)
    for axis in axes:
        axis.grid(axis="y", color="0.85", linewidth=.5)
    axes[0].set_ylabel("All-body median (px)")
    axes[1].set_ylabel("Hip median (px)")
    axes[2].set_ylabel("Whole VPoser norm")
    axes[2].set_xlabel("Actual nonuniform capture time (s)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Complete 448 frames: pixel consistency and whole-body prior; no smoothing")
    fig.savefig(root/"pixel_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({name: {key: row[key] for key in ("raw_error_px", "fit_le25px_rate_whole10752", "derivative_available_samples", "latent_norm", "optimizer_status_counts")}
                      for name, row in results.items()}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    analyze(parser.parse_args().root)
