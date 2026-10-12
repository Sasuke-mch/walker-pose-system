"""Compare every original frame using independent per-view pixel support."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
from pose_app.smplh_fitting.losses import raw_pixel_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    source = Path("research_records/engineering_validation/G20261005_reference_weight_fit_v1/runs/reference25")
    options = json.loads((source / "resolved_options.json").read_text())
    cal = load_stereo_fisheye(options["calibration_dir"])
    observations = []
    for side in ("left", "right"):
        rows = raw_side(Path(options[f"{side}_raw"]), side)
        if sorted(rows) != list(range(360)):
            raise RuntimeError("original observation frame set differs")
        observations.append(np.stack([rows[i] for i in range(360)]))
    masks = [raw_pixel_support(raw, (cal.image_width, cal.image_height))[0] for raw in observations]
    paths = {"reference25": source / "fit/result.npz"}
    paths.update({name: args.experiment / name / "result.npz"
                  for name in ("pilot", "control_full", "tuned_full")})
    result = {"boundary": "in-sample observation agreement only, no independent 3D accuracy",
              "raw_left": options["left_raw"], "raw_right": options["right_raw"],
              "support_counts": [int(mask.sum()) for mask in masks], "cases": {}}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as fitted:
            coco = fitted["predicted_coco"].astype(float)
            if not np.array_equal(fitted["pair_id"], np.arange(360)):
                raise RuntimeError("all 360 original frames required")
            camera_points = [coco, coco @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000]
            residuals = []
            views = []
            for points, raw, mask, intrinsics, distortion in zip(
                    camera_points, observations, masks, (cal.K0, cal.K1), (cal.D0, cal.D1)):
                pixels = fisheye_project_numpy(points, intrinsics, distortion)
                error = np.linalg.norm(pixels - raw[..., :2], axis=-1)
                if not np.isfinite(error[mask]).all() or (points[..., 2][mask] <= 0).any():
                    raise RuntimeError(f"invalid supported output in {name}")
                residuals.append(error)
                weights = np.clip(raw[..., 2][mask], 0, 1)
                views.append({"mean_px": float(error[mask].mean()),
                              "confidence_weighted_mean_px": float(np.average(error[mask], weights=weights)),
                              "median_px": float(np.median(error[mask])),
                              "p95_px": float(np.percentile(error[mask], 95)),
                              "max_px": float(error[mask].max()),
                              "per_joint_mean_px": [float(error[:, j][mask[:, j]].mean())
                                                    if mask[:, j].any() else None for j in range(17)]})
            all_errors = np.concatenate([e[m] for e, m in zip(residuals, masks)])
            pelvis = coco[:, [11, 12]].mean(1)
            relative = coco - pelvis[:, None]
            acceleration = np.linalg.norm(np.diff(relative[:, [13, 14, 15, 16]], n=2, axis=0), axis=-1) * 1000
            case = {"path": str(path), "frames": 360, "views": views,
                    "mean_px": float(all_errors.mean()), "median_px": float(np.median(all_errors)),
                    "p95_px": float(np.percentile(all_errors, 95)),
                    "mean_of_view_confidence_means_px": float(np.mean([v["confidence_weighted_mean_px"] for v in views])),
                    "betas": fitted["betas"].tolist(), "beta_norm": float(np.linalg.norm(fitted["betas"])),
                    "relative_leg_second_difference_p95_mm_per_frame2": float(np.percentile(acceleration, 95))}
            if name != "reference25":
                summary = json.loads((path.parent / "fit_summary.json").read_text())
                case["body_reprojection"] = summary["body_reprojection"]
                case["last_body_loss_row"] = [r for r in summary["history"] if r["stage"].startswith("C_")][-1]
                case["all_logged_3d_terms_zero"] = all(r["weighted_terms"]["observation_3d"] == 0 for r in summary["history"])
            result["cases"][name] = case
    output = args.experiment / "evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in case.items() if key in
                            ("mean_px", "p95_px", "mean_of_view_confidence_means_px", "beta_norm",
                             "relative_leg_second_difference_p95_mm_per_frame2")}
                      for name, case in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
