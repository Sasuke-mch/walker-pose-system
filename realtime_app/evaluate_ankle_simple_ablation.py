"""Compare two bounded single-parameter ankle-fit experiments against cached448."""
import json
from pathlib import Path

import numpy as np

from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research_records/engineering_validation/G20261012_smplh_2d448_view_v1/fit"
OUT = ROOT / "research_records/engineering_validation/G20261012_ankle_simple_ablation448_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(values):
    values = np.asarray(values)
    assert values.size and np.isfinite(values).all()
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95))}


def amplitude(points):
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=-1))))


def main():
    options = read(BASE / "resolved_options.json")
    cal = load_stereo_fisheye(ROOT / options["calibration_dir"])
    raw = {}
    for side in ("left", "right"):
        rows = raw_side(Path(options[side + "_raw"]), side)
        assert sorted(rows) == list(range(448))
        raw[side] = np.stack([rows[i] for i in range(448)])
    results = {}
    for name, path in (("baseline", BASE), ("weight2x", OUT / "weight2x"),
                       ("steps2x", OUT / "steps2x")):
        current = read(path / "resolved_options.json")
        differences = {k for k in current if current[k] != options[k]}
        allowed = {"output_dir"}
        if name == "weight2x":
            allowed.add("body_reprojection_weight")
        if name == "steps2x":
            allowed.add("joint_steps")
        assert differences <= allowed, differences
        with np.load(path / "result.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["pair_id"], np.arange(448))
            joints = archive["predicted_coco"].astype(float)
            valid = {s: archive["body_" + s + "_2d_valid"] for s in raw}
        ankles, other, views = [], [], {}
        for side in raw:
            points = joints if side == "left" else (
                joints @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000)
            projected = fisheye_project_numpy(points,
                cal.K0 if side == "left" else cal.K1,
                cal.D0 if side == "left" else cal.D1)
            error = np.linalg.norm(projected - raw[side][..., :2], axis=-1)
            ankles.extend(error[:, 15:17][valid[side][:, 15:17]])
            other.extend(error[:, :15][valid[side][:, :15]])
            assert valid[side][:, 15:17].all()
            observed_relative = raw[side][:, 15, :2] - raw[side][:, 16, :2]
            model_relative = projected[:, 15] - projected[:, 16]
            ratio = amplitude(model_relative) / amplitude(observed_relative)
            views[side] = {"relative_ankle_amplitude_ratio": ratio,
                           "amplitude_ratio_error": abs(ratio - 1),
                           "d2_rms_over_amplitude_diagnostic": {
                               "model": float(np.sqrt(np.mean(np.sum(
                                   np.diff(model_relative, n=2, axis=0)**2, axis=-1)))
                                   / amplitude(model_relative)),
                               "observed": float(np.sqrt(np.mean(np.sum(
                                   np.diff(observed_relative, n=2, axis=0)**2, axis=-1)))
                                   / amplitude(observed_relative))},
                           "relative_ankle_error_px": metric(np.linalg.norm(
                               model_relative - observed_relative, axis=-1)),
                           "left_ankle_px": metric(error[:, 15]),
                           "right_ankle_px": metric(error[:, 16]),
                           "all448_ankle_px": error[:, 15:17].tolist()}
        relative = joints[:, 15:17] - joints[:, 11:13].mean(axis=1)[:, None]
        acceleration = np.linalg.norm(np.diff(relative, n=2, axis=0), axis=-1) * 1000
        results[name] = {"option_changes": sorted(differences), "ankle_px": metric(ankles),
                         "other_joint_px": metric(other), "views": views,
                         "ankle_pelvis_relative_d2_mm_per_frame2": metric(acceleration),
                         "C_history": [r for r in read(path / "fit_summary.json")["history"]
                                       if r["stage"] == "C_body_vposer_refine"]}
    control = results["baseline"]
    for name in ("weight2x", "steps2x"):
        candidate = results[name]
        gates = {
            "ankle_mean_improves15pct": candidate["ankle_px"]["mean"] <= .85 * control["ankle_px"]["mean"],
            "ankle_p95_improves15pct": candidate["ankle_px"]["p95"] <= .85 * control["ankle_px"]["p95"],
            "other_mean_no_more_than10pct_worse": candidate["other_joint_px"]["mean"] <= 1.1 * control["other_joint_px"]["mean"],
            "d2_no_more_than20pct_worse": candidate["ankle_pelvis_relative_d2_mm_per_frame2"]["p95"] <= 1.2 * control["ankle_pelvis_relative_d2_mm_per_frame2"]["p95"],
            "amplitude_closer_both_views": all(candidate["views"][s]["amplitude_ratio_error"]
                                               < control["views"][s]["amplitude_ratio_error"] for s in raw)}
        candidate["gates"] = gates
        candidate["passes"] = all(gates.values())
    report = {"scope": "all448 same-source 2D consistency; no independent gait/contact truth",
              "additional_diagnostic": "D2/amplitude added after initial comparison to distinguish absolute motion scale from relative roughness; does not replace the preregistered gate or prove absence of jitter.",
              "script": str(Path(__file__).relative_to(ROOT)), "results": results}
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(name, json.dumps({"ankle": result["ankle_px"], "other": result["other_joint_px"],
            "d2": result["ankle_pelvis_relative_d2_mm_per_frame2"],
            "amplitude_ratios": [result["views"][s]["relative_ankle_amplitude_ratio"] for s in raw],
            "gates": result.get("gates")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
