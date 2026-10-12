"""Same-source cached448 acceptance checks for the ankle-only multiplier."""
import json
from pathlib import Path

import numpy as np

from audit_ankle_phase import correlation
from evaluate_ankle_simple_ablation import amplitude, metric, read
from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research_records/engineering_validation/G20261012_ankle_simple_ablation448_v1/weight2x"
OUT = ROOT / "research_records/engineering_validation/G20261012_ankle_weight448_v1"


def main():
    options = read(BASE / "resolved_options.json")
    cal = load_stereo_fisheye(ROOT / options["calibration_dir"])
    raw = {}
    for side in ("left", "right"):
        rows = raw_side(Path(options[side + "_raw"]), side)
        assert sorted(rows) == list(range(448))
        raw[side] = np.stack([rows[i] for i in range(448)])
    results = {}
    for name, path in (("control", BASE), ("ankle2x", OUT / "ankle2x")):
        current = read(path / "resolved_options.json")
        assert all(current[k] == options[k] for k in options if k != "output_dir")
        if name != "control":
            assert current["body_ankle_reprojection_multiplier"] == 2
            with np.load(BASE / "cold_initialization.npz") as a, np.load(path / "cold_initialization.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        with np.load(path / "result.npz", allow_pickle=False) as archive:
            joints = archive["predicted_coco"].astype(float)
            np.testing.assert_array_equal(archive["pair_id"], np.arange(448))
            masks = {s: archive["body_" + s + "_2d_valid"] for s in raw}
        ankles, other, views = [], [], {}
        for side in raw:
            points = joints if side == "left" else joints @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000
            pixels = fisheye_project_numpy(points, cal.K0 if side == "left" else cal.K1,
                                          cal.D0 if side == "left" else cal.D1)
            error = np.linalg.norm(pixels - raw[side][..., :2], axis=-1)
            assert masks[side][:, 11:17].all()
            ankles.extend(error[:, 15:17].ravel())
            other.extend(error[:, :15][masks[side][:, :15]])
            ratio = amplitude(pixels[:, 15] - pixels[:, 16]) / amplitude(raw[side][:, 15, :2] - raw[side][:, 16, :2])
            phases = {}
            for joint in (15, 16):
                x = pixels[:, joint] - pixels[:, 11:13].mean(1)
                y = raw[side][:, joint, :2] - raw[side][:, 11:13, :2].mean(1)
                index = np.arange(12, 436)
                curve = [{"k": k, "correlation": correlation(x[index + k], y[index])} for k in range(-10, 11)]
                phases[str(joint)] = {"best": max(curve, key=lambda v: v["correlation"]), "zero": curve[10]}
            views[side] = {"amplitude_ratio": ratio, "phases": phases,
                           "frame391_ankle_px": error[391, 15:17].tolist(),
                           "all448_ankle_px": error[:, 15:17].tolist()}
        relative = joints[:, 15:17] - joints[:, 11:13].mean(1)[:, None]
        results[name] = {"ankle_px": metric(ankles), "other_px": metric(other), "views": views,
                         "d2_mm_per_frame2": metric(np.linalg.norm(np.diff(relative, n=2, axis=0), axis=-1) * 1000)}
    a, b = results["control"], results["ankle2x"]
    gates = {"ankle_mean_improves15pct": b["ankle_px"]["mean"] <= .85 * a["ankle_px"]["mean"],
             "ankle_p95_improves15pct": b["ankle_px"]["p95"] <= .85 * a["ankle_px"]["p95"],
             "other_mean_no_more10pct_worse": b["other_px"]["mean"] <= 1.1 * a["other_px"]["mean"],
             "amplitude_closer_both_views": all(abs(b["views"][s]["amplitude_ratio"] - 1) < abs(a["views"][s]["amplitude_ratio"] - 1) for s in raw),
             "d2_no_more20pct_worse": b["d2_mm_per_frame2"]["p95"] <= 1.2 * a["d2_mm_per_frame2"]["p95"]}
    report = {"scope": "same-source2D consistency, not gait or physical accuracy",
              "results": results, "gates": gates, "passes": all(gates.values()),
              "same_cold_initialization": True, "same_old_options": True}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(name, {k: result[k] for k in ("ankle_px", "other_px", "d2_mm_per_frame2")})
        print("views", {s: {k: v for k, v in row.items() if k != "all448_ankle_px"} for s, row in result["views"].items()})
    print("gates", gates)


if __name__ == "__main__":
    main()
