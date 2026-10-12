"""Read-only phase diagnostic for current raw/smoothed448 ankle projections."""
import json
from pathlib import Path

import numpy as np

from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research_records/engineering_validation/G20261012_ankle_simple_ablation448_v1"


def correlation(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denominator == 0:
        raise ValueError("constant motion cannot identify a phase shift")
    return float(np.sum(a * b) / denominator)


def main():
    options = json.loads((RUN / "weight2x/resolved_options.json").read_text(encoding="utf-8"))
    cal = load_stereo_fisheye(ROOT / options["calibration_dir"])
    raw = {}
    for side in ("left", "right"):
        rows = raw_side(Path(options[side + "_raw"]), side)
        assert sorted(rows) == list(range(448))
        raw[side] = np.stack([rows[i] for i in range(448)])
    # Fixed identical comparison frames for every lag, preserving raw edges
    # in the source files. This is not an instruction to shift the viewer.
    index = np.arange(12, 436)
    report = {"scope": "raw2D vs model-projected pelvis-relative ankles; no true gait timing",
              "inputs": {"run": str(RUN), "raw_pixels": {s: options[s + "_raw"] for s in raw}},
              "phase_comparison_frames": "12..435,424 identical frames for every tested lag",
              "lag_definition": "compare model[t+k] to observation[t]; positive k means model later in this diagnostic",
              "search_frames": [-10, 10], "variants": {},
              "body_temporal_weight": options["body_temporal_weight"],
              "note": "Correlation optimum is a phase-consistency diagnostic, not measured latency or evidence for an index offset."}
    for variant in ("weight2x", "smoothed"):
        with np.load(RUN / variant / "result.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["pair_id"], np.arange(448))
            joints = archive["predicted_coco"].astype(float)
            triangulated = archive["raw_triangulated_points"] / 1000
            for side in raw:
                assert archive["body_" + side + "_2d_valid"][:, 11:17].all()
        measurements = {}
        for side in raw:
            points = joints if side == "left" else (
                joints @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000)
            projected = fisheye_project_numpy(points,
                cal.K0 if side == "left" else cal.K1,
                cal.D0 if side == "left" else cal.D1)
            if variant == "weight2x":
                tri_points = triangulated if side == "left" else (
                    triangulated @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000)
                tri_pixels = fisheye_project_numpy(tri_points,
                    cal.K0 if side == "left" else cal.K1,
                    cal.D0 if side == "left" else cal.D1)
                examples = report.setdefault("frame391", {})
                for joint in (15, 16):
                    examples.setdefault(str(joint), {})[side] = {
                        "model_error_px": float(np.linalg.norm(
                            projected[391, joint] - raw[side][391, joint, :2])),
                        "dlt_error_px_diagnostic": float(np.linalg.norm(
                            tri_pixels[391, joint] - raw[side][391, joint, :2]))}
            for joint in (15, 16):
                model = projected[:, joint] - projected[:, 11:13].mean(axis=1)
                observed = raw[side][:, joint, :2] - raw[side][:, 11:13, :2].mean(axis=1)
                curve = [{"k": k, "correlation": correlation(model[index + k], observed[index]),
                          "mean_vector_error_px": float(np.linalg.norm(
                              model[index + k] - observed[index], axis=-1).mean())}
                         for k in range(-10, 11)]
                zero = curve[10]
                best = max(curve, key=lambda row: row["correlation"])
                measurements[f"{side}_{joint}"] = {"zero": zero, "best": best,
                                                    "curve": curve}
        report["variants"][variant] = measurements
    output = RUN / "ankle_phase_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    for variant, measurements in report["variants"].items():
        print(variant, {key: value["best"]["k"] for key, value in measurements.items()})


if __name__ == "__main__":
    main()
