"""Audit cached 448-frame 2D fitting; no model decoding or weight loading."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import fisheye_project_numpy, fisheye_project_torch, load_stereo_fisheye
from pose_app.smpl_coco_observation import load_coco17_regressor, regress_coco17_numpy


ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "research_records/engineering_validation/G20261012_smplh_2d448_view_v1"
SCENE = ROOT / "research_records/engineering_validation/G20261012_supported_stop448_v1/view_v4/scene"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {"count": int(values.size), "mean": float(values.mean()),
            "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def main():
    options = read(FIT / "fit/resolved_options.json")
    summary = read(FIT / "fit/fit_summary.json")
    with np.load(FIT / "fit/result.npz", allow_pickle=False) as archive:
        fitted = {k: archive[k] for k in (
            "pair_id", "vertices", "predicted_coco", "smplh_joints",
            "raw_triangulated_points", "body_accepted", "body_left_2d_valid",
            "body_right_2d_valid")}
    with np.load(SCENE / "scene_transforms.npz", allow_pickle=True) as archive:
        scene = {k: archive[k] for k in (
            "rotation_ground_from_left", "translation_ground_from_left_mm", "stage")}
    np.testing.assert_array_equal(fitted["pair_id"], np.arange(448))
    cal = load_stereo_fisheye(ROOT / options["calibration_dir"])
    regressor = load_coco17_regressor(ROOT / options["regressor"])
    joints = fitted["predicted_coco"]
    reconstructed = regress_coco17_numpy(fitted["vertices"], regressor)
    np.testing.assert_allclose(reconstructed, joints, atol=2e-7, rtol=0)
    raw, errors, dlt_errors = {}, {}, {}
    report = {"scope": "read-only full448 cached audit; no neural weights or fitting",
              "inputs": {"fit": str(FIT / "fit/result.npz"), "scene": str(SCENE),
                         "regressor": options["regressor"],
                         "left_raw": options["left_raw"], "right_raw": options["right_raw"]},
              "regressor_max_saved_difference_m": float(np.abs(reconstructed - joints).max()),
              "per_view_joint_pixels": {}, "free_point_counterexamples": []}

    def project(points, side):
        if side == "right":
            points = points @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000
        return fisheye_project_numpy(points, getattr(cal, "K0" if side == "left" else "K1"),
                                     getattr(cal, "D0" if side == "left" else "D1"))

    for side in ("left", "right"):
        rows = raw_side(Path(options[side + "_raw"]), side)
        assert sorted(rows) == list(range(448))
        raw[side] = np.stack([rows[i] for i in range(448)])
        valid = fitted["body_" + side + "_2d_valid"]
        expected = (np.isfinite(raw[side]).all(-1) & (raw[side][..., 2] > 0)
                    & (raw[side][..., 0] >= 0) & (raw[side][..., 0] < cal.image_width)
                    & (raw[side][..., 1] >= 0) & (raw[side][..., 1] < cal.image_height))
        np.testing.assert_array_equal(valid, expected)
        errors[side] = np.linalg.norm(project(joints, side) - raw[side][..., :2], axis=-1)
        camera_points = (joints if side == "left" else
                         joints @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000)
        torch_pixels = fisheye_project_torch(
            torch.tensor(camera_points, dtype=torch.float64),
            getattr(cal, "K0" if side == "left" else "K1"),
            getattr(cal, "D0" if side == "left" else "D1")).numpy()
        np.testing.assert_allclose(torch_pixels, project(joints, side), atol=1e-9, rtol=0)
        dlt_errors[side] = np.linalg.norm(
            project(fitted["raw_triangulated_points"] / 1000, side) - raw[side][..., :2], axis=-1)
        report["per_view_joint_pixels"][side] = {
            str(j): {"fit_px": stats(errors[side][valid[:, j], j]),
                     "confidence": stats(raw[side][valid[:, j], j, 2]),
                     "dlt_px_diagnostic": stats(dlt_errors[side][valid[:, j], j]),
                     "fraction_above_huber25px": float((errors[side][valid[:, j], j] > 25).mean())}
            for j in range(17)}
    report["projection_numpy_torch_float64_agree"] = True
    report["source_channel_count"] = {
        side: len(read(options[side + "_raw"])["images"][0]["keypoints"][0])
        for side in ("left", "right")}
    report["fitting_channel_count"] = 17
    # Free 3D point is only a numerical consistency comparator for the 2D
    # target pair. It is not a body-fit candidate or independent 3D truth.
    for frame in (45, 391):
        for joint in (15, 16):
            def objective(point):
                result = []
                for side in ("left", "right"):
                    scaled = np.linalg.norm(project(point, side) - raw[side][frame, joint, :2]) / 25
                    huber = 0.5 * scaled**2 if scaled <= 1 else scaled - 0.5
                    result.append(float(np.clip(raw[side][frame, joint, 2], 0, 1)) * huber)
                return sum(result)
            point = fitted["raw_triangulated_points"][frame, joint] / 1000
            result = minimize(objective, point, method="BFGS", options={"gtol": 1e-7})
            report["free_point_counterexamples"].append({
                "frame": frame, "joint": joint, "optimizer_success": bool(result.success),
                "message": str(result.message), "fit_objective": objective(joints[frame, joint]),
                "free_objective": float(result.fun),
                "fit_px_left_right": [float(errors[s][frame, joint]) for s in ("left", "right")],
                "free_px_left_right": [float(np.linalg.norm(project(result.x, s)
                    - raw[s][frame, joint, :2])) for s in ("left", "right")]})
    rotation = scene["rotation_ground_from_left"]
    translation = scene["translation_ground_from_left_mm"]

    def ground(points):
        return np.einsum("nij,nkj->nki", rotation, points) + translation[:, None]

    model_ground = ground(joints * 1000)
    dlt_ground = ground(fitted["raw_triangulated_points"])
    report["ankles"] = {}
    for j, internal in ((15, 7), (16, 8)):
        diff_camera = joints[:, j] * 1000 - fitted["raw_triangulated_points"][:, j]
        diff_ground = model_ground[:, j] - dlt_ground[:, j]
        mask = fitted["body_accepted"][:, j]
        np.testing.assert_allclose(np.linalg.norm(diff_camera[mask], axis=-1),
                                   np.linalg.norm(diff_ground[mask], axis=-1), atol=1e-9)
        report["ankles"][str(j)] = {
            "camera_distance_to_dlt_mm_diagnostic": stats(np.linalg.norm(diff_camera[mask], axis=-1)),
            "frame391_ground_delta_xyz_mm": diff_ground[391].tolist(),
            "coco_to_internal_ankle_mm": stats(np.linalg.norm(
                joints[:, j] - fitted["smplh_joints"][:, internal], axis=-1) * 1000),
            "regressor_vertices": int((regressor[j] > 0).sum())}
    sets = read(ROOT / "research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json")["sets"]
    with np.load(FIT / "smoothed/result.npz", allow_pickle=False) as archive:
        smoothed = archive["vertices"]
    report["sole_reference_heights_mm"] = {}
    for side in ("left", "right"):
        indices = np.unique(sum(sets[side + "_sole_surface_candidate"].values(), []))
        height = ground(fitted["vertices"][:, indices] * 1000)[..., 2].min(axis=1)
        smooth_height = ground(smoothed[:, indices] * 1000)[..., 2].min(axis=1)
        report["sole_reference_heights_mm"][side] = {
            "raw": stats(height), "frames_below_minus20mm": int((height < -20).sum()),
            "frame391": float(height[391]),
            "max_smoothing_change_mm": float(np.abs(smooth_height - height).max()),
            "all448_raw": height.tolist(), "all448_smoothed": smooth_height.tolist()}
    report["all448_ankle_pixel_errors"] = {
        side: errors[side][:, 15:17].tolist() for side in ("left", "right")}
    report["optimization_evidence"] = {
        "late_C_history": [r for r in summary["history"]
                           if r["stage"] == "C_body_vposer_refine" and r["stage_step"] >= 240],
        "gradient_audit": summary["body_reprojection"]["gradient_audit"],
        "stage_schedule": summary["stage_schedule"],
        "active_body_terms": {k: options[k] for k in (
            "body_3d_weight", "body_reprojection_weight", "body_reprojection_scale_px",
            "vposer_prior_weight", "root_anchor_weight", "beta_prior_weight",
            "body_temporal_weight", "bone_weight", "surface_foot_contact_weight")}}
    report["limits"] = "DLT agreement and reference-ground sole heights are diagnostics, not independent accuracy/contact truth. No causal prior ablation performed."
    output = FIT / "foot_logic_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ankles", "free_point_counterexamples")}, indent=2))


if __name__ == "__main__":
    main()
