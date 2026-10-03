"""Raw-unit, fixed-input diagnostics for SMPL-H ablation attribution."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0, "rms": None, "median": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite diagnostic; retain invalid mask separately")
    return {"count": int(values.size), "rms": float(np.sqrt(np.mean(values ** 2))),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def leg_diagnostics(joints, fps):
    if fps <= 0 or joints.shape[0] < 3:
        raise ValueError("positive fps and at least three frames required")
    hip = joints[:, [1, 2]].mean(axis=1)
    relative = joints[:, [4, 5, 7, 8]] - hip[:, None]
    accel = np.linalg.norm(np.diff(relative, n=2, axis=0), axis=-1) * fps ** 2
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1) * fps
    excursion = np.linalg.norm(relative - relative.mean(axis=0, keepdims=True), axis=-1) * 1000
    knees = []
    for h, k, a in ((1, 4, 7), (2, 5, 8)):
        u, v = joints[:, h] - joints[:, k], joints[:, a] - joints[:, k]
        denominator = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        if np.any(denominator <= 1e-12):
            raise ValueError("degenerate knee geometry")
        # Flexion is 0 at a straight knee. This is a model-derived angle.
        knees.append(180 - np.rad2deg(np.arccos(np.clip((u * v).sum(-1) / denominator, -1, 1))))
    knees = np.stack(knees, axis=1)
    return {"leg_relative_acceleration_m_s2": stats(accel), "leg_relative_speed_m_s": stats(speed),
            "leg_motion_excursion_mm": stats(excursion),
            "knee_flexion_deg": {side: stats(knees[:, index]) for index, side in enumerate(("left", "right"))},
            "knee_range_deg": (knees.max(axis=0) - knees.min(axis=0)).tolist(),
            "knee_frame_change_deg": stats(np.abs(np.diff(knees, axis=0)))}, {
            "leg_relative_acceleration_m_s2": accel, "leg_relative_speed_m_s": speed,
            "leg_relative_points_m": relative, "knee_flexion_deg": knees}


def evaluate(result, observations, wrist_target, scene, soles, fps, native_reference):
    from pose_app.fisheye_camera import fisheye_project_numpy

    if not np.array_equal(result["pair_id"], observations["pair_id"]):
        raise ValueError("frame contract differs")
    expected_mask = observations["accepted"] & np.isfinite(observations["triangulated_mm"]).all(-1)
    if not np.array_equal(result["body_accepted"], expected_mask):
        raise ValueError("ablation changed accepted mask")
    if not np.array_equal(result["body_quality"], observations["quality"]):
        raise ValueError("ablation changed stored observation quality")
    if not np.array_equal(result["raw_triangulated_points"], observations["triangulated_mm"], equal_nan=True):
        raise ValueError("ablation changed triangulated input")
    residual = np.linalg.norm(result["predicted_coco"] - observations["triangulated_mm"] / 1000, axis=-1) * 1000
    q = observations["quality"][expected_mask]
    metrics = {"body_observation_mm": stats(residual[expected_mask]),
               "reference_quality_weighted_body_rms_mm": float(np.sqrt(np.sum(q * residual[expected_mask] ** 2) / q.sum())),
               "body_groups_mm": {}, "betas": result["betas"].tolist(), "beta_norm": float(np.linalg.norm(result["betas"]))}
    arrays = {"pair_id": result["pair_id"], "accepted": expected_mask, "body_residual_mm": residual}
    for group, indices in {"torso": [5, 6, 11, 12], "arms": [7, 8, 9, 10], "legs": [13, 14, 15, 16]}.items():
        metrics["body_groups_mm"][group] = stats(residual[:, indices][expected_mask[:, indices]])
    wrist_error = np.linalg.norm(result["smplh_joints"][:, [20, 21]] - wrist_target, axis=-1) * 1000
    metrics["wrist_reference_mm"] = {side: stats(wrist_error[:, index]) for index, side in enumerate(("left", "right"))}
    arrays["wrist_reference_mm"] = wrist_error
    temporal, temporal_arrays = leg_diagnostics(result["smplh_joints"], fps)
    metrics.update(temporal)
    arrays.update(temporal_arrays)
    projection_errors = []
    for index, side in enumerate(("left", "right")):
        points = result["predicted_coco"]
        if index:
            points = points @ observations["R_cam0_to_cam1"].T + observations["T_cam0_to_cam1_mm"] / 1000
        projected = fisheye_project_numpy(points.reshape(-1, 3), observations[f"K{index}"], observations[f"D{index}"]).reshape(points.shape[:2] + (2,))
        raw = observations[f"{side}_raw_pixels"]
        residual_px = np.linalg.norm(projected - raw[:, :, :2], axis=-1)
        usable = expected_mask & np.isfinite(residual_px)
        metrics[f"{side}_projection_invalid_accepted_points"] = int((expected_mask & ~np.isfinite(residual_px)).sum())
        metrics[f"{side}_negative_depth_accepted_points"] = int((expected_mask & (points[:, :, 2] <= 0)).sum())
        projection_errors.append(residual_px[usable])
        arrays[f"{side}_reprojection_px"] = residual_px
        arrays[f"{side}_reprojection_valid"] = usable
    metrics["body_reprojection_px"] = stats(np.concatenate(projection_errors))
    vg = np.einsum("nij,nvj->nvi", scene["rotation_ground_from_left"], result["vertices"]) + scene["translation_ground_from_left_mm"][:, None] / 1000
    metrics["feet"] = {}
    metrics["scene_accepted_frames"] = int(scene["accepted"].sum())
    for side, indices in soles.items():
        points = vg[:, indices]
        minimum = points[:, :, 2].min(axis=1) * 1000
        center = points.mean(axis=1)
        metrics["feet"][side] = {"minimum_z_mm": float(minimum.min()), "median_frame_min_z_mm": float(np.median(minimum)),
            "frames_min_z_below_minus5mm": int((minimum < -5).sum()),
            "sole_center_speed_m_s": stats(np.linalg.norm(np.diff(center, axis=0), axis=-1) * fps)}
        arrays[f"{side}_sole_min_z_mm"] = minimum
        arrays[f"{side}_sole_center_ground_m"] = center
    metrics["native_mano"] = {}
    for side in ("left", "right"):
        # Always evaluate the same baseline-associated native hypotheses, even
        # when the variant disabled their loss or initialization.
        poses = result[f"{side}_hand_pose"].reshape(-1, 15, 3)
        rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(len(poses), 15, 3, 3)
        targets = native_reference[f"mano_{side}_target_rotations"]
        weights = native_reference[f"mano_{side}_view_weights"]
        cos_angle = ((rotations[:, None] * targets).sum(axis=(-1, -2)) - 1) * .5
        angles = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
        valid = weights > 0
        metrics["native_mano"][side] = {"rotation_error_deg": stats(angles[valid]),
            "reference_chordal": float(((1 - np.clip(cos_angle, -1, 1)).mean(axis=-1) * weights).sum() / weights.sum()),
            "supported_view_frames": int(valid.sum()),
            "coefficients_rms_norm": float(np.sqrt(np.mean(np.sum(result[f"{side}_hand_pca"] ** 2, axis=1))))}
        relative_trace = (rotations[1:] * rotations[:-1]).sum(axis=(-1, -2))
        changes = np.rad2deg(np.arccos(np.clip((relative_trace - 1) * .5, -1, 1)))
        metrics["native_mano"][side]["finger_frame_rotation_change_deg"] = stats(changes)
        arrays[f"{side}_finger_frame_rotation_change_deg"] = changes
    return metrics, arrays
