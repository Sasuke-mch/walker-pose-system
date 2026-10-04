"""Small unreleased SE(3) window; strict observations and protocol priors stay separate."""

from itertools import pairwise

import cv2
import numpy as np
from scipy.optimize import least_squares

from .dynamic_ground_live import STAGE1
from .two_stage_motion import WorldPose


def rotation_vector(rotation):
    return cv2.Rodrigues(rotation)[0].ravel()


def solve_stage_window(entries, initial, *, support=False):
    """Return a gated proposal without mutating input rows or their original anchors."""
    if len(entries) < 3:
        return None, {"status": "unavailable", "reason": "window_too_short"}
    if any(
        row["pose_audit"]["window_observation"]["ground_stage"] != STAGE1
        and not row["pose_audit"]["window_observation"].get("pose_update_active", True)
        for row, _ in entries
    ):
        return None, {
            "status": "unavailable",
            "reason": "window_contains_inactive_transition",
        }
    times = np.array([p["pair_timestamp_sec"] for _, p in entries], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("window timestamps must increase")
    old = [
        WorldPose(
            np.asarray(r["rotation_world_from_left_camera"]),
            np.asarray(r["translation_world_from_left_camera_mm"]),
        )
        for r, _ in entries
    ]
    reference_poses = []
    for (row, _), pose in zip(entries, old):
        previous = row["pose_audit"].get("measurement_pose_before_support")
        reference_poses.append(
            WorldPose(
                np.asarray(previous["rotation_world_from_left_camera"]),
                np.asarray(previous["translation_world_from_left_camera_mm"]),
            )
            if previous
            else pose
        )
        R = pose.rotation_world_from_camera
        if (
            R.shape != (3, 3)
            or not np.isfinite(R).all()
            or not np.isfinite(pose.translation_world_from_camera_mm).all()
            or np.linalg.norm(R.T @ R - np.eye(3)) > 1e-6
            or abs(np.linalg.det(R) - 1) > 1e-6
        ):
            raise ValueError("window requires finite proper SE3 input")
    groups = [0]
    for row, _ in entries[1:]:
        # A stop is a pose hold, including the moving->stopped boundary.
        groups.append(
            groups[-1]
            if row["pose_audit"]["window_observation"]["ground_stage"] == STAGE1
            else groups[-1] + 1
        )
    count = max(groups)
    if not count:
        return None, {"status": "unavailable", "reason": "no_unreleased_motion"}
    bases = [old[groups.index(g)] for g in range(count + 1)]
    feet, edges = [], []
    for i, (row, payload) in enumerate(entries):
        observation = row["pose_audit"]["window_observation"]
        anchors = row.get("operational_ankle_anchors_mm") or {}
        rejection = row["pose_audit"].get(
            "measurement_reason", row["pose_audit"].get("reason")
        )
        if observation["ground_stage"] != STAGE1 and rejection not in (
            "foot_separation_inconsistent",
            "two_foot_translation_disagreement",
        ):
            for person in payload.get("persons_3d", []):
                for point in person["keypoints_3d"]:
                    side = {15: "left", 16: "right"}.get(point["index"])
                    if side in anchors and point["valid"]:
                        feet.append(
                            (
                                i,
                                np.asarray(point["xyz"], float),
                                np.asarray(anchors[side], float),
                            )
                        )
        edge = observation.get("rotation_to_from")
        if i and edge is not None:
            edges.append((i, np.asarray(edge, float)))
    if len(feet) < 2 or not edges:
        return None, {
            "status": "unavailable",
            "reason": "insufficient_strict_feet_or_rotation",
        }

    def decode(x):
        corrections = np.vstack((np.zeros(6), x.reshape(count, 6)))
        poses = [
            WorldPose(
                cv2.Rodrigues(c[:3])[0] @ b.rotation_world_from_camera,
                b.translation_world_from_camera_mm + c[3:],
            )
            for b, c in zip(bases, corrections)
        ]
        return [poses[g] for g in groups]

    def residual(x):
        poses = decode(x)
        result = []
        for i, point, anchor in feet:
            result.extend((poses[i].transform(point) - anchor) / 20)
        for i, edge in edges:
            predicted = poses[i - 1].rotation_world_from_camera @ edge.T
            result.extend(
                rotation_vector(poses[i].rotation_world_from_camera @ predicted.T)
                / np.radians(0.5)
            )
        for i in range(1, len(poses) - 1):
            dt0, dt1 = times[i] - times[i - 1], times[i + 1] - times[i]
            v0 = (
                poses[i].translation_world_from_camera_mm
                - poses[i - 1].translation_world_from_camera_mm
            ) / dt0
            v1 = (
                poses[i + 1].translation_world_from_camera_mm
                - poses[i].translation_world_from_camera_mm
            ) / dt1
            result.extend((v1 - v0) / ((dt0 + dt1) * 0.5) / 1000)
            w0 = (
                rotation_vector(
                    poses[i].rotation_world_from_camera
                    @ poses[i - 1].rotation_world_from_camera.T
                )
                / dt0
            )
            w1 = (
                rotation_vector(
                    poses[i + 1].rotation_world_from_camera
                    @ poses[i].rotation_world_from_camera.T
                )
                / dt1
            )
            result.extend((w1 - w0) / ((dt0 + dt1) * 0.5) / np.radians(100))
        correction = x.reshape(count, 6)
        result.extend((correction[:, :3] / np.radians(10)).ravel())
        result.extend((correction[:, 3:] / 50).ravel())
        if support:
            last = poses[-1]
            result.append(
                (
                    last.translation_world_from_camera_mm[2]
                    - initial.translation_world_from_camera_mm[2]
                )
                / 2
            )
            # The ground normal constrains tilt; XY and yaw are not fixed.
            result.extend(
                (
                    last.rotation_world_from_camera[2]
                    - initial.rotation_world_from_camera[2]
                )
                / np.radians(0.2)
            )
        return np.asarray(result)

    x0 = np.zeros(count * 6)
    before = residual(x0)
    solution = least_squares(residual, x0, max_nfev=20, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    poses = decode(solution.x)
    before_feet = [
        np.linalg.norm(reference_poses[i].transform(p) - a) for i, p, a in feet
    ]
    after_feet = [np.linalg.norm(poses[i].transform(p) - a) for i, p, a in feet]
    steps = [
        np.linalg.norm(
            b.translation_world_from_camera_mm - a.translation_world_from_camera_mm
        )
        for a, b in pairwise(poses)
    ]
    angles = [
        np.degrees(
            np.linalg.norm(
                rotation_vector(
                    b.rotation_world_from_camera @ a.rotation_world_from_camera.T
                )
            )
        )
        for a, b in pairwise(poses)
    ]
    disagreement = []
    for i in range(len(poses)):
        errors = [poses[i].transform(p) - a for k, p, a in feet if k == i]
        if len(errors) == 2:
            disagreement.append(np.linalg.norm(errors[0] - errors[1]))
    height = abs(
        poses[-1].translation_world_from_camera_mm[2]
        - initial.translation_world_from_camera_mm[2]
    )
    tilt = np.degrees(
        np.arccos(
            np.clip(
                poses[-1].rotation_world_from_camera[2]
                @ initial.rotation_world_from_camera[2],
                -1,
                1,
            )
        )
    )
    audit = {
        "status": "rejected",
        "support_requested": support,
        "frames": len(entries),
        "strict_foot_factors": len(feet),
        "rotation_factors": len(edges),
        "cost_before": float(before @ before),
        "cost_after": float(solution.fun @ solution.fun),
        "solver_success": bool(solution.success),
        "function_evaluations": int(solution.nfev),
        "foot_p95_before_mm": float(np.percentile(before_feet, 95)),
        "foot_p95_after_mm": float(np.percentile(after_feet, 95)),
        "maximum_step_mm": float(max(steps)),
        "maximum_rotation_step_deg": float(max(angles)),
        "support_height_residual_mm": float(height),
        "support_tilt_residual_deg": float(tilt),
    }
    reason = None
    if not solution.success or not np.isfinite(solution.x).all():
        reason = "solver_failed_or_budget_exhausted"
    elif audit["cost_after"] >= audit["cost_before"] - 1e-8:
        reason = "objective_not_decreased"
    elif max(steps) > 120 or max(angles) > 5 or max(disagreement, default=0) > 60:
        reason = "original_step_or_two_foot_gate_failed"
    elif audit["foot_p95_after_mm"] > audit["foot_p95_before_mm"] + 2:
        reason = "foot_anchor_residual_worsened"
    elif support and (height > 5 or tilt > 0.5):
        reason = "support_and_foot_constraints_incompatible"
    audit["reason"] = reason
    if reason:
        return None, audit
    audit["status"] = "accepted"
    return poses, audit
