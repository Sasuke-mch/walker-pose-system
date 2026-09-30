"""WiLoR local joint rotations -> SMPL-H soft pose information.

No model camera translation, global orientation, MANO shape or derived 2-D
keypoint is a metric observation here. Association uses the detector box and
the existing same-camera body wrist. Both camera hypotheses stay separate.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

CONVENTION = "wilor_canonical_right_rotmat_v1"
MIRROR = np.diag([-1., 1., 1.])


def audit_assets(smplh, mano_left, mano_right, canonical_right):
    """Validate chain order and the exact left/right rest-base mirror.

    Shape/hand lengths can differ between SMPL-H and MANO. Do not align
    arbitrary posed meshes and mistake that alignment for pose correctness.
    """
    expected = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    parents = np.asarray(smplh["kintree_table"])[0]
    if np.asarray(smplh["weights"]).shape != (6890, 52):
        raise ValueError("parameter_transfer_requires_52_joint_SMPLH")
    for data in (mano_left, mano_right, canonical_right):
        if not np.array_equal(np.asarray(data["kintree_table"])[0, 1:], expected):
            raise ValueError("MANO_finger_chain_mismatch")
    for start, wrist in ((22, 20), (37, 21)):
        remapped = np.where(expected == 0, wrist, expected + start - 1)
        if not np.array_equal(parents[start:start+15], remapped):
            raise ValueError("SMPLH_finger_chain_mismatch")
    if not np.allclose(np.asarray(canonical_right["v_template"]),
                       np.asarray(mano_right["v_template"]), atol=1e-6):
        raise ValueError("WiLoR_and_fitter_right_MANO_asset_mismatch")
    left_j = mano_left["J_regressor"] @ np.asarray(mano_left["v_template"])
    right_j = mano_right["J_regressor"] @ np.asarray(mano_right["v_template"])
    mirror_error = float(np.max(np.abs((right_j-right_j[0]) @ MIRROR - (left_j-left_j[0]))))
    if mirror_error > 1e-6:
        raise ValueError("MANO_left_right_rest_basis_not_mirrored")
    rest_angles = {}
    sj = np.asarray(smplh["J"])
    for side, start, joints in (("left", 22, left_j), ("right", 37, right_j)):
        mano_bones = joints[1:] - joints[expected]
        body_bones = sj[start:start+15] - sj[parents[start:start+15]]
        norms = np.linalg.norm(mano_bones, axis=-1) * np.linalg.norm(body_bones, axis=-1)
        if not np.isfinite(norms).all() or (norms <= 1e-12).any():
            raise ValueError("degenerate_rest_finger_bone")
        angles = np.degrees(np.arccos(np.clip((mano_bones*body_bones).sum(-1)/norms, -1, 1)))
        # Gross template axis/order mismatch gate, not an accuracy threshold.
        if angles.max() > 45:
            raise ValueError("MANO_SMPLH_rest_finger_axis_mismatch")
        rest_angles[side] = angles.tolist()
    return {"finger_chain_order": ["index", "middle", "pinky", "ring", "thumb"],
            "left_right_rest_mirror_max_m": mirror_error,
            "MANO_SMPLH_rest_bone_angle_deg": rest_angles,
            "rest_bone_angle_gate_deg": 45,
            "canonical_right_asset_matches": True,
            "global_orientation_translation_shape_transferred": False}


def validate_rotations(value, count: int) -> np.ndarray:
    r = np.asarray(value, dtype=np.float64)
    if r.shape != (count, 3, 3) or not np.isfinite(r).all():
        raise ValueError("invalid_rotation_shape_or_nonfinite")
    if (np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3))) > 1e-4
            or np.max(np.abs(np.linalg.det(r) - 1)) > 1e-4):
        raise ValueError("rotation_not_SO3")
    return r


def convert_record(record: dict) -> np.ndarray:
    if record.get("mano_parameter_convention") != CONVENTION:
        raise ValueError("missing_or_unknown_mano_parameter_convention")
    if record.get("side") not in ("left", "right"):
        raise ValueError("invalid_anatomical_side")
    params = record.get("mano_parameters", {})
    metadata = record.get("mano_parameter_metadata", {})
    if (metadata.get("joint_order") != ["index", "middle", "pinky", "ring", "thumb"]
            or metadata.get("rotation_type") != "parent_relative_rotmat"
            or metadata.get("pose_mean_added_by_layer") is not False
            or metadata.get("left_crop_mirrored") != (record["side"] == "left")):
        raise ValueError("inconsistent_mano_parameter_metadata")
    r = validate_rotations(params.get("hand_pose"), 15)
    validate_rotations(params.get("global_orient"), 1)
    b = np.asarray(params.get("betas"), dtype=np.float64)
    if b.shape != (10,) or not np.isfinite(b).all():
        raise ValueError("invalid_mano_shape")
    # Native network rotations are canonical RIGHT even for flipped LEFT crops.
    # Mirror the local bases as well as the geometry; never use M @ R alone.
    return MIRROR @ r @ MIRROR if record["side"] == "left" else r


def read_parameter_view(path: Path, body: np.ndarray, image_size: tuple[int, int]):
    """N x 2 x 15 rotations; rejected candidates remain in the audit.

    Association is independent of WiLoR's pinhole-projected 21 keypoints.
    Detector confidence is only box confidence, not joint confidence.
    """
    n = len(body)
    rotations = np.broadcast_to(np.eye(3), (n, 2, 15, 3, 3)).copy()
    weights = np.zeros((n, 2), dtype=np.float64)
    audit, seen = [], set()
    width, height = image_size
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame = int(row["frame_index"])
            if frame not in range(n) or frame in seen:
                raise ValueError("duplicate_or_out_of_range_parameter_frame")
            if Path(row["image"]).stem != f"pair_{frame:04d}":
                raise ValueError("parameter_image_frame_mismatch")
            seen.add(frame)
            for h, side in enumerate(("left", "right")):
                wrist = np.asarray(body[frame, 9 + h])
                wrist_ok = (np.isfinite(wrist).all() and wrist[2] >= .25
                            and 0 <= wrist[0] < width and 0 <= wrist[1] < height)
                entries = []
                for ordinal, c in enumerate(row["records"]):
                    if c.get("side") != side:
                        continue
                    rec = {"frame_index": frame, "hand": side, "record_index": ordinal,
                           "selected": False, "accepted": False,
                           "confidence_source": "detector_box_not_joint"}
                    try:
                        box_key = "detector_bbox_xyxy_raw" if "detector_bbox_xyxy_raw" in c else "bbox_xyxy"
                        box = np.asarray(c.get(box_key), dtype=float)
                        rec["association_box_source"] = box_key
                        if (c.get("pixel_frame") != "raw_fisheye" or box.shape != (4,)
                                or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])):
                            raise ValueError("invalid_raw_detector_box")
                        if not wrist_ok:
                            raise ValueError("body_wrist_unavailable")
                        # Expanded crop can extend outside the image: use the box
                        # itself, not its model-projected wrist, for association.
                        distance = float(np.linalg.norm(np.maximum(np.maximum(box[:2]-wrist[:2],
                                                            wrist[:2]-box[2:]), 0)))
                        rec["box_wrist_distance_px"] = distance
                        entries.append((distance, ordinal, c, rec))
                        rec["reason"] = "candidate"
                    except (ValueError, TypeError) as exc:
                        rec["reason"] = str(exc)
                    audit.append(rec)
                entries.sort(key=lambda x: x[0])
                if not entries:
                    audit.append({"frame_index": frame, "hand": side, "accepted": False,
                                  "reason": "no_associable_candidate"})
                    continue
                chosen = None
                reason = "box_wrist_distance"
                if entries[0][0] <= 150:
                    reason = "ambiguous_detector_boxes" if len(entries)>1 and entries[1][0]-entries[0][0] < 25 else "selected"
                    if reason == "selected":
                        chosen = entries[0][1]
                for _, ordinal, c, rec in entries:
                    rec["reason"] = reason if chosen is None else "farther_candidate"
                    if ordinal != chosen:
                        continue
                    rec["selected"] = True
                    try:
                        r = convert_record(c)
                        confidence = float(c.get("detector_confidence", float("nan")))
                        if not np.isfinite(confidence) or not 0 < confidence <= 1:
                            raise ValueError("invalid_or_missing_detector_confidence")
                        rotations[frame, h] = r
                        weights[frame, h] = confidence
                        rec.update(accepted=True, reason="accepted_local_pose_hypothesis",
                                   weight=confidence)
                    except (ValueError, TypeError) as exc:
                        rec["reason"] = str(exc)
    if seen != set(range(n)):
        raise ValueError("parameter_frame_coverage_mismatch")
    return rotations, weights, audit


def encode_pca(rotations, components, mean, scale):
    """Row-vector least squares, matching the fitter's external mean/scale.

    Returns coefficients and per-frame/per-joint rotation reconstruction error.
    """
    shape = rotations.shape[:-2]
    theta = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(*shape[:-1], 45)
    coeff = ((theta - mean) @ np.linalg.pinv(components)) * scale
    reconstructed = mean + (coeff / scale) @ components
    rr = Rotation.from_rotvec(reconstructed.reshape(-1, 3)).as_matrix().reshape(rotations.shape)
    delta = rotations.swapaxes(-1, -2) @ rr
    angles = Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude().reshape(shape)
    return coeff, angles


def rotation_pose_loss(axis_angles, targets, weights):
    """Chordal SO(3) loss = 1-cos(angle), bounded, no acos endpoint gradient.

    weights: N x views; targets: N x views x 15 x 3 x 3.
    """
    import torch
    from smplx.lbs import batch_rodrigues
    r = batch_rodrigues(axis_angles.reshape(-1, 3)).reshape(-1, 15, 3, 3)
    per_joint = (r[:, None] - targets).square().sum(dim=(-1, -2)) / 4.
    numerator = (per_joint.mean(-1) * weights).sum()
    return numerator / weights.sum().clamp_min(1e-8)
