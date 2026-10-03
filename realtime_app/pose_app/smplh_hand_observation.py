"""Named SMPL-H/MANO observation adapter; camera identity is not handedness."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# SMPL-H has 22 body joints INCLUDING root, followed by 15 joints per hand.
# WiLoR's mano_to_openpose output: wrist, thumb, index, middle, ring, pinky;
# each finger has three internal joints followed by one fingertip.
HAND_JOINTS = {
    "left": [20, 34, 35, 36, 22, 23, 24, 25, 26, 27, 31, 32, 33, 28, 29, 30],
    "right": [21, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45],
}
TIP_VERTICES = {"left": [2746, 2319, 2445, 2556, 2673],
                "right": [6191, 5782, 5905, 6016, 6133]}
HAND_NAMES = ["wrist"] + [f"{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in (1, 2, 3, "tip")]

def audit_cross_view_geometry(points_cam0: np.ndarray, points_cam1: np.ndarray,
                              K0: np.ndarray, D0: np.ndarray,
                              K1: np.ndarray, D1: np.ndarray,
                              R01: np.ndarray, T01_mm: np.ndarray,
                              image_size: tuple[int, int]) -> tuple[list[dict], dict]:
    """Audit paired raw-fisheye hand points without changing fitting masks.

    The returned records are diagnostics only.  Each finite paired point is
    undistorted, triangulated in the cam0 frame, reprojected into both raw
    fisheye images, and checked for positive depth and ray-angle degeneracy.
    Rejected points remain in the JSONL with an explicit reason.
    """
    import cv2
    from .fisheye_camera import fisheye_project_numpy
    p0 = np.asarray(points_cam0, dtype=np.float64)
    p1 = np.asarray(points_cam1, dtype=np.float64)
    if p0.shape != p1.shape or p0.ndim != 3 or p0.shape[-2:] != (21, 2):
        raise ValueError(f"expected paired hand points (N,21,2), got {p0.shape}/{p1.shape}")
    R = np.asarray(R01, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T01_mm, dtype=np.float64).reshape(3) / 1000.0
    P0 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    P1 = np.concatenate([R, T[:, None]], axis=1)
    width, height = image_size
    records, counts = [], {"paired_finite": 0, "positive_depth": 0,
                           "reprojection_ok": 0, "accepted": 0,
                           "rejected": 0}
    for frame in range(p0.shape[0]):
        for joint in range(21):
            a, b = p0[frame, joint], p1[frame, joint]
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            rec = {"frame_index": frame, "joint_index": joint,
                   "cam0_point": a.tolist(), "cam1_point": b.tolist(),
                   "finite_pair": finite, "accepted": False}
            if not finite:
                rec["reject_reason"] = "nonfinite_pair"
                records.append(rec); counts["rejected"] += 1; continue
            counts["paired_finite"] += 1
            in_bounds = (0 <= a[0] < width and 0 <= a[1] < height and
                         0 <= b[0] < width and 0 <= b[1] < height)
            rec["both_in_raw_bounds"] = bool(in_bounds)
            u0 = cv2.fisheye.undistortPoints(a.reshape(1, 1, 2), K0, D0).reshape(2)
            u1 = cv2.fisheye.undistortPoints(b.reshape(1, 1, 2), K1, D1).reshape(2)
            x0 = np.array([[u0[0]], [u0[1]]], dtype=np.float64)
            x1 = np.array([[u1[0]], [u1[1]]], dtype=np.float64)
            Xh = cv2.triangulatePoints(P0, P1, x0, x1).reshape(4)
            if abs(float(Xh[3])) < 1e-12:
                rec["reject_reason"] = "triangulation_at_infinity"
                records.append(rec); counts["rejected"] += 1; continue
            X = Xh[:3] / Xh[3]
            X1 = R @ X + T
            rec["point_cam0_m"] = X.tolist()
            rec["point_cam1_m"] = X1.tolist()
            rec["positive_depth"] = bool(X[2] > 0 and X1[2] > 0)
            counts["positive_depth"] += int(rec["positive_depth"])
            ray0 = np.array([u0[0], u0[1], 1.0])
            ray1_in_0 = R.T @ np.array([u1[0], u1[1], 1.0])
            cosang = np.dot(ray0, ray1_in_0) / max(np.linalg.norm(ray0) * np.linalg.norm(ray1_in_0), 1e-12)
            rec["ray_angle_deg"] = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            q0 = fisheye_project_numpy((X * 1000.0).reshape(1, 3), K0, D0)[0]
            q1 = fisheye_project_numpy((X1 * 1000.0).reshape(1, 3), K1, D1)[0]
            e0, e1 = float(np.linalg.norm(q0-a)), float(np.linalg.norm(q1-b))
            rec["reprojected_cam0"] = q0.tolist(); rec["reprojected_cam1"] = q1.tolist()
            rec["reprojection_error_px"] = {"cam0": e0, "cam1": e1, "mean": 0.5*(e0+e1)}
            rec["reprojection_ok"] = bool(np.isfinite(e0+e1) and e0 <= 10.0 and e1 <= 10.0)
            counts["reprojection_ok"] += int(rec["reprojection_ok"])
            rec["accepted"] = bool(in_bounds and rec["positive_depth"] and rec["reprojection_ok"])
            if not rec["accepted"]:
                rec["reject_reason"] = ("raw_bounds" if not in_bounds else
                                         "nonpositive_depth" if not rec["positive_depth"] else
                                         "reprojection_error")
            counts["accepted"] += int(rec["accepted"]); counts["rejected"] += int(not rec["accepted"])
            records.append(rec)
    counts["acceptance_rate"] = (counts["accepted"] / max(counts["paired_finite"], 1))
    return records, counts

def hand21(joints, vertices, side):
    """Differentiable model points in WiLoR OpenPose order, including fingertips."""
    import torch
    internal = joints[:, HAND_JOINTS[side]]
    tips = vertices[:, TIP_VERTICES[side]]
    # MANO internal order is index, middle, pinky, ring, thumb. WiLoR
    # reorders it with [0,13,14,15,16,1,2,3,17,...] to thumb first.
    # HAND_JOINTS and TIP_VERTICES are already in that output order.
    order = range(5)
    return torch.stack(
        [internal[:, 0]]
        + [
            p
            for tip_i, f in enumerate(order)
            for p in (
                internal[:, 1 + 3*f],
                internal[:, 2 + 3*f],
                internal[:, 3 + 3*f],
                tips[:, tip_i],
            )
        ],
        dim=1,
    )

def validate_smplh(model):
    parents = model.parents.detach().cpu().numpy()
    if model.J_regressor.shape != (52, 6890) or len(parents) != 52:
        raise ValueError("Expected the 52-joint, 6890-vertex SMPL-H topology")
    for side, wrist in (("left", 20), ("right", 21)):
        ids = HAND_JOINTS[side]
        for finger in range(5):
            a, b, c = ids[1+3*finger:4+3*finger]
            if [int(parents[a]), int(parents[b]), int(parents[c])] != [wrist, a, b]:
                raise ValueError(f"SMPL-H {side} finger parent chain mismatch")

    # The parent graph cannot distinguish ring from pinky.  Check the rest
    # template once so a MANO/SMPL-H asset or hand-order change fails closed
    # instead of silently swapping the two distal fingers.
    import torch
    v_template = model.v_template.detach()
    if v_template.ndim == 3:
        v_template = v_template[0]
    rest_regressor = torch.as_tensor(model.J_regressor.detach().cpu().numpy(),
                                     dtype=v_template.dtype, device=v_template.device)
    rest_joints = rest_regressor @ v_template
    for side in ("left", "right"):
        ids = HAND_JOINTS[side]
        ring_chain = ids[1 + 3 * 3: 4 + 3 * 3]
        pinky_chain = ids[1 + 3 * 4: 4 + 3 * 4]
        ring_len = float(torch.linalg.vector_norm(rest_joints[ring_chain[1]] - rest_joints[ring_chain[0]])
                         + torch.linalg.vector_norm(rest_joints[ring_chain[2]] - rest_joints[ring_chain[1]]))
        pinky_len = float(torch.linalg.vector_norm(rest_joints[pinky_chain[1]] - rest_joints[pinky_chain[0]])
                          + torch.linalg.vector_norm(rest_joints[pinky_chain[2]] - rest_joints[pinky_chain[1]]))
        if not ring_len > pinky_len:
            raise ValueError(f"SMPL-H {side} ring/pinky template order check failed: {ring_len} <= {pinky_len}")
        distal = [ids[3], ids[6], ids[9], ids[12], ids[15]]
        tips = TIP_VERTICES[side]
        for finger, (distal_joint, tip_vertex) in enumerate(zip(distal, tips)):
            all_distances = torch.stack([torch.linalg.vector_norm(rest_joints[j] - v_template[tip_vertex]) for j in distal])
            if int(torch.argmin(all_distances)) != finger:
                raise ValueError(f"SMPL-H {side} fingertip order check failed at finger {finger}")

def read_view(path: Path, body: np.ndarray, image_size: tuple[int, int], camera: str):
    """Keep all candidates; select by same-side PMPose wrist proximity.

    Missing historical detector confidence is a labelled neutral weight (1),
    never a claimed model confidence. Bounds are diagnostic, not a deletion gate.
    A 150px wrist gate and 25px ambiguity margin are engineering association rules.
    """
    n = len(body)
    points = np.full((n, 2, 21, 2), np.nan, np.float32)
    weights = np.zeros((n, 2, 21), np.float32)
    audit, seen = [], set()
    width, height = image_size
    for line in path.open(encoding="utf-8"):
        row = json.loads(line); i = int(row["frame_index"])
        if i < 0 or i >= n: raise ValueError(f"{camera} frame {i} outside 0..{n-1}")
        if i in seen: raise ValueError(f"duplicate {camera} frame {i}")
        seen.add(i)
        if Path(row["image"]).stem != f"pair_{i:04d}":
            raise ValueError(f"image/pair_id mismatch: {row['image']}")
        for h, side in enumerate(("left", "right")):
            wrist = body[i, 9+h]
            entries = []
            for ordinal, candidate in enumerate(row["records"]):
                if candidate.get("side") != side: continue
                if candidate.get("pixel_frame") != "raw_fisheye":
                    raise ValueError("hand observation must be in raw fisheye pixels")
                p = np.asarray(candidate["keypoints_2d_raw_fisheye"], np.float32)
                if p.shape != (21, 2): raise ValueError("hand point shape is not 21x2")
                distance = float(np.linalg.norm(p[0]-wrist[:2])) if np.isfinite(wrist).all() and wrist[2] >= .25 and np.isfinite(p[0]).all() else None
                entries.append((distance, ordinal, candidate, p))
            ranked = sorted(entries, key=lambda x: x[0] if x[0] is not None else float("inf"))
            selected = None
            reason = "no_candidate"
            if ranked:
                dist = ranked[0][0]
                reason = "body_wrist_unavailable" if dist is None else "wrist_distance"
                if dist is not None and dist <= 150:
                    if len(ranked)>1 and ranked[1][0] is not None and ranked[1][0]-dist < 25:
                        reason = "ambiguous_same_side_candidates"
                    else:
                        selected = ranked[0][1]; reason = "selected"
                        c, p = ranked[0][2:]
                        conf = c.get("detector_confidence")
                        if conf is not None and not np.isfinite(conf):
                            raise ValueError(f"nonfinite detector confidence: {camera} frame {i} hand {side}")
                        weight = 1.0 if conf is None else float(np.clip(conf, 0, 1))
                        points[i,h] = p
                        finite = np.isfinite(p).all(-1)
                        bounds = finite & (p[:, 0] >= 0) & (p[:, 0] < width) & (p[:, 1] >= 0) & (p[:, 1] < height)
                        weights[i,h] = finite * np.where(bounds, 1.0, 0.1) * weight
            for dist, ordinal, c, p in entries:
                finite = np.isfinite(p).all(-1)
                bounds = finite & (p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)
                audit.append({"frame_index":i,"image":row["image"],"camera":camera,"hand":side,
                    "record_index":ordinal,"candidate_index":c.get("candidate_index"),
                    "selected":ordinal==selected,"reason":reason if ordinal==selected or selected is None else "farther_same_side_candidate",
                    "wrist_distance_px":dist,"detector_confidence":c.get("detector_confidence"),
                    "confidence_source":"missing_neutral_weight" if c.get("detector_confidence") is None else "detector_box",
                    "finite":finite.tolist(),"in_bounds":bounds.tolist(),"points":p.tolist(),
                    "source":"wilor_model_projection"})
            if not entries:
                audit.append({"frame_index":i,"camera":camera,"hand":side,"selected":False,"reason":"no_candidate"})
    if seen != set(range(n)): raise ValueError(f"{camera} frame range differs from body")
    return points, weights, audit


def finite_pixel_values(points: np.ndarray) -> np.ndarray:
    """Arithmetic copy only; original observations and validity stay unchanged."""
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def read_wilor(path: Path, n: int, side: str, body_points: np.ndarray,
               image_size: tuple[int, int], camera: str, consume_pixels: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Use the single audited WiLoR association implementation.

    The shared reader validates frame/image identity, duplicate frames and
    same-camera wrist association.  ``valid`` remains finite-only so rejected
    or out-of-bounds points are preserved for diagnostics and soft weighting.
    """
    from pose_app.smplh_hand_observation import read_view

    if not consume_pixels:
        # Native-MANO-only runs must not require projected points at all.
        # Original records remain in the input JSONL, not replaced with zero
        # observations. The unavailable arrays preserve downstream shapes.
        return (np.full((n,21,2), np.nan, np.float32), np.zeros((n,21), bool),
                np.zeros((n,21), bool), np.zeros((n,21), np.float32),
                [{"source": str(path), "camera": camera, "hand": side,
                  "reason": "hand_2d_disabled_not_consumed", "selected": False}])

    points, weights, audit = read_view(path, body_points, image_size, camera)
    hand_index = 0 if side == "left" else 1
    obs = points[:, hand_index]
    valid = np.isfinite(obs).all(axis=-1)
    width, height = image_size
    bounds_ok = (valid & (obs[:, :, 0] >= 0) & (obs[:, :, 0] < width)
                 & (obs[:, :, 1] >= 0) & (obs[:, :, 1] < height))
    return obs, valid, bounds_ok, weights[:, hand_index], audit
