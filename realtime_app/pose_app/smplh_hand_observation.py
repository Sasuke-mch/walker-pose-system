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
    rest_joints = torch.as_tensor(model.J_regressor.detach().cpu().numpy(), dtype=v_template.dtype) @ v_template
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
            distances = torch.linalg.vector_norm(rest_joints[distal] - v_template[tip_vertex])
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
