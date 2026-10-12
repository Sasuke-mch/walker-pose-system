"""Explicit bounded parameter smoothing, followed by real SMPL-H re-decoding."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def smooth_rotations(vectors, eligible, strength=.1, limit_degrees=2.):
    shape = vectors.shape
    rotations = Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(*shape[:-1], 3, 3)
    center = rotations[1:-1]
    left = np.swapaxes(center, -1, -2) @ rotations[:-2]
    right = np.swapaxes(center, -1, -2) @ rotations[2:]
    delta = strength * (Rotation.from_matrix(left.reshape(-1, 3, 3)).as_rotvec()
                        + Rotation.from_matrix(right.reshape(-1, 3, 3)).as_rotvec())
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    delta *= np.minimum(1, np.deg2rad(limit_degrees) / np.maximum(norm, 1e-12))
    updated = center @ Rotation.from_rotvec(delta).as_matrix().reshape(center.shape)
    result = rotations.copy()
    result[1:-1][eligible] = updated[eligible]
    output = Rotation.from_matrix(result.reshape(-1, 3, 3)).as_rotvec().reshape(shape)
    # Preserve endpoints/gaps byte-for-byte, including axis-angle conventions.
    active = np.zeros(len(vectors), dtype=bool)
    active[1:-1] = eligible
    output[~active] = vectors[~active]
    return output


def smooth_translation(values, eligible, strength=.1, limit_m=.003):
    delta = strength * (values[:-2] + values[2:] - 2 * values[1:-1])
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    delta *= np.minimum(1, limit_m / np.maximum(norm, 1e-12))
    result = values.copy()
    result[1:-1][eligible] += delta[eligible]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    options = json.loads((args.source / "resolved_options.json").read_text(encoding="utf-8"))
    with np.load(args.source / "result.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    ids = data["pair_id"]
    supported = (data["body_left_2d_valid"] | data["body_right_2d_valid"]).any(1)
    eligible = (supported[:-2] & supported[1:-1] & supported[2:]
                & (np.diff(ids)[:-1] == 1) & (np.diff(ids)[1:] == 1))
    before = {key: data[key].copy() for key in ("body_pose", "global_orient", "transl")}
    data["body_pose"] = smooth_rotations(data["body_pose"].reshape(len(ids), 21, 3), eligible).reshape(len(ids), 63)
    data["global_orient"] = smooth_rotations(data["global_orient"], eligible)
    data["transl"] = smooth_translation(data["transl"], eligible)
    from pose_app.smpl_coco_observation import (
        _install_legacy_smpl_pickle_compatibility,
        load_coco17_regressor,
        regress_coco17_torch,
    )
    _install_legacy_smpl_pickle_compatibility()
    import smplx
    import torch
    from smplx.utils import Struct
    with Path(options["smplh_model"]).open("rb") as handle:
        model_data = pickle.load(handle, encoding="latin1")
    for side, suffix in (("left", "l"), ("right", "r")):
        with Path(options[f"mano_{side}"]).open("rb") as handle:
            mano = pickle.load(handle, encoding="latin1")
        model_data[f"hands_components{suffix}"] = np.asarray(mano["hands_components"], np.float32)
        model_data[f"hands_mean{suffix}"] = np.asarray(mano["hands_mean"], np.float32)
    device = options["device"]
    model = smplx.SMPLH(options["smplh_model"], data_struct=Struct(**model_data),
                        gender="male", use_pca=False, flat_hand_mean=True, batch_size=32).to(device)
    regressor = load_coco17_regressor(options["regressor"])
    from pose_app.smplh_hand_observation import hand21
    vertices, joints, coco, left_points, right_points = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            stop = min(start + 32, len(ids))
            def tensor(key, start=start, stop=stop):
                return torch.as_tensor(data[key][start:stop], dtype=torch.float32, device=device)
            out = model(betas=torch.as_tensor(data["betas"], dtype=torch.float32, device=device).expand(stop-start, -1),
                        global_orient=tensor("global_orient"), transl=tensor("transl"),
                        body_pose=tensor("body_pose"), left_hand_pose=tensor("left_hand_pose"),
                        right_hand_pose=tensor("right_hand_pose"), return_verts=True)
            vertices.append(out.vertices.cpu().numpy())
            joints.append(out.joints.cpu().numpy())
            coco.append(regress_coco17_torch(out.vertices, regressor).cpu().numpy())
            left_points.append(hand21(out.joints, out.vertices, "left").cpu().numpy())
            right_points.append(hand21(out.joints, out.vertices, "right").cpu().numpy())
    for key in ("vertices", "predicted_coco", "smplh_joints", "vposer_latent", "hand_points_left", "hand_points_right"):
        data[f"raw_{key}"] = data.pop(key)
    data["vertices"] = np.concatenate(vertices)
    data["smplh_joints"] = np.concatenate(joints)
    data["predicted_coco"] = np.concatenate(coco)
    data["hand_points_left"] = np.concatenate(left_points)
    data["hand_points_right"] = np.concatenate(right_points)
    # No viewer may consume old ground-space geometry from a source archive.
    data.pop("vertices_ground_m", None)
    for key in ("vertices", "smplh_joints", "predicted_coco"):
        if not np.isfinite(data[key]).all():
            raise RuntimeError(f"nonfinite re-decoded {key}")
    np.testing.assert_array_equal(data["faces"], model.faces)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "result.npz", **data)
    report = {"source": str(args.source), "output": str(args.output), "frames": len(ids),
              "mode": "explicit_user_requested_display_variant_not_refit", "strength_each_neighbor": .1,
              "rotation_limit_degrees_per_joint": 2., "translation_limit_m": .003,
              "eligible_triplets": int(eligible.sum()), "endpoints_and_gaps_unchanged": True,
              "beta_and_hands_unchanged": True, "vertices_redecoded_from_parameters": True,
              "raw_observations_ground_stage_walker_camera_smoothed": False,
              "maximum_translation_correction_m": float(np.linalg.norm(data["transl"]-before["transl"], axis=-1).max()),
              "maximum_vertex_correction_m": float(np.linalg.norm(data["vertices"]-data["raw_vertices"], axis=-1).max())}
    (args.output / "smoothing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
