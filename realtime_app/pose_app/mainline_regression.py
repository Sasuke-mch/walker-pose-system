"""Read-only engineering regression checks for the frozen no-contact SMPL-H route.

NaN positions in rejected observations are compared, never silently dropped.
This module does not change optimization, create golden results, or certify accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ResourceUnavailableError(RuntimeError):
    """Required execution device is unavailable, rather than a regression pass."""


def compare_arrays(reference, candidate, atol=0.0):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {"passed": False, "reason": "shape_or_dtype_changed"}
    if reference.dtype.kind not in "fci":
        return {"passed": bool(np.array_equal(reference, candidate))}
    if not np.array_equal(np.isnan(reference), np.isnan(candidate)):
        return {"passed": False, "reason": "nan_locations_changed"}
    if not np.array_equal(np.isposinf(reference), np.isposinf(candidate)) or not np.array_equal(
        np.isneginf(reference), np.isneginf(candidate)
    ):
        return {"passed": False, "reason": "infinity_changed"}
    valid = np.isfinite(reference)
    delta = np.abs(reference[valid].astype(np.float64) - candidate[valid].astype(np.float64))
    maximum = float(delta.max(initial=0))
    return {"passed": maximum <= atol, "max_abs": maximum, "atol": atol}


def compare_archives(reference, candidate, atol=0.0):
    checks = {"keys": {"passed": set(reference) == set(candidate)}}
    for name in sorted(set(reference) & set(candidate)):
        # Integer IDs/topology, masks and labels must remain exact.
        tolerance = atol if reference[name].dtype.kind == "f" else 0.0
        checks[name] = compare_arrays(reference[name], candidate[name], tolerance)
    return checks


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def observation_snapshot(args):
    from pose_app.body_observations import raw_side, raw_triangulate
    from pose_app.fisheye_camera import load_stereo_fisheye

    # raw_side is preserved as production logic; detect duplicate IDs before its
    # dictionary construction can overwrite them in a regression audit.
    for path in (args.left_raw, args.right_raw):
        images = read_json(path).get("images", [])
        source_ids = [int(row.get("image_id", row.get("frame_index", index)))
                      for index, row in enumerate(images)]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(f"duplicate frame IDs: {path}")
    left = raw_side(args.left_raw, "left")
    right = raw_side(args.right_raw, "right")
    if set(left) != set(right):
        raise ValueError("left/right frame IDs differ")
    ids = sorted(left)
    if not ids or ids != list(range(ids[0], ids[-1] + 1)):
        raise ValueError("empty or non-contiguous frame IDs")
    left = np.stack([left[i] for i in ids]).astype(np.float32)
    right = np.stack([right[i] for i in ids]).astype(np.float32)
    cal = load_stereo_fisheye(args.calibration_dir)
    tri, el, er, gap, dl, dr, mask, reason, quality, terms = raw_triangulate(left, right, cal)
    snapshot = {
        "pair_id": np.asarray(ids, np.int64), "left_raw_pixels": left,
        "right_raw_pixels": right, "triangulated_mm": tri,
        "reprojection_left_px": el, "reprojection_right_px": er,
        "ray_gap_mm": gap, "depth_left_mm": dl, "depth_right_mm": dr,
        "accepted": mask, "reason": np.asarray(reason, dtype="U64"), "quality": quality,
        "K0": cal.K0, "D0": cal.D0, "K1": cal.K1, "D1": cal.D1,
        "R_cam0_to_cam1": cal.R_cam0_to_cam1, "T_cam0_to_cam1_mm": cal.T_cam0_to_cam1_mm,
        "image_size": np.asarray([cal.image_width, cal.image_height], np.int64),
        "camera_roles": np.asarray(["cam0=LEFT", "cam1=RIGHT"]),
        "units": np.asarray(["raw_pixels=px", "triangulation=mm", "model=m"]),
    }
    for name, term in zip(("q2d", "qray", "qrep", "qsync", "qdepth"), terms):
        snapshot[name] = term
    return snapshot


def audit_contract(result, initialization, summary, snapshot, args):
    from pose_app.smplh_fitting.stages import build_stage_schedule

    checks = {}
    def check(name, condition):
        checks[name] = {"passed": bool(condition)}

    n = len(snapshot["pair_id"])
    check("frames", np.array_equal(result["pair_id"], snapshot["pair_id"]))
    check("initial_frames", np.array_equal(initialization["pair_id"], snapshot["pair_id"]))
    check("shared_beta", result["betas"].shape == (1, 10))
    check("cold_beta_zero", initialization["betas"].shape == (1, 10)
          and np.all(initialization["betas"] == 0))
    check("cold_latent_zero", initialization["vposer_latent"].shape == (n, 32)
          and np.all(initialization["vposer_latent"] == 0))
    check("saved_pose_shapes", result["body_pose"].shape == (n, 63)
          and result["vposer_latent"].shape == (n, 32))
    for key in ("vertices", "betas", "body_pose", "vposer_latent", "global_orient", "transl",
                "left_hand_pose", "right_hand_pose", "predicted_coco", "smplh_joints"):
        check("finite_" + key, np.isfinite(result[key]).all())
    checks["triangulated_mm"] = compare_arrays(snapshot["triangulated_mm"], result["raw_triangulated_points"])
    checks["body_accepted"] = compare_arrays(
        snapshot["accepted"] & np.isfinite(snapshot["triangulated_mm"]).all(-1), result["body_accepted"])
    checks["body_quality"] = compare_arrays(snapshot["quality"], result["body_quality"])
    check("no_contact", summary["contact_enabled"] is False)
    check("shared_hand", summary["shared_hand_pose"] is True and summary["hand_pose_parameter_frames"] == 1)
    for side in ("left", "right"):
        coefficients = result[f"{side}_hand_pca"]
        check(side + "_shared_pca", coefficients.shape == (n, args.hand_pca_comps)
              and np.array_equal(coefficients, np.broadcast_to(coefficients[:1], coefficients.shape)))
    names, train, steps = build_stage_schedule(args, False)
    check("frozen_stage_policy", train == [["transl", "latent"], ["beta"],
          ["beta", "root", "transl", "latent"], ["lhand", "rhand"],
          ["lhand", "rhand"], ["lhand", "rhand"]])
    expected = [(name, parameters, budget) for name, parameters, budget in zip(names, train, steps)]
    actual = [(row["stage"], row["trainable"], row["steps"]) for row in summary["stage_schedule"]]
    check("stage_trainability_and_budget", actual == expected)
    check("no_body_smoothing_or_2d", summary["body_temporal_weight"] == args.body_temporal_weight == 0
          and summary["body_reprojection"]["weight"] == args.body_reprojection_weight == 0)
    return checks


def replay_parameters(result, args, chunk_size=32):
    # Import the fitter's compatibility setup before reading upstream pickle assets.
    from pose_app.smplh_fitting import pipeline  # noqa: F401
    import pickle
    import smplx
    import torch
    from smplx.utils import Struct
    from pose_app.smplx_fitting import load_vposer_explicit
    from pose_app.smpl_coco_observation import load_coco17_regressor, regress_coco17_torch
    from pose_app.smplh_hand_observation import hand21

    with args.smplh_model.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    hand_data = {}
    for side, path in (("l", args.mano_left), ("r", args.mano_right)):
        with path.open("rb") as stream:
            mano = pickle.load(stream, encoding="latin1")
        hand_data[side] = mano
        data["hands_components" + side] = np.asarray(mano["hands_components"], np.float32)
        data["hands_mean" + side] = np.asarray(mano["hands_mean"], np.float32)
    model = smplx.SMPLH(str(args.smplh_model), data_struct=Struct(**data), gender="male",
                        use_pca=False, flat_hand_mean=True, batch_size=1).to(args.device)
    vposer, _, _ = load_vposer_explicit(args.vposer_dir, args.device)
    reg = load_coco17_regressor(args.regressor)
    rebuilt = {key: [] for key in ("body_pose", "vertices", "smplh_joints", "predicted_coco",
                                   "hand_points_left", "hand_points_right", "left_hand_pose", "right_hand_pose")}
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=args.device)

    with torch.no_grad():
        for start in range(0, len(result["pair_id"]), chunk_size):
            sl = slice(start, start + chunk_size)
            pose = vposer.decode(tensor(result["vposer_latent"][sl]))["pose_body"].reshape(-1, 63)
            hands = {}
            for side, abbreviation in (("left", "l"), ("right", "r")):
                components = tensor(hand_data[abbreviation]["hands_components"][:args.hand_pca_comps])
                scale = torch.linalg.vector_norm(components, dim=1).clamp_min(1e-6)
                hands[side] = tensor(hand_data[abbreviation]["hands_mean"])[None] + (
                    tensor(result[f"{side}_hand_pca"][sl]) / scale[None]) @ components
            count = len(pose)
            out = model(betas=tensor(result["betas"]).expand(count, -1),
                        global_orient=tensor(result["global_orient"][sl]), body_pose=pose,
                        left_hand_pose=hands["left"], right_hand_pose=hands["right"],
                        transl=tensor(result["transl"][sl]), return_verts=True)
            values = {"body_pose": pose, "vertices": out.vertices, "smplh_joints": out.joints,
                      "predicted_coco": regress_coco17_torch(out.vertices, reg),
                      "hand_points_left": hand21(out.joints, out.vertices, "left"),
                      "hand_points_right": hand21(out.joints, out.vertices, "right"),
                      "left_hand_pose": hands["left"], "right_hand_pose": hands["right"]}
            for name, value in values.items():
                rebuilt[name].append(value.cpu().numpy())
    # Chunked GEMM can differ at float32 rounding level; tolerances are absolute.
    checks = {"faces": compare_arrays(result["faces"], np.asarray(model.faces))}
    for name, parts in rebuilt.items():
        checks[name] = compare_arrays(result[name], np.concatenate(parts), 2e-6)
    return checks


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
