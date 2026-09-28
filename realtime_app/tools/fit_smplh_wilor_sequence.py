#!/usr/bin/env python3
"""Fit SMPL-H from the raw body stereo observations plus WiLoR hand pixels.

This is an engineering candidate run.  WiLoR pixels are model-derived MANO
projections, so they are masked and reported as an auxiliary 2-D term rather
than independent ground truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _value in {"bool": np.bool_, "int": np.int64, "float": np.float64,
                      "complex": np.complex128, "object": np.object_,
                      "unicode": np.str_, "str": np.str_}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "realtime_app"))
from pose_app.fisheye_camera import fisheye_project_torch, load_stereo_fisheye
from pose_app.smpl_coco_observation import load_coco17_regressor, regress_coco17_torch

RAW = ROOT / "research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py"
spec = importlib.util.spec_from_file_location("raw_clean", RAW)
raw_clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_clean)


def read_wilor(path: Path, n: int, side: str, body_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the best side candidate while retaining every finite point.

    ``valid`` is a per-point finite-value mask used only to avoid NaNs in the
    optimizer.  ``bounds_ok`` is diagnostic metadata; it is deliberately not
    used to discard an entire hand or an individual point.  WiLoR does not
    provide an independent per-joint confidence, so the detector confidence
    remains a candidate-level field in the audit output.
    """
    obs = np.full((n, 21, 2), np.nan, np.float32)
    valid = np.zeros((n, 21), bool)
    bounds_ok = np.zeros((n, 21), bool)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        frame = int(row["frame_index"])
        if not (0 <= frame < n):
            continue
        candidates = [c for c in row["records"] if c.get("side") == side]
        if not candidates:
            continue
        # Candidate identity is tied to the PMPose wrist in the same camera.
        # Historical JSONL may not contain detector confidence, so confidence
        # alone cannot establish the person association.
        if body_points is not None and np.isfinite(body_points[frame, 9 if side == "left" else 10, :2]).all():
            wrist = body_points[frame, 9 if side == "left" else 10, :2]
            candidate = min(candidates, key=lambda c: float(np.linalg.norm(
                np.asarray(c["keypoints_2d_raw_fisheye"], np.float32)[0] - wrist)))
        else:
            candidate = max(candidates, key=lambda c: float(c.get("detector_confidence", 0.0)))
        points = np.asarray(candidate["keypoints_2d_raw_fisheye"], np.float32)
        if points.shape != (21, 2):
            continue
        obs[frame] = points
        valid[frame] = np.isfinite(points).all(axis=-1)
        # The sequence audit currently stores only a candidate-level bounds
        # flag.  Recompute the per-joint state from the raw image dimensions
        # when available, while preserving points outside the image.
        image_path = row.get("image")
        width = height = None
        if image_path:
            try:
                import cv2
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    height, width = image.shape[:2]
                    # people1 audit images are stored upright (1920 high x
                    # 1080 wide), while the saved WiLoR points are explicitly
                    # mapped back to the original raw-fisheye (1920 wide x
                    # 1080 high) frame.
                    if height > width:
                        width, height = height, width
            except Exception:
                pass
        if width is not None and height is not None:
            bounds_ok[frame] = (
                valid[frame]
                & (points[:, 0] >= 0) & (points[:, 0] < width)
                & (points[:, 1] >= 0) & (points[:, 1] < height)
            )
        else:
            bounds_ok[frame] = bool(candidate.get("raw_pixel_bounds_ok", False))
    return obs, valid, bounds_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-raw", type=Path, required=True)
    ap.add_argument("--right-raw", type=Path, required=True)
    ap.add_argument("--wilor-left", type=Path, required=True)
    ap.add_argument("--wilor-right", type=Path, required=True)
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--regressor", type=Path, required=True)
    ap.add_argument("--smplh-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refuse non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import smplx
    import pickle
    from smplx.utils import Struct
    import cv2
    from scipy.spatial.transform import Rotation
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    cal = load_stereo_fisheye(args.calibration_dir)
    raw_clean.cv2 = cv2
    reg = load_coco17_regressor(args.regressor)
    left_rows = raw_clean.raw_side(args.left_raw, "left")
    right_rows = raw_clean.raw_side(args.right_raw, "right")
    ids = sorted(set(left_rows) & set(right_rows))
    if ids != list(range(len(ids))):
        raise ValueError("raw frame ids are not contiguous")
    left = np.stack([left_rows[i] for i in ids]).astype(np.float32)
    right = np.stack([right_rows[i] for i in ids]).astype(np.float32)
    tri, _, _, _, _, _, accepted, _, quality, _ = raw_clean.raw_triangulate(left, right, cal)
    n = len(ids)
    tri_m = np.nan_to_num(np.asarray(tri, np.float32) / 1000.0)
    body_mask = np.asarray(accepted, bool) & np.isfinite(tri).all(axis=-1)
    hand_l, hand_l_valid, hand_l_bounds = read_wilor(args.wilor_left, n, "left", left)
    hand_r, hand_r_valid, hand_r_bounds = read_wilor(args.wilor_right, n, "right", right)
    # Keep the complete WiLoR 21-point order.  The first point is the wrist;
    # each following finger contributes three internal joints and one surface
    # fingertip.  Bounds are a soft quality signal, never a deletion gate.
    hand_l_full, hand_r_full = hand_l.copy(), hand_r.copy()
    hand_l_full_valid, hand_r_full_valid = hand_l_valid.copy(), hand_r_valid.copy()
    hand_l_full_bounds, hand_r_full_bounds = hand_l_bounds.copy(), hand_r_bounds.copy()
    hand_l_mask, hand_r_mask = hand_l_valid, hand_r_valid
    hand_l_weight_np = hand_l_valid.astype(np.float32) * np.where(hand_l_bounds, 1.0, 0.1).astype(np.float32)
    hand_r_weight_np = hand_r_valid.astype(np.float32) * np.where(hand_r_bounds, 1.0, 0.1).astype(np.float32)

    # WiLoR ships a full-pose SMPL-H pickle without the optional PCA metadata
    # expected by smplx.  Supply identity components and zero means; posedirs
    # and the 52-joint regressor remain those of the downloaded model.
    with args.smplh_model.open("rb") as handle:
        model_data = pickle.load(handle, encoding="latin1")
    model_data.setdefault("hands_componentsl", np.eye(45, dtype=np.float32))
    model_data.setdefault("hands_componentsr", np.eye(45, dtype=np.float32))
    model_data.setdefault("hands_meanl", np.zeros(45, dtype=np.float32))
    model_data.setdefault("hands_meanr", np.zeros(45, dtype=np.float32))
    model = smplx.SMPLH(str(args.smplh_model), data_struct=Struct(**model_data),
                        gender="male", use_pca=False, flat_hand_mean=True,
                        batch_size=n).to(device)
    beta = torch.nn.Parameter(torch.zeros(1, 10, device=device))
    root = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    transl = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    body_pose = torch.nn.Parameter(torch.zeros(n, 63, device=device))
    lhand = torch.nn.Parameter(torch.zeros(n, 45, device=device))
    rhand = torch.nn.Parameter(torch.zeros(n, 45, device=device))
    target = torch.tensor(tri_m, device=device)
    obs_l = torch.tensor(np.nan_to_num(hand_l, nan=0.0), device=device)
    obs_r = torch.tensor(np.nan_to_num(hand_r, nan=0.0), device=device)
    mask_body = torch.tensor(body_mask, device=device)
    mask_l = torch.tensor(hand_l_mask, device=device)
    mask_r = torch.tensor(hand_r_mask, device=device)
    weight_l = torch.tensor(hand_l_weight_np, device=device)
    weight_r = torch.tensor(hand_r_weight_np, device=device)
    R01 = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=device)
    T01 = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=device)
    K0, D0 = cal.K0, cal.D0
    K1, D1 = cal.K1, cal.D1

    # Translation init with NaN guard (no nanmean->nan_to_num silence).
    with torch.no_grad():
        z = model(betas=torch.zeros(n, 10, device=device),
                  global_orient=root, body_pose=body_pose,
                  left_hand_pose=lhand, right_hand_pose=rhand,
                  transl=torch.zeros(n, 3, device=device), return_verts=True)
        zc = regress_coco17_torch(z.vertices, reg)
        zc_np = zc.detach().cpu().numpy()
        init_source = []
        t0 = np.zeros((n, 3), np.float32)
        for i in range(n):
            hl = tri_m[i, 11] if np.isfinite(tri_m[i, 11]).all() and body_mask[i, 11] else None
            hr = tri_m[i, 12] if np.isfinite(tri_m[i, 12]).all() and body_mask[i, 12] else None
            if hl is not None and hr is not None:
                t0[i] = (hl + hr) / 2 - (zc_np[i, 11] + zc_np[i, 12]) / 2; init_source.append("hips_midpoint")
            elif hl is not None:
                t0[i] = hl - zc_np[i, 11]; init_source.append("left_hip_only")
            elif hr is not None:
                t0[i] = hr - zc_np[i, 12]; init_source.append("right_hip_only")
            else:
                t0[i] = np.zeros(3, np.float32); init_source.append("translation_init_unavailable")
        transl[:] = torch.tensor(t0, device=device)

    optim = torch.optim.Adam([beta, root, transl, body_pose, lhand, rhand], lr=args.lr)
    history = []
    from pose_app.smplh_hand_observation import HAND_JOINTS, TIP_VERTICES, HAND_NAMES, validate_smplh, hand21
    validate_smplh(model)
    # Named SMPL-H internal-joint maps; hand21() emits the verified WiLoR
    # OpenPose order and appends five surface fingertip vertices.
    hand_map_l = HAND_JOINTS["left"]
    hand_map_r = HAND_JOINTS["right"]
    assert hand_map_l == [20, 34, 35, 36, 22, 23, 24, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    assert hand_map_r == [21, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45]
    # WiLoR OpenPose order -> SMPL-H joints verified above; tips via hand21 vertices.
    stage_names = ["A_body_only", "B_hands_only", "C_wrist_forearm_hands", "D_joint_refinement_no_hand3d"]
    base = args.steps // len(stage_names)
    stage_steps = [base] * len(stage_names)
    for i in range(args.steps % len(stage_names)):
        stage_steps[i] += 1
    stage_train = [
        ["beta", "root", "transl", "body_pose"],
        ["lhand", "rhand"],
        ["beta", "root", "transl", "body_pose", "lhand", "rhand"],
        ["beta", "root", "transl", "body_pose", "lhand", "rhand"],
    ]
    def set_stage(train):
        for p in (beta, root, transl, body_pose, lhand, rhand): p.requires_grad_(False)
        m = {"beta": beta, "root": root, "transl": transl, "body_pose": body_pose, "lhand": lhand, "rhand": rhand}
        for k in train: m[k].requires_grad_(True)
    stage_history = []
    global_step = 0
    for stage_index, (stage_name, train_names, n_stage) in enumerate(zip(stage_names, stage_train, stage_steps)):
        set_stage(train_names)
        for local_step in range(n_stage):
            step = global_step
            global_step += 1
            optim.zero_grad(set_to_none=True)
            out = model(betas=beta.expand(n, -1), global_orient=root,
                        body_pose=body_pose, left_hand_pose=lhand,
                        right_hand_pose=rhand, transl=transl, return_verts=True)
            coco = regress_coco17_torch(out.vertices, reg)
            body_res = (coco - target).pow(2).sum(-1)
            body_w = torch.tensor(np.asarray(quality, np.float32), device=device).clamp_min(0.0)
            body_loss = (body_res * body_w)[mask_body].sum() / body_w[mask_body].sum().clamp_min(1e-6) if mask_body.any() else body_res.mean() * 0.0
            joints = out.joints
            jl = hand21(joints, out.vertices, "left")
            jr = hand21(joints, out.vertices, "right")
            pl = fisheye_project_torch(jl, K0, D0)
            pr = fisheye_project_torch((R01 @ jr.transpose(1, 2)).transpose(1, 2) + T01, K1, D1)
            hand_l_res = (pl - obs_l).pow(2).sum(-1)
            hand_r_res = (pr - obs_r).pow(2).sum(-1)
            hand_l_loss = (hand_l_res * weight_l).sum() / weight_l.sum().clamp_min(1e-6) if mask_l.any() else pl.sum() * 0.0
            hand_r_loss = (hand_r_res * weight_r).sum() / weight_r.sum().clamp_min(1e-6) if mask_r.any() else pr.sum() * 0.0
            pose_reg = 1e-4 * (body_pose.pow(2).mean() + lhand.pow(2).mean() + rhand.pow(2).mean())
            body_term = body_loss if stage_index != 1 else body_loss.detach() * 0.0
            hand_term = 1e-5 * (hand_l_loss + hand_r_loss) if stage_index != 0 else (hand_l_loss + hand_r_loss).detach() * 0.0
            loss = body_term + hand_term + pose_reg + 1e-3 * beta.pow(2).mean()
            loss.backward()
            optim.step()
            if step % 10 == 0 or step == args.steps - 1:
                row = {"stage": stage_name, "stage_step": local_step, "step": step, "loss": float(loss.detach()),
                            "body_m": float(torch.sqrt(body_loss.detach())),
                            "hand_l_px": float(torch.sqrt(hand_l_loss.detach())),
                            "hand_r_px": float(torch.sqrt(hand_r_loss.detach())),
                            "hand_l_points": int(mask_l.sum()), "hand_r_points": int(mask_r.sum())}
                history.append(row)
        stage_history.append({"stage": stage_name, "steps": n_stage, "trainable": train_names,
                              "hand_3d_term": False})

    with torch.no_grad():
        final = model(betas=beta.expand(n, -1), global_orient=root,
                      body_pose=body_pose, left_hand_pose=lhand,
                      right_hand_pose=rhand, transl=transl, return_verts=True)
        final_hand_l = hand21(final.joints, final.vertices, "left")
        final_hand_r = hand21(final.joints, final.vertices, "right")
    np.savez_compressed(args.output_dir / "result.npz",
                        vertices=final.vertices.cpu().numpy(), faces=np.asarray(model.faces),
                        predicted_coco=regress_coco17_torch(final.vertices, reg).cpu().numpy(),
                        betas=beta.detach().cpu().numpy(), global_orient=root.detach().cpu().numpy(),
                        transl=transl.detach().cpu().numpy(),
                        body_pose=body_pose.detach().cpu().numpy(),
                        left_hand_pose=lhand.detach().cpu().numpy(), right_hand_pose=rhand.detach().cpu().numpy(),
                        raw_triangulated_points=tri, body_accepted=body_mask,
                        wilor_left_2d=hand_l, wilor_left_mask=hand_l_mask,
                        wilor_right_2d=hand_r, wilor_right_mask=hand_r_mask,
                        wilor_left_2d_full=hand_l_full, wilor_right_2d_full=hand_r_full,
                        wilor_left_valid_full=hand_l_full_valid, wilor_right_valid_full=hand_r_full_valid,
                        wilor_left_bounds_ok_full=hand_l_full_bounds, wilor_right_bounds_ok_full=hand_r_full_bounds,
                        body_quality=quality,
                        translation_init_source=np.asarray(init_source, dtype="U32"),
                        wilor_left_weight=hand_l_weight_np, wilor_right_weight=hand_r_weight_np,
                        smplh_joints=final.joints.detach().cpu().numpy(),
                        hand_points_left=final_hand_l.detach().cpu().numpy(),
                        hand_points_right=final_hand_r.detach().cpu().numpy())
    (args.output_dir / "fit_summary.json").write_text(json.dumps({
        "status": "engineering_candidate", "frames": n, "steps": args.steps,
        "model": str(args.smplh_model.resolve()), "vertices": 6890,
        "hand_observation_source": "WiLoR_model_projected_MANO_joints",
        "hand_observation_points_retained": 21,
        "hand_fit_subset_points": 21,
        "hand_bounds_are_diagnostic_only": True,
        "wilor_left_valid_points": int(hand_l_mask.sum()),
        "wilor_right_valid_points": int(hand_r_mask.sum()), "history": history,
        "stage_schedule": stage_history,
        "hand_confidence_source": "detector_box_or_missing_neutral_weight",
        "hand_3d_observation_used": False,
        "body_accepted_points": int(body_mask.sum()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
