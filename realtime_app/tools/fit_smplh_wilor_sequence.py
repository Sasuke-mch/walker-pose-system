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


def read_wilor(path: Path, n: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    obs = np.full((n, 21, 2), np.nan, np.float32)
    mask = np.zeros((n, 21), bool)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        frame = int(row["frame_index"])
        if not (0 <= frame < n):
            continue
        candidates = [c for c in row["records"] if c.get("side") == side and c.get("raw_pixel_bounds_ok")]
        if not candidates:
            continue
        candidate = max(candidates, key=lambda c: float(c.get("detector_confidence", 0.0)))
        points = np.asarray(candidate["keypoints_2d_raw_fisheye"], np.float32)
        if points.shape == (21, 2) and np.isfinite(points).all():
            obs[frame] = points
            mask[frame] = True
    return obs, mask


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
    hand_l, hand_l_mask = read_wilor(args.wilor_left, n, "left")
    hand_r, hand_r_mask = read_wilor(args.wilor_right, n, "left")
    hand_keep = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
    hand_l, hand_l_mask = hand_l[:, hand_keep], hand_l_mask[:, hand_keep]
    hand_r, hand_r_mask = hand_r[:, hand_keep], hand_r_mask[:, hand_keep]

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
    R01 = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=device)
    T01 = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=device)
    K0, D0 = cal.K0, cal.D0
    K1, D1 = cal.K1, cal.D1

    # Initialize translation from observed pelvis and the zero-pose SMPL-H mesh.
    with torch.no_grad():
        z = model(betas=torch.zeros(n, 10, device=device),
                  global_orient=root, body_pose=body_pose,
                  left_hand_pose=lhand, right_hand_pose=rhand,
                  transl=torch.zeros(n, 3, device=device), return_verts=True)
        zc = regress_coco17_torch(z.vertices, reg)
        pelvis_obs = np.nanmean(tri_m[:, [11, 12]], axis=1)
        transl[:] = torch.tensor(pelvis_obs, device=device) - zc[:, [11, 12]].mean(1)

    optim = torch.optim.Adam([beta, root, transl, body_pose, lhand, rhand], lr=args.lr)
    history = []
    # The WiLoR pickle contains 21 body/hand regressor joints followed by
    # 15 left- and 15 right-hand joints (51 total); wrist joints are the last
    # two body entries.
    hand_map_l = [19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    hand_map_r = [20, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # WiLoR OpenPose order -> SMPL-H joints: wrist, then index/middle/pinky/ring/thumb.
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)
        out = model(betas=beta.expand(n, -1), global_orient=root,
                    body_pose=body_pose, left_hand_pose=lhand,
                    right_hand_pose=rhand, transl=transl, return_verts=True)
        coco = regress_coco17_torch(out.vertices, reg)
        body_res = (coco - target).pow(2).sum(-1)
        body_loss = body_res[mask_body].mean() if mask_body.any() else body_res.mean() * 0.0
        joints = out.joints
        jl = joints[:, hand_map_l]
        jr = joints[:, hand_map_r]
        pl = fisheye_project_torch(jl, K0, D0)
        pr = fisheye_project_torch((R01 @ jr.transpose(1, 2)).transpose(1, 2) + T01, K1, D1)
        hand_l_loss = (pl - obs_l).pow(2).sum(-1)[mask_l].mean() if mask_l.any() else pl.sum() * 0.0
        hand_r_loss = (pr - obs_r).pow(2).sum(-1)[mask_r].mean() if mask_r.any() else pr.sum() * 0.0
        pose_reg = 1e-4 * (body_pose.pow(2).mean() + lhand.pow(2).mean() + rhand.pow(2).mean())
        loss = body_loss + 1e-5 * (hand_l_loss + hand_r_loss) + pose_reg + 1e-3 * beta.pow(2).mean()
        loss.backward()
        optim.step()
        if step % 10 == 0 or step == args.steps - 1:
            history.append({"step": step, "loss": float(loss.detach()),
                            "body_m": float(torch.sqrt(body_loss.detach())),
                            "hand_l_px": float(torch.sqrt(hand_l_loss.detach())),
                            "hand_r_px": float(torch.sqrt(hand_r_loss.detach())),
                            "hand_l_points": int(mask_l.sum()), "hand_r_points": int(mask_r.sum())})

    with torch.no_grad():
        final = model(betas=beta.expand(n, -1), global_orient=root,
                      body_pose=body_pose, left_hand_pose=lhand,
                      right_hand_pose=rhand, transl=transl, return_verts=True)
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
                        body_quality=quality)
    (args.output_dir / "fit_summary.json").write_text(json.dumps({
        "status": "engineering_candidate", "frames": n, "steps": args.steps,
        "model": str(args.smplh_model.resolve()), "vertices": 6890,
        "hand_observation_source": "WiLoR_model_projected_MANO_joints",
        "wilor_left_valid_points": int(hand_l_mask.sum()),
        "wilor_right_valid_points": int(hand_r_mask.sum()), "history": history,
        "body_accepted_points": int(body_mask.sum()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
