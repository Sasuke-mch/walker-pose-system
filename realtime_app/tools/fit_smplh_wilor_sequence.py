#!/usr/bin/env python3
"""Fit VPoser-parameterized SMPL-H from body stereo plus WiLoR hand pixels.

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
from pose_app.smplx_fitting import load_vposer_explicit
from pose_app import smpl_surface_contact as surface_contact

RAW = ROOT / "research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py"
spec = importlib.util.spec_from_file_location("raw_clean", RAW)
raw_clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_clean)


def read_wilor(path: Path, n: int, side: str, body_points: np.ndarray | None = None,
               image_size: tuple[int, int] = (1920, 1080)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            ranked = sorted(candidates, key=lambda c: float(np.linalg.norm(
                np.asarray(c["keypoints_2d_raw_fisheye"], np.float32)[0] - wrist)))
            distances = [float(np.linalg.norm(np.asarray(c["keypoints_2d_raw_fisheye"], np.float32)[0] - wrist)) for c in ranked]
            if distances[0] > 150.0 or (len(distances) > 1 and distances[1] - distances[0] < 25.0):
                continue
            candidate = ranked[0]
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
        width, height = image_size
        bounds_ok[frame] = (
                valid[frame]
                & (points[:, 0] >= 0) & (points[:, 0] < width)
                & (points[:, 1] >= 0) & (points[:, 1] < height)
            )
    return obs, valid, bounds_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-raw", type=Path, required=True)
    ap.add_argument("--right-raw", type=Path, required=True)
    ap.add_argument("--wilor-left", type=Path, required=True,
                    help="WiLoR JSONL from left camera; both left/right records are read")
    ap.add_argument("--wilor-right", type=Path, required=True,
                    help="WiLoR JSONL from right camera; both left/right records are read")
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--regressor", type=Path, required=True)
    ap.add_argument("--smplh-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--base-steps", type=int, default=None,
                    help="Stage A body/VPoser steps; defaults to steps//6")
    ap.add_argument("--beta-steps", type=int, default=None,
                    help="Stage B shared-beta steps; defaults to steps//6")
    ap.add_argument("--joint-steps", type=int, default=None,
                    help="Stage C VPoser/body refinement steps; defaults to steps//6")
    ap.add_argument("--hand-steps", type=int, default=None,
                    help="Stage D1 proximal-weighted hand steps; defaults to steps//6")
    ap.add_argument("--contact-steps", type=int, default=None,
                    help="Stage D2 hand-contact steps; defaults to steps//6")
    ap.add_argument("--contact-refine-steps", type=int, default=None,
                    help="Stage D3 limited contact refinement; defaults to steps//6")
    ap.add_argument("--vposer-dir", type=Path, default=None)
    ap.add_argument("--vposer-prior-weight", type=float, default=0.02)
    ap.add_argument("--contact-labels", type=Path, default=None)
    ap.add_argument("--scene-transforms", type=Path, default=None)
    ap.add_argument("--contact-vertex-sets", type=Path, default=None)
    ap.add_argument("--walker-topology", type=Path, default=None,
                    help="validated walker topology; used to check handle semantics")
    ap.add_argument("--surface-hand-contact-weight", type=float, default=0.0)
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
    if args.vposer_dir is None:
        raise ValueError("--vposer-dir is required: SMPL-H body pose is parameterized by VPoser")
    if args.surface_hand_contact_weight < 0:
        raise ValueError("--surface-hand-contact-weight must be non-negative")
    contact_args = (args.contact_labels, args.scene_transforms,
                    args.contact_vertex_sets, args.walker_topology)
    if any(v is not None for v in contact_args) and not all(v is not None for v in contact_args):
        raise ValueError("contact mode requires contact-labels, scene-transforms, contact-vertex-sets and walker-topology together")
    if all(v is not None for v in contact_args) and args.surface_hand_contact_weight <= 0.0:
        raise ValueError("contact inputs supplied but --surface-hand-contact-weight is zero")

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
    if not np.isfinite(np.asarray(quality, dtype=np.float32)).all():
        raise ValueError("body quality contains non-finite values")
    # Each camera file contains both anatomical sides. Fit each hand from both
    # cameras; a single camera must not be mistaken for an anatomical hand.
    image_size = (int(cal.image_width), int(cal.image_height))
    hand_ll, hand_ll_valid, hand_ll_bounds = read_wilor(args.wilor_left, n, "left", left, image_size)
    hand_lr, hand_lr_valid, hand_lr_bounds = read_wilor(args.wilor_left, n, "right", left, image_size)
    hand_rl, hand_rl_valid, hand_rl_bounds = read_wilor(args.wilor_right, n, "left", right, image_size)
    hand_rr, hand_rr_valid, hand_rr_bounds = read_wilor(args.wilor_right, n, "right", right, image_size)
    hand_l, hand_l_valid, hand_l_bounds = hand_ll, hand_ll_valid, hand_ll_bounds
    hand_r, hand_r_valid, hand_r_bounds = hand_rr, hand_rr_valid, hand_rr_bounds
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
    vposer, vp_cfg, vp_ckpt = load_vposer_explicit(args.vposer_dir, args.device)
    latent_dim = int(vp_cfg.model_params.latentD)
    if latent_dim != 32:
        raise ValueError(f"VPoser latentD={latent_dim}, expected the installed V02_05 latentD=32")
    beta = torch.nn.Parameter(torch.zeros(1, 10, device=device))
    root = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    transl = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    latent = torch.nn.Parameter(torch.zeros(n, latent_dim, device=device))
    lhand = torch.nn.Parameter(torch.zeros(n, 45, device=device))
    rhand = torch.nn.Parameter(torch.zeros(n, 45, device=device))
    target = torch.tensor(tri_m, device=device)
    obs_ll = torch.tensor(np.nan_to_num(hand_ll, nan=0.0), device=device)
    obs_lr = torch.tensor(np.nan_to_num(hand_lr, nan=0.0), device=device)
    obs_rl = torch.tensor(np.nan_to_num(hand_rl, nan=0.0), device=device)
    obs_rr = torch.tensor(np.nan_to_num(hand_rr, nan=0.0), device=device)
    mask_body = torch.tensor(body_mask, device=device)
    mask_ll = torch.tensor(hand_ll_valid, device=device); mask_lr = torch.tensor(hand_lr_valid, device=device)
    mask_rl = torch.tensor(hand_rl_valid, device=device); mask_rr = torch.tensor(hand_rr_valid, device=device)
    weight_ll = torch.tensor(hand_ll_valid.astype(np.float32) * np.where(hand_ll_bounds, 1.0, 0.1), device=device)
    weight_lr = torch.tensor(hand_lr_valid.astype(np.float32) * np.where(hand_lr_bounds, 1.0, 0.1), device=device)
    weight_rl = torch.tensor(hand_rl_valid.astype(np.float32) * np.where(hand_rl_bounds, 1.0, 0.1), device=device)
    weight_rr = torch.tensor(hand_rr_valid.astype(np.float32) * np.where(hand_rr_bounds, 1.0, 0.1), device=device)
    hand_order_weights_np = np.asarray(
        [1.00, 0.95, 0.82, 0.68, 0.48,
         0.95, 0.82, 0.68, 0.48, 0.95, 0.82, 0.68, 0.48,
         0.95, 0.82, 0.68, 0.48, 0.95, 0.82, 0.68, 0.48], np.float32)
    order_w = torch.tensor(hand_order_weights_np, device=device)[None, :]
    weight_ll = weight_ll * order_w; weight_lr = weight_lr * order_w
    weight_rl = weight_rl * order_w; weight_rr = weight_rr * order_w
    contact_enabled = args.contact_labels is not None
    contact_hand_weight = contact_handle = contact_R = contact_T = palm_idx = None
    if contact_enabled:
        labels = np.load(args.contact_labels, allow_pickle=True)
        scene = np.load(args.scene_transforms, allow_pickle=True)
        if int(labels["handle_ends_ground_m"].shape[0]) != n:
            raise ValueError("contact labels frame count does not match raw input")
        if tuple(labels["hand_contact_weight"].shape) != (n, 2):
            raise ValueError("hand_contact_weight must have shape (frames,2)")
        if tuple(labels["handle_ends_ground_m"].shape) != (n, 2, 2, 3):
            raise ValueError("handle_ends_ground_m must have shape (frames,2,2,3)")
        if tuple(scene["rotation_ground_from_left"].shape) != (n, 3, 3):
            raise ValueError("rotation_ground_from_left must have shape (frames,3,3)")
        if tuple(scene["translation_ground_from_left_mm"].shape) != (n, 3):
            raise ValueError("translation_ground_from_left_mm must have shape (frames,3)")
        if not np.isfinite(labels["hand_contact_weight"]).all() or np.any(labels["hand_contact_weight"] < 0):
            raise ValueError("hand_contact_weight must be finite and non-negative")
        if not np.isfinite(labels["handle_ends_ground_m"]).all():
            raise ValueError("handle_ends_ground_m contains non-finite values")
        contact_handle = torch.tensor(labels["handle_ends_ground_m"], dtype=torch.float32, device=device)
        contact_hand_weight = torch.tensor(labels["hand_contact_weight"], dtype=torch.float32, device=device)
        contact_R = torch.tensor(scene["rotation_ground_from_left"], dtype=torch.float32, device=device)
        contact_T = torch.tensor(scene["translation_ground_from_left_mm"] / 1000.0, dtype=torch.float32, device=device)
        sets = json.loads(args.contact_vertex_sets.read_text(encoding="utf-8"))["sets"]
        topology = json.loads(args.walker_topology.read_text(encoding="utf-8"))
        handles = topology.get("handle_segments", {})
        if set(handles) != {"left", "right"} or any(len(v) != 2 for v in handles.values()):
            raise ValueError("walker topology must expose exactly two endpoints for left/right handles")
        palm_idx = {s: torch.tensor(sum(sets[f"{s}_palm_surface_candidate"].values(), []), dtype=torch.long, device=device)
                    for s in ("left", "right")}
        if any(int(v.numel()) == 0 for v in palm_idx.values()):
            raise ValueError("contact vertex sets contain no palm vertices")
        if any(int(v.min()) < 0 or int(v.max()) >= 6890 for v in palm_idx.values()):
            raise ValueError("contact vertex index is outside the 6890-vertex SMPL-H topology")
        if float(np.asarray(labels["hand_contact_weight"]).sum()) <= 0.0:
            raise ValueError("contact mode requested but all hand_contact_weight values are zero")
    R01 = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=device)
    T01 = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=device)
    K0, D0 = cal.K0, cal.D0
    K1, D1 = cal.K1, cal.D1

    # Translation init with NaN guard (no nanmean->nan_to_num silence).
    with torch.no_grad():
        z = model(betas=torch.zeros(n, 10, device=device),
                  global_orient=root, body_pose=torch.zeros(n, 63, device=device),
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

    def decode_body_pose(z):
        pose = vposer.decode(z)["pose_body"].reshape(n, 63)
        if pose.shape != (n, 63) or not bool(torch.isfinite(pose).all()):
            raise RuntimeError(f"VPoser decoded pose has invalid shape/values: {tuple(pose.shape)}")
        return pose
    with torch.no_grad():
        _ = decode_body_pose(latent)
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
    default_stage = max(1, args.steps // 6)
    stage_steps = [x if x is not None else default_stage for x in
                   (args.base_steps, args.beta_steps, args.joint_steps,
                    args.hand_steps, args.contact_steps, args.contact_refine_steps)]
    stage_names = ["A_body_vposer", "B_shared_beta", "C_body_vposer_refine",
                   "D1_hand_proximal", "D2_hand_surface_contact", "D3_contact_refine"]
    stage_train = [["root", "transl", "latent"], ["beta"],
                   ["beta", "root", "transl", "latent"], ["lhand", "rhand"],
                   ["lhand", "rhand"], ["lhand", "rhand", "root", "transl"]]
    def set_stage(train):
        for p in (beta, root, transl, latent, lhand, rhand): p.requires_grad_(False)
        m = {"beta": beta, "root": root, "transl": transl, "latent": latent, "lhand": lhand, "rhand": rhand}
        for k in train: m[k].requires_grad_(True)
    stage_history = []
    global_step = 0
    for stage_index, (stage_name, train_names, n_stage) in enumerate(zip(stage_names, stage_train, stage_steps)):
        set_stage(train_names)
        lr_scale = {0: 1.0, 1: 0.10, 2: 0.25, 3: 0.50, 4: 0.25, 5: 0.10}[stage_index]
        stage_params = [m[k] for k in train_names]
        optim = torch.optim.Adam(stage_params, lr=args.lr * lr_scale)
        for local_step in range(n_stage):
            step = global_step
            global_step += 1
            optim.zero_grad(set_to_none=True)
            body_pose = decode_body_pose(latent)
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
            # Both cameras constrain each anatomical hand. This avoids the
            # old camera-side/anatomical-side mix-up.
            hand_ll_res = (pl - obs_ll).pow(2).sum(-1)
            hand_rl_res = (pr - obs_rl).pow(2).sum(-1)
            hand_lr_res = (pl - obs_lr).pow(2).sum(-1)
            hand_rr_res = (pr - obs_rr).pow(2).sum(-1)
            def weighted(res, w, mask):
                return (res * w).sum() / w.sum().clamp_min(1e-6) if mask.any() else res.sum() * 0.0
            hand_l_loss = weighted(hand_ll_res, weight_ll, mask_ll) + weighted(hand_rl_res, weight_rl, mask_rl)
            hand_r_loss = weighted(hand_lr_res, weight_lr, mask_lr) + weighted(hand_rr_res, weight_rr, mask_rr)
            pose_reg = args.vposer_prior_weight * latent.pow(2).mean() + 1e-4 * (lhand.pow(2).mean() + rhand.pow(2).mean())
            if stage_index in (3, 4):
                body_term = body_loss.detach() * 0.0
            elif stage_index == 5:
                # D3 opens only root/translation and keeps a low-strength
                # COCO guardrail so contact cannot translate the whole body.
                body_term = 0.10 * body_loss
            else:
                body_term = body_loss
            # Pixel residuals are numerically much larger than metre-scale
            # body residuals. Keep hand fitting auxiliary and prevent it from
            # moving the body/root to explain WiLoR's model-derived pixels.
            hand_term = 1e-7 * (hand_l_loss + hand_r_loss) if stage_index >= 3 else (hand_l_loss + hand_r_loss).detach() * 0.0
            contact_loss = pl.sum() * 0.0
            if contact_enabled and stage_index >= 4:
                vg = torch.einsum("nij,nvj->nvi", contact_R, out.vertices) + contact_T[:, None, :]
                terms = []
                for side, j in (("left", 0), ("right", 1)):
                    pts = vg[:, palm_idx[side], :]
                    a, b = contact_handle[:, j, 0, :], contact_handle[:, j, 1, :]
                    terms.append(surface_contact.hand_surface_loss(pts, a[:, None, :], b[:, None, :], 0.016)
                                 * contact_hand_weight[:, j])
                contact_loss = torch.stack(terms, dim=1).sum() / contact_hand_weight.sum().clamp_min(1e-6)
            loss = body_term + hand_term + pose_reg + 1e-3 * beta.pow(2).mean() + \
                   (args.surface_hand_contact_weight * contact_loss if stage_index >= 4 else contact_loss.detach() * 0.0)
            loss.backward()
            optim.step()
            if step % 10 == 0 or step == sum(stage_steps) - 1:
                row = {"stage": stage_name, "stage_step": local_step, "step": step, "loss": float(loss.detach()),
                            "body_m": float(torch.sqrt(body_loss.detach())),
                            "hand_l_px": float(torch.sqrt(hand_l_loss.detach())),
                            "hand_r_px": float(torch.sqrt(hand_r_loss.detach())),
                            "hand_l_points": int(mask_ll.sum() + mask_rl.sum()), "hand_r_points": int(mask_lr.sum() + mask_rr.sum()),
                            "contact": float(contact_loss.detach()),
                            "contact_active": bool(contact_enabled and stage_index >= 4)}
                history.append(row)
        stage_history.append({"stage": stage_name, "steps": n_stage, "trainable": train_names,
                              "hand_3d_term": False,
                              "learning_rate": float(args.lr * lr_scale),
                              "hand_priority": "wrist_to_distal_fixed_weights" if stage_index >= 3 else "inactive",
                              "contact_term": bool(contact_enabled and stage_index >= 4)})

    with torch.no_grad():
        body_pose = decode_body_pose(latent)
        final = model(betas=beta.expand(n, -1), global_orient=root,
                      body_pose=body_pose, left_hand_pose=lhand,
                      right_hand_pose=rhand, transl=transl, return_verts=True)
        final_hand_l = hand21(final.joints, final.vertices, "left")
        final_hand_r = hand21(final.joints, final.vertices, "right")
        contact_diag = {}
        if contact_enabled:
            vg = torch.einsum("nij,nvj->nvi", contact_R, final.vertices) + contact_T[:, None, :]
            for side, j in (("left", 0), ("right", 1)):
                a, b = contact_handle[:, j, 0, :], contact_handle[:, j, 1, :]
                contact_diag[f"{side}_palm_ground_m"] = vg[:, palm_idx[side], :].cpu().numpy()
                contact_diag[f"{side}_hand_capsule_residual_m"] = surface_contact.capsule_surface_residual(
                    vg[:, palm_idx[side], :], a[:, None, :], b[:, None, :], 0.016).cpu().numpy()
            contact_diag["hand_contact_weight"] = contact_hand_weight.cpu().numpy()
    np.savez_compressed(args.output_dir / "result.npz",
                        vertices=final.vertices.cpu().numpy(), faces=np.asarray(model.faces),
                        predicted_coco=regress_coco17_torch(final.vertices, reg).cpu().numpy(),
                        betas=beta.detach().cpu().numpy(), global_orient=root.detach().cpu().numpy(),
                        transl=transl.detach().cpu().numpy(),
                        body_pose=body_pose.detach().cpu().numpy(),
                        vposer_latent=latent.detach().cpu().numpy(),
                        left_hand_pose=lhand.detach().cpu().numpy(), right_hand_pose=rhand.detach().cpu().numpy(),
                        raw_triangulated_points=tri, body_accepted=body_mask,
                        wilor_left_2d=hand_l, wilor_left_mask=hand_l_mask,
                        wilor_right_2d=hand_r, wilor_right_mask=hand_r_mask,
                        wilor_left_camera_left_2d=hand_ll, wilor_left_camera_right_2d=hand_rl,
                        wilor_right_camera_left_2d=hand_lr, wilor_right_camera_right_2d=hand_rr,
                        wilor_left_camera_left_mask=hand_ll_valid, wilor_left_camera_right_mask=hand_rl_valid,
                        wilor_right_camera_left_mask=hand_lr_valid, wilor_right_camera_right_mask=hand_rr_valid,
                        wilor_left_2d_full=hand_l_full, wilor_right_2d_full=hand_r_full,
                        wilor_left_valid_full=hand_l_full_valid, wilor_right_valid_full=hand_r_full_valid,
                        wilor_left_bounds_ok_full=hand_l_full_bounds, wilor_right_bounds_ok_full=hand_r_full_bounds,
                        body_quality=quality,
                        translation_init_source=np.asarray(init_source, dtype="U32"),
                        wilor_left_weight=hand_l_weight_np, wilor_right_weight=hand_r_weight_np,
                        smplh_joints=final.joints.detach().cpu().numpy(),
                        hand_points_left=final_hand_l.detach().cpu().numpy(),
                        hand_points_right=final_hand_r.detach().cpu().numpy(), **contact_diag)
    (args.output_dir / "fit_summary.json").write_text(json.dumps({
        "status": "engineering_candidate", "frames": n, "steps": int(sum(stage_steps)),
        "model": str(args.smplh_model.resolve()), "vertices": 6890,
        "hand_observation_source": "WiLoR_model_projected_MANO_joints",
        "hand_observation_points_retained": 21,
        "hand_fit_subset_points": 21,
        "hand_bounds_are_diagnostic_only": True,
        "wilor_left_valid_points": int(hand_ll_valid.sum() + hand_rl_valid.sum()),
        "wilor_right_valid_points": int(hand_lr_valid.sum() + hand_rr_valid.sum()), "history": history,
        "wilor_observation_views": {"left_camera": ["left_hand", "right_hand"],
                                     "right_camera": ["left_hand", "right_hand"]},
        "stage_schedule": stage_history,
        "hand_confidence_source": "detector_box_or_missing_neutral_weight",
        "vposer_checkpoint": str(vp_ckpt), "vposer_latent_dim": latent_dim,
        "hand_priority_weights": hand_order_weights_np.tolist(),
        "contact_enabled": bool(contact_enabled),
        "contact_schedule": ["D1 hand observation", "D2 surface hand contact", "D3 limited contact refinement"],
        "hand_3d_observation_used": False,
        "body_accepted_points": int(body_mask.sum()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
