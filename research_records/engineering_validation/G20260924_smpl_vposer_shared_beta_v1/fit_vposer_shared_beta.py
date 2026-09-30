#!/usr/bin/env python3
"""Single-frame VPoser SMPL fits with one beta shared by the whole clip.

The raw PMPose files are triangulated in this run. No stored triangulation,
temporal prefit, fit parameters, contact targets or HTML are read.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "realtime_app"
sys.path.insert(0, str(APP))

import cv2
from pose_app.fisheye_camera import fisheye_project_numpy, fisheye_project_torch, load_stereo_fisheye
from pose_app.smpl_coco_observation import load_coco17_regressor, load_smpl_male, regress_coco17_torch
from pose_app.smplx_fitting import load_vposer_explicit

RAW = ROOT / "research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py"
import importlib.util
spec = importlib.util.spec_from_file_location("raw_clean", RAW)
raw_clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_clean)
raw_clean.cv2 = cv2

NAMES = ("nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
         "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
         "left_knee","right_knee","left_ankle","right_ankle")


def resolve_scene_frame_mask(st_accepted_full, wsl, n, tri_any):
    """Fail-closed scene-accepted mask for temporal triplets (pure numpy).

    Returns a boolean (n,) frame mask. Raises ValueError when the field is
    missing, has wrong shape, or holds abnormal values; never silently
    returns all-True, which would fabricate a temporal pass.
    """
    if st_accepted_full is None:
        raise ValueError("scene transforms carry no 'accepted' field: temporal mask unavailable")
    arr = np.asarray(st_accepted_full)
    try:
        win = arr[wsl]
    except Exception as exc:
        raise ValueError(f"scene 'accepted' window slice failed: {exc}")
    if win.ndim == 2:
        if win.shape[1] != 17:
            raise ValueError(f"scene 'accepted' joint dim {win.shape[1]} != 17")
        flat = np.asarray(win).reshape(-1)
        if flat.dtype != bool and not np.isin(flat, [0, 1]).all():
            raise ValueError("scene 'accepted' holds values outside {0,1}")
        mask = win.any(axis=-1)
    elif win.ndim == 1:
        mask = win
    else:
        raise ValueError(f"scene 'accepted' ndim {win.ndim} not in (1,2)")
    mask = np.asarray(mask).reshape(-1)
    if mask.shape[0] != n:
        raise ValueError(f"scene 'accepted' window frames {mask.shape[0]} != {n}")
    if mask.dtype != bool:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("scene 'accepted' holds values outside {0,1}")
        mask = mask.astype(bool)
    if not np.isfinite(np.asarray(tri_any, dtype=float)).all():
        raise ValueError("triangulation frame mask is non-finite")
    return mask


def aa_from_matrix(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(m).as_rotvec().astype(np.float32)


def orientation_init(target: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Align model shoulder/pelvis basis to the observed triangulated basis."""
    pelvis = target[[11, 12]].mean(0)
    up = target[[5, 6]].mean(0) - pelvis
    lat = target[6] - target[5]
    up /= max(np.linalg.norm(up), 1e-8)
    lat -= up * np.dot(lat, up)
    lat /= max(np.linalg.norm(lat), 1e-8)
    dep = np.cross(lat, up); dep /= max(np.linalg.norm(dep), 1e-8)
    target_basis = np.stack([lat, up, dep], axis=1)
    m = target_basis @ local.T
    if np.linalg.det(m) < 0: m[:, 2] *= -1
    return aa_from_matrix(m)


def robust_norm(x, delta):
    r = torch.linalg.vector_norm(x, dim=-1)
    return delta * delta * (torch.sqrt(1.0 + (r / delta) ** 2) - 1.0)


def robust_scalar(x, delta):
    return delta * delta * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--regressor", type=Path, required=True)
    ap.add_argument("--vposer-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-steps", type=int, default=120)
    ap.add_argument("--beta-steps", type=int, default=120)
    ap.add_argument("--joint-steps", type=int, default=120)
    ap.add_argument("--contact-labels", type=Path, default=None)
    ap.add_argument("--scene-transforms", type=Path, default=None)
    ap.add_argument("--foot-contact-weight", type=float, default=0.0)
    ap.add_argument("--hand-contact-weight", type=float, default=0.0)
    ap.add_argument("--contact-steps", type=int, default=80)
    ap.add_argument("--stage-d-trace", action="store_true",
                    help="Write per-step Stage D loss terms for convergence diagnosis")
    ap.add_argument("--stage-d-lr-schedule", choices=("fixed", "cosine", "exponential"), default="fixed",
                    help="Stage D learning-rate schedule; default preserves the historical fixed LR")
    ap.add_argument("--stage-d-lr-min-factor", type=float, default=0.10,
                    help="Final/target LR factor for cosine or exponential Stage D decay")
    ap.add_argument("--obs-3d-weight", type=float, default=1.0)
    ap.add_argument("--obs-3d-mode", choices=("uniform", "reprojection_uncertainty"), default="uniform",
                    help="3-D observation model; uniform preserves the frozen baseline")
    ap.add_argument("--obs-3d-base-sigma-mm", type=float, default=100.0,
                    help="base 3-D robust scale for reprojection_uncertainty mode")
    ap.add_argument("--obs-3d-max-sigma-factor", type=float, default=4.0,
                    help="upper clamp for the uncertainty scale factor")
    # Gradient audit (v5 summary): with 2D=0.25 the realized 3D/2D gradient
    # L2 norms were nearly equal (0.327/0.328). Lowering 2D to 0.20 puts the
    # realized 3D gradient ~1.25x above 2D ("3D slightly dominant").
    ap.add_argument("--obs-2d-weight", type=float, default=0.20)
    ap.add_argument("--stage-d-obs-3d-scale", type=float, default=0.70)
    ap.add_argument("--stage-d-obs-2d-scale", type=float, default=0.10)
    ap.add_argument("--force-stage-d-no-contact", action="store_true",
                    help="Stage D control: same optimizer/steps/frozen beta but zero surface weights")
    ap.add_argument("--gradient-audit-only", action="store_true",
                    help="Stop after Stage C and audit term gradients; never run Stage D")
    ap.add_argument("--contact-vertex-sets", type=Path, default=None)
    ap.add_argument("--walker-topology", type=Path, default=None)
    ap.add_argument("--surface-foot-contact-weight", type=float, default=0.0)
    ap.add_argument("--surface-hand-contact-weight", type=float, default=0.0)
    ap.add_argument("--surface-foot-nonpenetration-weight", type=float, default=0.0,
                    help="Stage-D foot surface non-penetration weight")
    ap.add_argument("--surface-foot-tangential-weight", type=float, default=0.0,
                    help="Stage-D fixed sole-vertex tangential speed weight; support labels only")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--temporal-mode", choices=("none", "stage_d", "stage_c_and_d"), default="none")
    ap.add_argument("--temporal-local-weight", type=float, default=0.0)
    ap.add_argument("--temporal-root-weight", type=float, default=0.0)
    ap.add_argument("--temporal-huber-scale-mm", type=float, default=30.0)
    ap.add_argument("--temporal-max-gap-frames", type=int, default=1)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be supplied together")
    if args.hand_contact_weight < 0.0 or args.surface_hand_contact_weight < 0.0:
        raise ValueError("hand contact weights must be non-negative")
    if args.surface_foot_nonpenetration_weight < 0 or args.surface_foot_tangential_weight < 0:
        raise ValueError("surface foot kinematic weights must be non-negative")
    if not math.isfinite(args.obs_3d_base_sigma_mm) or not math.isfinite(args.obs_3d_max_sigma_factor) or args.obs_3d_base_sigma_mm <= 0.0 or args.obs_3d_max_sigma_factor < 1.0:
        raise ValueError("3-D uncertainty scale must be positive and max factor >= 1")
    surface_mode = args.contact_vertex_sets is not None
    eff_foot_w, eff_hand_w = args.surface_foot_contact_weight, args.surface_hand_contact_weight
    if surface_mode and (args.walker_topology is None or args.contact_labels is None or args.scene_transforms is None):
        raise ValueError("surface mode needs --walker-topology, --contact-labels and --scene-transforms together")
    if surface_mode and args.foot_contact_weight != 0.0:
        raise ValueError("surface mode must not use COCO ankle contact: --foot-contact-weight must be 0.0")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refuse non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    global torch
    import torch
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")

    cal = load_stereo_fisheye(args.calibration_dir)
    reg = load_coco17_regressor(args.regressor)
    left_rows, right_rows = raw_clean.raw_side(args.left, "left"), raw_clean.raw_side(args.right, "right")
    ids = sorted(set(left_rows) & set(right_rows))
    if ids != list(range(len(ids))): raise ValueError("raw frame ids are not contiguous")
    left = np.stack([left_rows[i] for i in ids]).astype(np.float32)
    right = np.stack([right_rows[i] for i in ids]).astype(np.float32)
    raw, el, er, gap, dl, dr, accepted, reason, q, parts = raw_clean.raw_triangulate(left, right, cal)
    full_frames = len(ids)
    if args.start is not None:
        if not (0 <= args.start <= args.end < full_frames):
            raise ValueError(f"invalid window [{args.start},{args.end}] for {full_frames} frames")
        sl = slice(args.start, args.end + 1)
        left, right = left[sl], right[sl]
        raw, el, er = raw[sl], el[sl], er[sl]
        gap, dl, dr = gap[sl], dl[sl], dr[sl]
        accepted, reason = accepted[sl], reason[sl]
        q = q[sl]
        parts = tuple(p[sl] for p in parts)
        ids = list(range(args.start, args.end + 1))
    window = [args.start, args.end] if args.start is not None else [0, full_frames - 1]
    # raw_triangulate receives dictionaries and preserves all failures. Missing
    # values remain NaN; only finite accepted observations enter the 3-D term.
    tri = np.asarray(raw, dtype=np.float32) / 1000.0
    np.savez_compressed(out / "raw_observations.npz", left_2d=left, right_2d=right)
    np.savez_compressed(out / "triangulation.npz", raw=raw, left_reprojection_error=el,
                        right_reprojection_error=er, ray_gap=gap, depth_left=dl,
                        depth_right=dr, accepted=accepted, confidence=q,
                        confidence_2d=parts[0], confidence_ray=parts[1],
                        confidence_reproj=parts[2], confidence_sync=parts[3],
                        confidence_depth=parts[4], reject_reason=reason)

    n = len(ids)
    model = load_smpl_male(args.model, batch_size=n, device=args.device, dtype=torch.float32)
    vposer, vp_cfg, vp_ckpt = load_vposer_explicit(args.vposer_dir, args.device)
    zero_beta = torch.zeros((n, 10), dtype=torch.float32, device=device)
    zero_pose = torch.zeros((n, 69), dtype=torch.float32, device=device)
    zero_out = model(betas=zero_beta, body_pose=zero_pose,
                     global_orient=torch.zeros((n,3), device=device),
                     transl=torch.zeros((n,3), device=device), return_verts=True)
    zero_coco = regress_coco17_torch(zero_out.vertices, reg).detach().cpu().numpy()
    local_basis = np.eye(3, dtype=np.float32)
    # Build a local basis from the zero-pose model, once.
    lp = zero_coco[0][[11,12]].mean(0); lu = zero_coco[0][[5,6]].mean(0)-lp; ll = zero_coco[0][6]-zero_coco[0][5]
    lu /= max(np.linalg.norm(lu),1e-8); ll -= lu*np.dot(ll,lu); ll /= max(np.linalg.norm(ll),1e-8)
    ld=np.cross(ll,lu); ld/=max(np.linalg.norm(ld),1e-8); local_basis=np.stack([ll,lu,ld],axis=1)
    target_np = np.nan_to_num(tri, nan=0.0)
    # q already combines the existing 2-D confidence, ray-gap, reprojection,
    # sync and depth factors. The candidate experiment keeps q unchanged and
    # adds only a per-point uncertainty scale derived from reprojection error
    # and ray gap, avoiding a second copy of the same confidence product.
    reproj_px = 0.5 * (np.asarray(el, dtype=np.float32) + np.asarray(er, dtype=np.float32))
    ray_gap_mm = np.asarray(gap, dtype=np.float32)
    finite_quality = np.isfinite(reproj_px) & np.isfinite(ray_gap_mm)
    if args.obs_3d_mode != "uniform" and np.any(accepted & ~finite_quality):
        raise ValueError("accepted 3-D candidate has unavailable uncertainty diagnostics")
    sigma_factor_np = np.ones_like(q, dtype=np.float32)
    sigma_factor_np[finite_quality] = np.clip(
        1.0 + reproj_px[finite_quality] / 8.0 + ray_gap_mm[finite_quality] / 30.0,
        1.0, float(args.obs_3d_max_sigma_factor))
    sigma_factor_np[~np.isfinite(sigma_factor_np)] = float(args.obs_3d_max_sigma_factor)
    sigma_factor_np[~accepted] = 1.0
    if args.obs_3d_mode == "uniform":
        sigma_np = np.full_like(q, 0.10, dtype=np.float32)
    else:
        sigma_np = (float(args.obs_3d_base_sigma_mm) / 1000.0) * sigma_factor_np
    root0=np.stack([orientation_init(target_np[t], local_basis) for t in range(n)])
    pelvis_target=np.nanmean(target_np[:,[11,12]],axis=1)
    model_pelvis=zero_coco[0][[11,12]].mean(0)
    trans0=pelvis_target-model_pelvis[None,:]
    root=torch.nn.Parameter(torch.tensor(root0, device=device))
    transl=torch.nn.Parameter(torch.tensor(trans0, device=device))
    latent=torch.nn.Parameter(torch.zeros((n, int(vp_cfg.model_params.latentD)), device=device))
    beta=torch.nn.Parameter(torch.zeros((1,10), device=device))
    obs_l=torch.tensor(left[:,:,:2],device=device); obs_r=torch.tensor(right[:,:,:2],device=device)
    target=torch.tensor(target_np,device=device)
    w=torch.tensor(q,dtype=torch.float32,device=device)
    sigma3d=torch.tensor(sigma_np,dtype=torch.float32,device=device)
    contact_enabled = False
    if (args.contact_labels is None) != (args.scene_transforms is None):
        raise ValueError("--contact-labels and --scene-transforms must be supplied together")
    if args.contact_labels is not None:
        cl = np.load(args.contact_labels, allow_pickle=True)
        st = np.load(args.scene_transforms, allow_pickle=True)
        if int(cl["foot_contact_weight"].shape[0]) != full_frames or int(st["rotation_ground_from_left"].shape[0]) != full_frames:
            raise ValueError("contact/scene frame count does not match full raw input")
        wsl = slice(window[0], window[1] + 1)
        # hand_contact_weight / hand_label are audit-only inputs: they never
        # gate the surface hand loss (which applies to every frame/side).
        support_labels = np.asarray(cl["foot_label"])[wsl]
        support_mask_np = support_labels == "support"
        support_mask = torch.tensor(support_mask_np.astype(np.float32), dtype=torch.float32, device=device)
        hand_w_stats = torch.tensor(np.asarray(cl["hand_contact_weight"])[wsl].astype(np.float32),
                                    dtype=torch.float32, device=device)
        hand_label_stats = np.asarray(cl["hand_label"])[wsl]
        hand_score_stats = np.asarray(cl["hand_candidate_score"])[wsl]
        Rgc = torch.tensor(np.asarray(st["rotation_ground_from_left"])[wsl], dtype=torch.float32, device=device)
        Tgc = torch.tensor(np.asarray(st["translation_ground_from_left_mm"])[wsl] / 1000.0, dtype=torch.float32, device=device)
        foot_w = torch.tensor(np.asarray(cl["foot_contact_weight"])[wsl], dtype=torch.float32, device=device)
        hand_w = torch.zeros_like(foot_w)
        handle_ends = torch.nan_to_num(torch.tensor(np.asarray(cl["handle_ends_ground_m"])[wsl], dtype=torch.float32, device=device), nan=0.0)
    else:
        Rgc = Tgc = foot_w = hand_w = handle_ends = None
    R01=torch.tensor(cal.R_cam0_to_cam1,dtype=torch.float32,device=device)
    t01=torch.tensor(cal.T_cam0_to_cam1_mm/1000.0,dtype=torch.float32,device=device)
    K0=torch.tensor(cal.K0,dtype=torch.float32,device=device); D0=torch.tensor(cal.D0,dtype=torch.float32,device=device)
    K1=torch.tensor(cal.K1,dtype=torch.float32,device=device); D1=torch.tensor(cal.D1,dtype=torch.float32,device=device)
    TEMPORAL_LOCAL_IDX = [5,6,7,8,9,10,11,12,13,14,15,16]
    temporal_enabled = [False]
    scale_m = float(args.temporal_huber_scale_mm) / 1000.0
    _acc_np = np.asarray(accepted)
    _acc_any = _acc_np.any(axis=-1) if _acc_np.ndim == 2 else _acc_np
    # Fail-closed: missing/malformed scene 'accepted' raises instead of
    # silently enabling all triplets (which would fabricate a temporal pass).
    _st_src = (st["accepted"] if ("st" in locals() and st is not None and "accepted" in st)
               else None)
    # Without scene transforms temporal stays unavailable (frame_ok=zeros
    # below); the fail-closed resolver only runs when Rgc exists.
    _st_acc = (np.ones((n,), dtype=bool) if Rgc is None else resolve_scene_frame_mask(
        _st_src, wsl, n, _acc_any))
    if Rgc is not None:
        Rgc_finite = torch.isfinite(Rgc).all(dim=(1,2)) & torch.isfinite(Tgc).all(dim=1)
        frame_ok = torch.tensor(np.asarray(_acc_any, dtype=bool) & _st_acc, device=device) & Rgc_finite
    else:
        frame_ok = torch.zeros((n,), dtype=torch.bool, device=device)
    frame_ok_np = frame_ok.detach().cpu().numpy()
    def temporal_terms(jc):
        z = torch.zeros((), device=device)
        if args.temporal_mode == "none" or not temporal_enabled[0] or Rgc is None or n < 3:
            return z, z, {"local_valid":0,"root_valid":0,"rejected_triplets":0,"gap_triplets":0,"status":"unavailable"}
        dt = 1.0/30.0
        jg = torch.einsum("nij,nvj->nvi", Rgc, jc) + Tgc[:,None,:]
        pelvis = 0.5*(jg[:,11]+jg[:,12])
        local = jg[:,TEMPORAL_LOCAL_IDX,:] - pelvis[:,None,:]
        rej=0; gapc=0; la=[]; ra=[]
        for t in range(1,n-1):
            ok = bool(frame_ok_np[t-1] and frame_ok_np[t] and frame_ok_np[t+1])
            if not ok:
                rej+=1; continue
            if 1 > int(args.temporal_max_gap_frames):
                gapc+=1; continue
            a = (local[t+1]-2*local[t]+local[t-1])/(dt*dt)
            r = (pelvis[t+1]-2*pelvis[t]+pelvis[t-1])/(dt*dt)
            la.append(a); ra.append(r)
        info={"status":"ok" if la else "unavailable"}
        if not la:
            info.update({"local_valid":0,"root_valid":0,"rejected_triplets":rej,"gap_triplets":gapc})
            return z, z, info
        A=torch.stack(la); R=torch.stack(ra)
        rho_l = torch.sqrt(1.0+(A/scale_m)**2)-1.0
        rho_r = torch.sqrt(1.0+(R/scale_m)**2)-1.0
        info.update({"local_valid":int(A.shape[0]),"root_valid":int(R.shape[0]),"rejected_triplets":int(rej),"gap_triplets":int(gapc)})
        return rho_l.mean(), rho_r.mean(), info

    from pose_app import smpl_surface_contact as surf
    surface = None
    if surface_mode:
        sets_doc = json.loads(args.contact_vertex_sets.read_text(encoding="utf-8"))
        if sets_doc.get("model_topology") != "male_SMPL_6890":
            raise ValueError("surface sets must target male SMPL 6890")
        sole_idx = {}
        for side in ("left", "right"):
            key = f"{side}_sole_surface_candidate"
            sole_idx[side] = np.concatenate([np.asarray(sets_doc["sets"][key][part], dtype=np.int64)
                                             for part in ("heel", "ball", "toe")])
        palm_idx = {side: np.asarray(sets_doc["sets"][f"{side}_palm_surface_candidate"]["palm_fingers"], dtype=np.int64)
                    for side in ("left", "right")}
        for name, idx in {**{f"{s}_sole": v for s, v in sole_idx.items()},
                          **{f"{s}_palm": v for s, v in palm_idx.items()}}.items():
            if idx.size == 0 or idx.min() < 0 or idx.max() > 6889 or not np.isfinite(idx).all():
                raise ValueError(f"surface set {name} has invalid SMPL indices")
        topo = json.loads(args.walker_topology.read_text(encoding="utf-8-sig"))
        nodes_cam = {k: np.asarray(v, dtype=np.float64) / 1000.0
                     for k, v in topo["nodes_left_camera_mm"].items()}
        segments = {"left": ("front_right_top", "rear_right_top"),
                    "right": ("front_left_top", "rear_left_top")}
        for side, (na, nb) in segments.items():
            assert topo["handle_segments"][side] == [na, nb], "handle segment contract changed"
        # No tube radius in the static topology: explicit engineering assumption.
        handle_radius_m = 0.016
        handle_assumption = ("handle_radius_m=0.016 is an engineering assumption for a ~32 mm "
                             "walker handle tube; the static topology JSON carries no radius field")
        Rgc_np = Rgc.detach().cpu().numpy().astype(np.float64)
        Tgc_np = Tgc.detach().cpu().numpy().astype(np.float64)
        capsules = {}
        for side, (na, nb) in segments.items():
            a = np.einsum("nij,j->ni", Rgc_np, nodes_cam[na]) + Tgc_np
            b = np.einsum("nij,j->ni", Rgc_np, nodes_cam[nb]) + Tgc_np
            capsules[side] = (torch.tensor(a, dtype=torch.float32, device=device),
                              torch.tensor(b, dtype=torch.float32, device=device))
        surface = {"sole_idx": {s: torch.tensor(v, dtype=torch.long, device=device) for s, v in sole_idx.items()},
                   "palm_idx": {s: torch.tensor(v, dtype=torch.long, device=device) for s, v in palm_idx.items()},
                   "capsules": capsules, "radius_m": handle_radius_m,
                   "radius_assumption": handle_assumption,
                   "sets_file": str(args.contact_vertex_sets.resolve()),
                   "topology_file": str(args.walker_topology.resolve())}

    def forward(b, z=latent):
        pose63=vposer.decode(z)["pose_body"].reshape(n,-1)
        pose69=torch.cat([pose63, torch.zeros((n,6),device=device)],dim=1)
        result=model(betas=b.expand(n,-1), body_pose=pose69, global_orient=root, transl=transl, return_verts=True)
        jc=regress_coco17_torch(result.vertices,reg)
        jr=torch.einsum("ij,nvj->nvi",R01,jc)+t01
        pl=fisheye_project_torch(jc,K0,D0); pr=fisheye_project_torch(jr,K1,D1)
        return result,pose69,jc,pl,pr

    def point_segment_distance(p, a, b):
        ab = b - a
        den = (ab * ab).sum(-1, keepdim=True).clamp_min(1e-8)
        u = ((p - a) * ab).sum(-1, keepdim=True) / den
        u = u.clamp(0.0, 1.0)
        return torch.linalg.vector_norm(p - (a + u * ab), dim=-1)

    valid2d=torch.isfinite(obs_l).all(-1)&torch.isfinite(obs_r).all(-1)
    surface_audit = {"coco_joints_in_surface_contact": False,
                     "contact_points": "surface_vertices_not_coco_joints"}
    def surface_terms(vertices):
        """Foot/hand surface contact using SMPL vertices only (no COCO joints)."""
        vg = torch.einsum("nij,nvj->nvi", Rgc, vertices) + Tgc[:, None, :]  # [N,6890,3] ground m
        left = vg[:, surface["sole_idx"]["left"], :]
        right = vg[:, surface["sole_idx"]["right"], :]
        foot_res = {"left": surf.signed_ground_distances(left),
                    "right": surf.signed_ground_distances(right)}
        lfoot_side = {s: surf.foot_surface_loss(foot_res[s]) for s in ("left", "right")}
        lfoot = lfoot_side["left"] * foot_w[:, 0] + lfoot_side["right"] * foot_w[:, 1]
        lfoot = lfoot.sum() / (foot_w.sum() + 1e-6)
        lhand = torch.zeros((), device=device)
        hand_cover = {}
        hand_res = {}
        hand_side_terms = {}
        for side_i, s in enumerate(("left", "right")):
            palm = vg[:, surface["palm_idx"][s], :]
            a, b = surface["capsules"][s]
            residuals = surf.capsule_surface_residual(palm, a[:, None, :], b[:, None, :], surface["radius_m"])
            hand_res[s] = residuals
            hand_cover[s] = surf.surface_coverage(residuals)
            hand_side_terms[s] = surf.hand_surface_loss(
                palm, a[:, None, :], b[:, None, :], surface["radius_m"]
            )
        # All-frame hand assumption: every frame and both sides enter the
        # hand loss with uniform normalization. No hand label/weight gating.
        lhand = (hand_side_terms["left"] + hand_side_terms["right"]).sum() / (2.0 * n)
        hand_mode_audit = {"hand_contact_mode": "all_frames_assumed",
                           "hand_labels_used_for_hand_loss": False,
                           "hand_contact_weight_gate": False,
                           "hand_frames_optimized": n, "hand_sides_optimized": 2}
        return lfoot, lhand, foot_res, hand_res, hand_cover, vg, hand_mode_audit

    def losses(b):
        result,pose,jc,pl,pr=forward(b)
        finite=torch.isfinite(target).all(-1)
        w3=w*finite
        if args.obs_3d_mode == "uniform":
            l3=((robust_norm(jc-target,0.10)/(0.10**2))*w3).sum()/(w3.sum()+1e-6)
        else:
            point_residual = torch.linalg.vector_norm(jc - target, dim=-1)
            point_loss = torch.sqrt(1.0 + (point_residual / sigma3d) ** 2) - 1.0
            l3=(point_loss*w3).sum()/(w3.sum()+1e-6)
        e2=robust_norm(pl-obs_l,20.0)+robust_norm(pr-obs_r,20.0)
        l2=(e2/(20.0**2)*w*valid2d).sum()/(2*(w*valid2d).sum()+1e-6)
        lp=latent.square().mean()
        lfoot=torch.zeros((),device=device); lhand=torch.zeros((),device=device)
        if surface_mode and (contact_enabled or audit_mode):
            lfoot, lhand, foot_res, hand_res_tensors, hand_cover, _, hand_mode = surface_terms(result.vertices)
            surface_audit.update(hand_mode)
            for s in ("left", "right"):
                if tuple(int(v) for v in foot_res[s].shape) != (n, 54):
                    raise ValueError(f"foot residual shape {tuple(foot_res[s].shape)} != ({n},54)")
                if tuple(int(v) for v in hand_res_tensors[s].shape) != (n, 778):
                    raise ValueError(f"hand residual shape {tuple(hand_res_tensors[s].shape)} != ({n},778)")
                if not bool(torch.isfinite(foot_res[s]).all()) or not bool(torch.isfinite(hand_res_tensors[s]).all()):
                    raise ValueError(f"non-finite surface residual for {s}")
            surface_audit["foot_residual_shapes"] = {s: [int(v) for v in foot_res[s].shape] for s in foot_res}
            surface_audit["hand_residual_shapes"] = {s: [int(v) for v in hand_res_tensors[s].shape] for s in hand_res_tensors}
            surface_audit["hand_vertex_index_shapes"] = {s: [int(v) for v in surface["palm_idx"][s].shape] for s in surface["palm_idx"]}
            surface_audit["foot_active_vertex_count"] = {s: int(surface["sole_idx"][s].numel()) for s in surface["sole_idx"]}
            surface_audit["hand_active_vertex_count"] = {s: int(surface["palm_idx"][s].numel()) for s in surface["palm_idx"]}
            surface_audit["handle_radius_m"] = surface["radius_m"]
            surface_audit["handle_radius_assumption"] = surface["radius_assumption"]
            surface_audit["hand_coverage"] = hand_cover
            surface_audit["observation_loss_units"] = "dimensionless_delta_normalized"
            surface_audit["stage_a_to_c_obs_3d_coeff"] = obs3d_abc
            surface_audit["stage_a_to_c_obs_2d_coeff"] = obs2d_abc
            surface_audit["stage_d_obs_3d_coeff"] = obs3d_d
            surface_audit["stage_d_obs_2d_coeff"] = obs2d_d
            surface_audit["foot_penetration_fraction"] = {
                s: float((foot_res[s] < 0).float().mean().detach().cpu()) for s in foot_res}
            surface_audit["hand_penetration_fraction"] = {
                s: float((hand_res_tensors[s] < 0).float().mean().detach().cpu()) for s in hand_res_tensors}
            surface_audit["hand_label_audit_only"] = {
                "hand_label_values": sorted(set(str(v) for v in np.asarray(hand_label_stats).ravel().tolist())),
                "hand_contact_weight_sum": float(hand_w_stats.sum()),
                "hand_contact_weight_nonzero_count": int((hand_w_stats > 0).sum()),
                "hand_candidate_score_median": float(np.median(np.asarray(hand_score_stats, dtype=np.float64))),
            }
        elif contact_enabled:
            jg=torch.einsum("nij,nvj->nvi", Rgc, jc) + Tgc[:,None,:]
            ankle_z=jg[:,[15,16],2]
            assert ankle_z.shape == foot_w.shape
            lfoot=(robust_scalar(ankle_z,0.025)*foot_w).sum()/(foot_w.sum()+1e-6)
            wrist=jg[:,[9,10],:]
            a=handle_ends[:,:,0,:]; bseg=handle_ends[:,:,1,:]
            hd=point_segment_distance(wrist,a,bseg)
            assert hd.shape == hand_w.shape
            lhand=(robust_scalar(hd,0.050)*hand_w).sum()/(hand_w.sum()+1e-6)
        foot_nonpenetration = torch.zeros((), device=device)
        foot_tangential = torch.zeros((), device=device)
        if surface_mode and (args.surface_foot_nonpenetration_weight > 0 or args.surface_foot_tangential_weight > 0):
            if n < 2:
                raise ValueError("foot kinematic terms require at least two frames")
            # Fixed sole vertices are used across adjacent frames. This avoids
            # measuring motion of a changing per-frame minimum vertex.
            vg = torch.einsum("nij,nvj->nvi", Rgc, result.vertices) + Tgc[:, None, :]
            z = torch.cat([vg[:, surface["sole_idx"]["left"], 2],
                           vg[:, surface["sole_idx"]["right"], 2]], dim=1)
            foot_nonpenetration = surf.nonpenetration_loss(z, margin_m=0.003).mean()
            if args.surface_foot_tangential_weight > 0:
                prev = torch.cat([vg[:-1, surface["sole_idx"]["left"], :],
                                  vg[:-1, surface["sole_idx"]["right"], :]], dim=1)
                curr = torch.cat([vg[1:, surface["sole_idx"]["left"], :],
                                  vg[1:, surface["sole_idx"]["right"], :]], dim=1)
                active = torch.cat([support_mask[:-1, 0:1].expand(-1, 54),
                                    support_mask[:-1, 1:2].expand(-1, 54)], dim=1)
                active = active * torch.cat([support_mask[1:, 0:1].expand(-1, 54),
                                             support_mask[1:, 1:2].expand(-1, 54)], dim=1)
                foot_tangential = surf.tangential_velocity_loss(curr, prev, 1.0 / 30.0,
                                                                 [0.0, 0.0, 1.0], active)
        tl=torch.zeros((),device=device); tr=torch.zeros((),device=device); tinfo={"local_valid":0,"root_valid":0,"rejected_triplets":0,"gap_triplets":0,"status":"unavailable"}
        if args.temporal_mode!="none" and temporal_enabled[0]:
            tl,tr,tinfo = temporal_terms(jc)
        return result,pose,jc,pl,pr,l3,l2,lp,lfoot,lhand,tl,tr,tinfo,foot_nonpenetration,foot_tangential

    def run(opt, steps, beta_reg=True, obs3d_coeff=1.0, obs2d_coeff=0.20, trace=None, scheduler=None):
        last=None
        for step in range(steps):
            opt.zero_grad(); vals=losses(beta); _,pose,_,_,_,l3,l2,lp,lfoot,lhand,tl,tr,tinfo,foot_np,foot_tv=vals
            trace_params = None
            if trace is not None:
                trace_params = {"latent": latent.detach().clone(), "root": root.detach().clone(), "transl": transl.detach().clone()}
            lb=beta.square().mean()
            if surface_mode:
                contact_loss = eff_foot_w*lfoot + eff_hand_w*lhand
            else:
                contact_loss = args.foot_contact_weight*lfoot + args.hand_contact_weight*lhand
            tloss = args.temporal_local_weight*tl + args.temporal_root_weight*tr
            loss=(obs3d_coeff*l3+obs2d_coeff*l2+0.02*lp
                  +(contact_loss if contact_enabled else 0.0)+(0.02*lb if beta_reg else 0.0)
                  +(tloss if temporal_enabled[0] else 0.0)
                  + args.surface_foot_nonpenetration_weight * foot_np
                  + args.surface_foot_tangential_weight * foot_tv)
            if trace is not None:
                trace.append({"step": step, "total": float(loss.detach()),
                              "obs3d": float(l3.detach()), "obs2d": float(l2.detach()),
                              "pose": float(lp.detach()), "foot": float(lfoot.detach()),
                              "hand": float(lhand.detach()),
                              "foot_nonpenetration": float(foot_np.detach()),
                              "foot_tangential": float(foot_tv.detach()),
                              "temporal_local": float(tl.detach()), "temporal_root": float(tr.detach()),
                              "temporal_total": float(tloss.detach())})
            loss.backward(); torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 10.0); opt.step()
            if scheduler is not None:
                scheduler.step()
            if trace is not None:
                trace[-1]["param_step_l2_local"] = float(torch.linalg.vector_norm(latent.detach()-trace_params["latent"]).detach())
                trace[-1]["param_step_l2_root"] = float(torch.linalg.vector_norm(root.detach()-trace_params["root"]).detach())
                trace[-1]["param_step_l2_translation"] = float(torch.linalg.vector_norm(transl.detach()-trace_params["transl"]).detach())
                trace[-1]["param_step_l2"] = float((trace[-1]["param_step_l2_local"]**2+trace[-1]["param_step_l2_root"]**2+trace[-1]["param_step_l2_translation"]**2)**0.5)
                trace[-1]["lr_latent_root"] = float(opt.param_groups[0]["lr"])
                trace[-1]["lr_translation"] = float(opt.param_groups[1]["lr"])
            if beta_reg: beta.data.clamp_(-1.5,1.5)
            last=(float(loss.detach()),float(l3.detach()),float(l2.detach()),float(lp.detach()),float(lfoot.detach()),float(lhand.detach()),float(lb.detach()))
        return last

    def save_stage(tag):
        with torch.no_grad():
            result,pose,jc,pl,pr,_,_,_,lfoot,lhand,_,_,_,_,_=losses(beta)
        v=result.vertices.cpu().numpy(); p=jc.cpu().numpy()
        r=root.detach().cpu().numpy(); tr=transl.detach().cpu().numpy(); po=pose.detach().cpu().numpy(); be=beta.detach().cpu().numpy()[0]
        e3=np.linalg.norm(p-target_np,axis=-1)*1000.0
        eleft=np.linalg.norm(pl.cpu().numpy()-left[:,:,:2],axis=-1)
        eright=np.linalg.norm(pr.cpu().numpy()-right[:,:,:2],axis=-1)
        m=accepted & np.isfinite(e3)
        finite2=np.isfinite(eleft)&np.isfinite(eright)
        extra = {}
        if surface_mode and tag.startswith("stage_d_surface"):
            vg_np = (np.einsum("nij,nvj->nvi", Rgc.detach().cpu().numpy(), v)
                     + Tgc.detach().cpu().numpy()[:, None, :])
            sole_idx_np = {s: surface["sole_idx"][s].detach().cpu().numpy() for s in ("left", "right")}
            palm_idx_np = {s: surface["palm_idx"][s].detach().cpu().numpy() for s in ("left", "right")}
            foot_z = {s: vg_np[:, sole_idx_np[s], 2] for s in ("left", "right")}
            cap_np = {s: (surface["capsules"][s][0].detach().cpu().numpy(),
                          surface["capsules"][s][1].detach().cpu().numpy()) for s in ("left", "right")}
            hand_res = {}
            for s in ("left", "right"):
                pts = vg_np[:, palm_idx_np[s], :]
                a, b = cap_np[s]
                ab = b - a
                abb = ab[:, None, :]
                u = (((pts - a[:, None, :]) * abb).sum(-1, keepdims=True)
                     / np.maximum((abb * abb).sum(-1, keepdims=True), 1e-8)).clip(0.0, 1.0)
                hand_res[s] = np.linalg.norm(pts - (a[:, None, :] + u * ab[:, None, :]), axis=-1) - surface["radius_m"]
            if not (np.isfinite(vg_np).all() and all(np.isfinite(foot_z[s]).all() for s in foot_z)
                    and all(np.isfinite(hand_res[s]).all() for s in hand_res)):
                raise ValueError("non-finite surface contact arrays")
            extra = {
                "surface_foot_vertices_ground_m": np.stack([vg_np[:, sole_idx_np["left"], :],
                                                            vg_np[:, sole_idx_np["right"], :]], axis=1),
                "surface_hand_vertices_ground_m": np.stack([vg_np[:, palm_idx_np["left"], :],
                                                            vg_np[:, palm_idx_np["right"], :]], axis=1),
                "foot_surface_residuals_m": np.stack([foot_z["left"], foot_z["right"]], axis=1),
                "hand_surface_residuals_m": np.stack([hand_res["left"], hand_res["right"]], axis=1),
                "foot_softmin_height_m": np.stack([
                    -0.005 * (np.log(np.exp(-foot_z[s] / 0.005).sum(axis=1))
                              - np.log(float(foot_z[s].shape[1]))) for s in ("left", "right")], axis=1),
                "hand_softmin_capsule_distance_m": np.stack([
                    -0.005 * (np.log(np.exp(-hand_res[s] / 0.005).sum(axis=1))
                              - np.log(float(hand_res[s].shape[1]))) for s in ("left", "right")], axis=1),
                "foot_active_vertex_count": np.array([sole_idx_np["left"].size, sole_idx_np["right"].size]),
                "hand_active_vertex_count": np.array([palm_idx_np["left"].size, palm_idx_np["right"].size]),
            }
        np.savez_compressed(out/f"result_{tag}.npz",vertices=v,faces=np.asarray(model.faces),predicted_coco=p,
                            betas=np.repeat(be[None,:],n,axis=0),body_pose=po,global_orient=r,transl=tr,
                            raw_triangulated_points=raw,temporally_processed_points=raw,joint_confidence=q,
                            accepted_mask=accepted,reject_reason=reason, **extra)
        return {"tag":tag,"3d_median_mm":float(np.median(e3[m])) if m.any() else None,
                "3d_p95_mm":float(np.percentile(e3[m],95)) if m.any() else None,
                "2d_median_px":float(np.median(np.r_[eleft[finite2],eright[finite2]])) if finite2.any() else None,
                "2d_p95_px":float(np.percentile(np.r_[eleft[finite2],eright[finite2]],95)) if finite2.any() else None,
                "beta":be.tolist(),"beta_at_boundary":bool(np.any(np.isclose(np.abs(be),1.5,atol=1e-3)))}

    # Stage A: beta fixed zero, optimize per-frame latent/root/translation.
    # 3D-dominant dimensionless observation: 3D coeff 1.0 > 2D coeff 0.25.
    obs3d_abc = float(args.obs_3d_weight)
    obs2d_abc = float(args.obs_2d_weight)
    obs3d_d = float(args.obs_3d_weight * args.stage_d_obs_3d_scale)
    obs2d_d = float(args.obs_2d_weight * args.stage_d_obs_2d_scale)
    if not (obs3d_d > 0 and obs2d_d > 0):
        raise ValueError("Stage D must keep nonzero observation constraints")
    audit_mode = bool(args.gradient_audit_only)
    if audit_mode and not surface_mode:
        raise ValueError("--gradient-audit-only needs the surface inputs")
    beta.requires_grad_(False)
    stage_timing = {}
    temporal_enabled[0] = False
    t_stage = time.perf_counter()
    s0=run(torch.optim.Adam([{"params":[latent,root],"lr":0.01},{"params":[transl],"lr":0.003}],), args.base_steps, False,
           obs3d_coeff=obs3d_abc, obs2d_coeff=obs2d_abc)
    stage_timing["stage_a_seconds"] = time.perf_counter() - t_stage
    stage_a={"loss":s0,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage("stage_a_beta0")}
    # Stage B: freeze motion and optimize one shared beta only.
    for p in (latent,root,transl): p.requires_grad_(False)
    beta.requires_grad_(True)
    t_stage = time.perf_counter()
    s1=run(torch.optim.Adam([{"params":[beta],"lr":0.0005}],), args.beta_steps, True,
           obs3d_coeff=obs3d_abc, obs2d_coeff=obs2d_abc)
    stage_timing["stage_b_seconds"] = time.perf_counter() - t_stage
    stage_b={"loss":s1,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage("stage_b_shared_beta")}
    # Stage C: joint refinement with beta learning rate ten times lower.
    for p in (latent,root,transl): p.requires_grad_(True)
    temporal_enabled[0] = bool(args.temporal_mode == "stage_c_and_d")
    t_stage = time.perf_counter()
    s2=run(torch.optim.Adam([{"params":[latent,root],"lr":0.003},{"params":[transl],"lr":0.001},{"params":[beta],"lr":0.0003}],), args.joint_steps, True,
           obs3d_coeff=obs3d_abc, obs2d_coeff=obs2d_abc)
    stage_timing["stage_c_seconds"] = time.perf_counter() - t_stage
    stage_c={"loss":s2,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage("stage_c_joint")}

    if audit_mode:
        # Stage C gradient audit: same parameter state, forward once, no
        # Stage D optimization, no opt.step(). beta stays frozen.
        beta.requires_grad_(False)
        train_params = {"latent": latent, "root": root, "transl": transl}
        grad_audit = gradient_audit(beta, forward, losses, surface, surface_terms, train_params,
                                    obs3d_abc, obs2d_abc)
        (out / "gradient_audit.json").write_text(
            json.dumps(grad_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "completed_stage_c_gradient_audit",
                          "stage_d_executed": False}, ensure_ascii=False, indent=2))
        return 0

    beta_before_d = beta.detach().clone()
    beta_frozen_during_stage_d = False
    stage_d = None
    control_d = surface_mode and bool(args.force_stage_d_no_contact)
    if control_d and (args.surface_foot_contact_weight > 0 or args.surface_hand_contact_weight > 0):
        raise ValueError("--force-stage-d-no-contact requires zero surface contact weights")
    if control_d:
        eff_foot_w, eff_hand_w = 0.0, 0.0
    want_surface_d = surface_mode and (
        args.surface_foot_contact_weight > 0 or args.surface_hand_contact_weight > 0 or control_d
    )
    want_ankle_d = (not surface_mode) and args.contact_labels is not None and args.foot_contact_weight > 0
    stage_d_trace = [] if args.stage_d_trace else None
    if want_surface_d or want_ankle_d:
        contact_enabled = True
        temporal_enabled[0] = bool(args.temporal_mode in ("stage_d", "stage_c_and_d"))
        # Real Stage C -> D continuation in the same process: freeze beta and
        # optimize only latent/root/transl from the Stage C memory state.
        beta.requires_grad_(False)
        beta_frozen_during_stage_d = True
        # Temporal gradient audit at first Stage D forward (no opt.step yet).
        with torch.enable_grad():
            _res0, _, _jc0, _, _, _l3, _l2, _lp, _lf, _lh, _tl0, _tr0, _tinfo0, _fnp0, _ftv0 = losses(beta)
            _plist = [("latent", latent), ("root", root), ("transl", transl)]
            _zero_attach = 0.0*(latent.sum()+root.sum()+transl.sum())
            _flat = [(k, (c*v + _zero_attach, c, _plist)) for k, v, c in [
                ("obs3d", obs3d_d*_l3, obs3d_d), ("obs2d", obs2d_d*_l2, obs2d_d),
                ("foot", eff_foot_w*_lf, eff_foot_w), ("hand", eff_hand_w*_lh, eff_hand_w),
                ("temporal_local", args.temporal_local_weight*_tl0, args.temporal_local_weight),
                ("temporal_root", args.temporal_root_weight*_tr0, args.temporal_root_weight)]]
            _rows = gradient_norms(_flat, {"latent": latent, "root": root, "transl": transl})
        (out / "temporal_gradient_audit.json").write_text(json.dumps(
            {"temporal_mode": args.temporal_mode, "terms": _rows, "masks": _tinfo0,
             "dt": "1/30", "huber_scale_mm": float(args.temporal_huber_scale_mm)}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        stage_d_opt = torch.optim.Adam([{"params":[latent,root],"lr":0.001},{"params":[transl],"lr":0.0005}],)
        if args.stage_d_lr_min_factor <= 0.0 or args.stage_d_lr_min_factor > 1.0:
            raise ValueError("--stage-d-lr-min-factor must be in (0,1]")
        if args.stage_d_lr_schedule == "cosine":
            stage_d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                stage_d_opt, T_max=max(1, args.contact_steps),
                eta_min=0.001 * args.stage_d_lr_min_factor)
        elif args.stage_d_lr_schedule == "exponential":
            gamma = args.stage_d_lr_min_factor ** (1.0 / max(1, args.contact_steps))
            stage_d_scheduler = torch.optim.lr_scheduler.ExponentialLR(stage_d_opt, gamma=gamma)
        else:
            stage_d_scheduler = None
        t_stage = time.perf_counter()
        s3=run(stage_d_opt, args.contact_steps, False,
               obs3d_coeff=obs3d_d, obs2d_coeff=obs2d_d, trace=stage_d_trace,
               scheduler=stage_d_scheduler)
        stage_timing["stage_d_seconds"] = time.perf_counter() - t_stage
        if not torch.equal(beta.detach(), beta_before_d.detach()):
            raise RuntimeError("beta changed during Stage D despite requires_grad_(False)")
        if want_surface_d:
            foot_on = eff_foot_w > 0
            hand_on = eff_hand_w > 0
            tag = ("stage_d_no_contact_control" if control_d
                   else "stage_d_surface_foot_hand" if (foot_on and hand_on)
                   else "stage_d_surface_hand" if hand_on else "stage_d_surface_foot")
        else:
            tag = "stage_d_contact"
        stage_d={"loss":s3,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage(tag),
                 "surface_mode": bool(want_surface_d)}
        if control_d:
            # Zero-contact control: no surface gradient proof is fabricated.
            surface_audit["surface_probe"] = "skipped_no_contact_control"
            surface_audit["stage_d_control_steps"] = int(args.contact_steps)
            surface_audit["stage_d_control_weights"] = {"foot": 0.0, "hand": 0.0}
        if want_surface_d and not control_d:
            # Same-forward COCO isolation proof: one forward(beta) call yields
            # both the vertices and the COCO joints; the probe surface loss is
            # built from those vertices only, then differentiated w.r.t. both.
            with torch.enable_grad():
                result_probe, _, jc_probe, _, _ = forward(beta)
                probe_foot, probe_hand, *_rest = surface_terms(result_probe.vertices)
                probe_loss = (
                    args.surface_foot_contact_weight * probe_foot
                    + args.surface_hand_contact_weight * probe_hand
                )
                grad_to_coco = torch.autograd.grad(
                    probe_loss, jc_probe, torch.ones_like(probe_loss),
                    allow_unused=True, retain_graph=True)[0]
                grad_to_verts = torch.autograd.grad(
                    probe_loss, result_probe.vertices, torch.ones_like(probe_loss),
                    allow_unused=True)[0]
            coco_norm = 0.0 if grad_to_coco is None else float(grad_to_coco.detach().abs().max())
            if grad_to_verts is None or not bool(torch.isfinite(grad_to_verts).all()):
                raise RuntimeError("surface loss has no finite gradient to SMPL vertices")
            verts_norm = float(grad_to_verts.detach().abs().max())
            if verts_norm <= 0.0:
                raise RuntimeError("surface loss does not depend on SMPL vertices")
            surface_audit["same_forward_graph"] = True
            surface_audit["surface_probe_grad_norm_to_coco"] = coco_norm
            surface_audit["surface_probe_grad_norm_to_vertices"] = verts_norm
            surface_audit["surface_loss_depends_on_vertices"] = True
            surface_audit["coco_joints_in_surface_contact"] = bool(coco_norm > 0.0)
            if surface_audit["coco_joints_in_surface_contact"]:
                raise RuntimeError("COCO joints leaked into the surface contact graph")

    with torch.no_grad():
        temporal_enabled[0] = bool(args.temporal_mode in ("stage_d", "stage_c_and_d"))
        result,pose,jc,pl,pr,_,_,_,lfoot,lhand,tl_fin,tr_fin,tinfo_fin,foot_np_fin,foot_tv_fin=losses(beta)
        if stage_d_trace is not None and stage_d is not None:
            _, _, _, _, _, l3_end, l2_end, lp_end, _, _, tl_end, tr_end, _, foot_np_end, foot_tv_end = losses(beta)
            total_end = (obs3d_d*l3_end + obs2d_d*l2_end + 0.02*lp_end
                         + eff_foot_w*lfoot + eff_hand_w*lhand
                         + args.temporal_local_weight*tl_end + args.temporal_root_weight*tr_end)
            stage_d_trace.append({"step": int(args.contact_steps), "total": float(total_end),
                                  "obs3d": float(l3_end), "obs2d": float(l2_end),
                                  "pose": float(lp_end), "foot": float(lfoot),
                                  "hand": float(lhand), "temporal_local": float(tl_end),
                                  "temporal_root": float(tr_end),
                                  "temporal_total": float(args.temporal_local_weight*tl_end+args.temporal_root_weight*tr_end),
                                  "foot_nonpenetration": float(foot_np_end), "foot_tangential": float(foot_tv_end)})
            (out / "stage_d_trace.json").write_text(
                json.dumps({"loss_evaluation": "pre_step_for_0_to_N_minus_1_post_step_for_N",
                            "terms": stage_d_trace}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
    verts=result.vertices.cpu().numpy(); pred=jc.cpu().numpy(); roots=root.detach().cpu().numpy(); trans=transl.detach().cpu().numpy(); pose_np=pose.detach().cpu().numpy(); beta_np=beta.detach().cpu().numpy()[0]
    d3=np.linalg.norm(pred-target_np,axis=-1)*1000.0; eL=np.linalg.norm(pl.cpu().numpy()-left[:,:,:2],axis=-1); eR=np.linalg.norm(pr.cpu().numpy()-right[:,:,:2],axis=-1)
    mask=accepted & np.isfinite(d3)
    eLmed=eL[valid2d.cpu().numpy()]; eRmed=eR[valid2d.cpu().numpy()]
    metrics={"status":"completed_staged_single_frame_vposer_shared_beta","window":window,"frames":n,"full_sequence_fit":False,
             "stage_timing_seconds": stage_timing, "wall_seconds": float(sum(stage_timing.values())), "frames_per_second": float(n/max(sum(stage_timing.values()),1e-6)),
             "obs_3d_mode": args.obs_3d_mode, "obs_3d_base_sigma_mm": float(args.obs_3d_base_sigma_mm), "obs_3d_max_sigma_factor": float(args.obs_3d_max_sigma_factor),
             "obs_3d_sigma_factor": {"median": float(np.median(sigma_factor_np[accepted])) if accepted.any() else None, "p95": float(np.percentile(sigma_factor_np[accepted],95)) if accepted.any() else None, "max": float(np.max(sigma_factor_np[accepted])) if accepted.any() else None},
             "temporal_mode": args.temporal_mode, "temporal_local_weight": float(args.temporal_local_weight), "temporal_root_weight": float(args.temporal_root_weight), "temporal_huber_scale_mm": float(args.temporal_huber_scale_mm),
             "temporal_local_loss": float(tl_fin.detach()), "temporal_root_loss": float(tr_fin.detach()), "temporal_local_valid_count": int(tinfo_fin.get("local_valid",0)), "temporal_root_valid_count": int(tinfo_fin.get("root_valid",0)), "temporal_rejected_triplet_count": int(tinfo_fin.get("rejected_triplets",0)), "temporal_gap_count": int(tinfo_fin.get("gap_triplets",0)),
             "stage_c_to_d_same_process":bool(stage_d is not None),"beta_frozen_during_stage_d":bool(beta_frozen_during_stage_d),"stage_a_beta_zero":stage_a,"stage_b_shared_beta":stage_b,"stage_c_joint":stage_c,"stage_d_contact":stage_d,"contact":{"enabled":bool(stage_d is not None),"foot_weight":float(args.foot_contact_weight),"hand_weight":float(args.hand_contact_weight),"label_file":str(args.contact_labels.resolve()) if args.contact_labels else None,"scene_file":str(args.scene_transforms.resolve()) if args.scene_transforms else None,"final_foot_loss":float(lfoot.detach()),"final_hand_loss":float(lhand.detach())},"2d":{"median_px":float(np.median(np.r_[eLmed,eRmed])),"p95_px":float(np.percentile(np.r_[eLmed,eRmed],95)),"left_p95_px":float(np.percentile(eLmed,95)),"right_p95_px":float(np.percentile(eRmed,95))},"3d":{"median_mm":float(np.median(d3[mask])) if mask.any() else None,"p95_mm":float(np.percentile(d3[mask],95)) if mask.any() else None,"joint_p95_mm":{NAMES[j]:float(np.percentile(d3[:,j][mask[:,j]],95)) if mask[:,j].any() else None for j in range(17)}},"beta":{"values":beta_np.tolist(),"shared":True,"at_boundary":bool(np.any(np.isclose(np.abs(beta_np),1.5,atol=1e-3)))},"right_knee_accepted":int(accepted[:,14].sum()),"vposer":{"latent_dim":int(vp_cfg.model_params.latentD),"checkpoint":str(vp_ckpt.resolve())},"source_audit":{"raw_left":str(args.left.resolve()),"raw_right":str(args.right.resolve()),"old_fit_inputs_read":False,"stored_triangulation_read":False,"temporal_prefit_read":False}}
    np.savez_compressed(out/"result.npz",vertices=verts,faces=np.asarray(model.faces),predicted_coco=pred,betas=np.repeat(beta_np[None,:],n,axis=0),body_pose=pose_np,global_orient=roots,transl=trans,raw_triangulated_points=raw,temporally_processed_points=raw,joint_confidence=q,accepted_mask=accepted,reject_reason=reason,obs3d_sigma_m=sigma_np,obs3d_sigma_factor=sigma_factor_np)
    (out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"temporal_metrics.json").write_text(json.dumps({"temporal_mode":args.temporal_mode,"dt":"1/30 (30 FPS, no timestamps)","frame_ok":"triangulation accepted.any + scene accepted + finite Rgc/Tgc; triplet needs 3 consecutive ok, gap<=max_gap","huber_scale_mm":float(args.temporal_huber_scale_mm),"local_joints":"predicted_coco_ground minus pelvis_ground, COCO [5,6,7,8,9,10,11,12,13,14,15,16]","accel":"second difference /(dt^2), pseudo-Huber mean","local_loss":float(tl_fin.detach()),"root_loss":float(tr_fin.detach()),"local_valid":int(tinfo_fin.get("local_valid",0)),"root_valid":int(tinfo_fin.get("root_valid",0)),"rejected_triplets":int(tinfo_fin.get("rejected_triplets",0)),"gap_triplets":int(tinfo_fin.get("gap_triplets",0)),"status":str(tinfo_fin.get("status",""))},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if surface_mode:
        scm = {"status": "technical_chain_only_not_physical_contact_validation",
               "model": "male_smpl_6890", "contact_points": "surface_vertices_not_coco_joints",
               "window": window, "frames": n, "full_sequence_fit": False,
               "stage_c_to_d_same_process": bool(stage_d is not None),
               "beta_frozen_during_stage_d": bool(beta_frozen_during_stage_d),
               "hand_contact_enabled": bool(args.surface_hand_contact_weight > 0),
               "surface_foot_contact_weight": float(args.surface_foot_contact_weight),
               "surface_hand_contact_weight": float(args.surface_hand_contact_weight),
               "stage_c": stage_c["metrics"], "stage_d_surface": (stage_d["metrics"] if stage_d else None),
               "beta": beta_np.tolist()}
        (out/"surface_contact_metrics.json").write_text(json.dumps(scm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out/"surface_contact_forward_audit.json").write_text(json.dumps(surface_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out/"source_audit.json").write_text(json.dumps({
            "inputs": {"raw_left": str(args.left.resolve()), "raw_right": str(args.right.resolve()),
                       "contact_labels": str(args.contact_labels.resolve()),
                       "scene_transforms": str(args.scene_transforms.resolve()),
                       "contact_vertex_sets": surface["sets_file"],
                       "walker_topology": surface["topology_file"],
                       "model": str(args.model.resolve()), "regressor": str(args.regressor.resolve()),
                       "vposer_dir": str(args.vposer_dir.resolve()),
                       "calibration_dir": str(args.calibration_dir.resolve())},
            "forbidden_inputs_read": [],
            "old_fit_inputs_read": False, "stored_triangulation_read": False,
            "temporal_prefit_read": False, "old_interaction_jsonl_read": False,
            "old_motion_json_read": False, "old_html_read": False,
            "old_dynamic_walker_pose_read": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out/"command.txt").write_text(" ".join(sys.argv)+"\n",encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return 0

def gradient_norms(flat_terms, params):
    """Return finite L2 and max-abs gradient per term and named parameter.

    All (term, parameter) pairs share one forward graph; only the final pair
    releases it.
    """
    import math
    import torch
    rows = {}
    total = sum(len(plist) for _, plist in flat_terms)
    done = 0
    for name, (term, coeff, plist) in flat_terms:
        l2d, maxd, sq, finite = {}, {}, 0.0, True
        for pname, p in plist:
            done += 1
            g = torch.autograd.grad(term, p, torch.ones_like(term), allow_unused=True,
                                    retain_graph=(done < total))[0]
            if g is None:
                l2d[pname], maxd[pname] = 0.0, 0.0
                continue
            if not bool(torch.isfinite(g).all()):
                finite = False
            l2 = float(g.detach().float().pow(2).sum().sqrt())
            l2d[pname] = l2
            maxd[pname] = float(g.detach().abs().max())
            sq += l2 * l2
        l2d["all"] = math.sqrt(sq)
        maxd["all"] = max((maxd[k] for k, _ in plist), default=0.0)
        if not finite:
            raise ValueError(f"non-finite gradient in term {name}")
        rows[name] = {"term_value": float(term.detach()),
                      "coeff": float(coeff),
                      "gradient_l2": l2d, "gradient_max_abs": maxd, "finite": True}
    return rows


def gradient_audit(beta, forward, losses, surface, surface_terms, train_params,
                   obs3d_coeff, obs2d_coeff):
    """Audit per-term gradients at the frozen Stage C state (no Stage D)."""
    import torch
    if beta.requires_grad:
        raise ValueError("beta must be frozen for the gradient audit")
    with torch.enable_grad():
        result, pose, jc, pl, pr, l3, l2, lp, lfoot, lhand, _, _, _, foot_np, foot_tv = losses(beta)
        plist = list(train_params.items())
        flat = [("obs3d", (obs3d_coeff * l3, obs3d_coeff, plist)),
                ("obs2d", (obs2d_coeff * l2, obs2d_coeff, plist)),
                ("foot", (lfoot, 1.0, plist)),
                ("hand", (lhand, 1.0, plist))]
        rows = gradient_norms(flat, train_params)
    return {"status": "completed_stage_c_gradient_audit",
            "engineering_validation_only": True,
            "stage_d_executed": False,
            "beta_optimized_in_audit": False,
            "terms": rows}


if __name__=="__main__": raise SystemExit(main())
