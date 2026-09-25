#!/usr/bin/env python3
"""Single-frame VPoser SMPL fits with one beta shared by the whole clip.

The raw PMPose files are triangulated in this run. No stored triangulation,
temporal prefit, fit parameters, contact targets or HTML are read.
"""
from __future__ import annotations

import argparse, json, math, sys
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
    ap.add_argument("--contact-vertex-sets", type=Path, default=None)
    ap.add_argument("--walker-topology", type=Path, default=None)
    ap.add_argument("--surface-foot-contact-weight", type=float, default=0.0)
    ap.add_argument("--surface-hand-contact-weight", type=float, default=0.0)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be supplied together")
    if args.hand_contact_weight < 0.0 or args.surface_hand_contact_weight < 0.0:
        raise ValueError("hand contact weights must be non-negative")
    surface_mode = args.contact_vertex_sets is not None
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
        l3=(robust_norm(jc-target,0.10)*w3).sum()/(w3.sum()+1e-6)
        e2=robust_norm(pl-obs_l,20.0)+robust_norm(pr-obs_r,20.0)
        l2=(e2*w*valid2d).sum()/(2*(w*valid2d).sum()+1e-6)
        lp=latent.square().mean()
        lfoot=torch.zeros((),device=device); lhand=torch.zeros((),device=device)
        if surface_mode and contact_enabled:
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
        return result,pose,jc,pl,pr,l3,l2,lp,lfoot,lhand

    def run(opt, steps, beta_reg=True):
        last=None
        for _ in range(steps):
            opt.zero_grad(); vals=losses(beta); _,pose,_,_,_,l3,l2,lp,lfoot,lhand=vals
            lb=beta.square().mean()
            if surface_mode:
                contact_loss = args.surface_foot_contact_weight*lfoot + args.surface_hand_contact_weight*lhand
            else:
                contact_loss = args.foot_contact_weight*lfoot + args.hand_contact_weight*lhand
            loss=l3+0.10*l2+0.02*lp+(contact_loss if contact_enabled else 0.0)+(0.02*lb if beta_reg else 0.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 10.0); opt.step()
            if beta_reg: beta.data.clamp_(-1.5,1.5)
            last=(float(loss.detach()),float(l3.detach()),float(l2.detach()),float(lp.detach()),float(lfoot.detach()),float(lhand.detach()),float(lb.detach()))
        return last

    def save_stage(tag):
        with torch.no_grad():
            result,pose,jc,pl,pr,_,_,_,lfoot,lhand=losses(beta)
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
                    -0.005 * np.log(np.exp(-foot_z[s] / 0.005).sum(axis=1)) for s in ("left", "right")], axis=1),
                "hand_softmin_capsule_distance_m": np.stack([
                    -0.005 * np.log(np.exp(-hand_res[s] / 0.005).sum(axis=1)) for s in ("left", "right")], axis=1),
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
    beta.requires_grad_(False)
    s0=run(torch.optim.Adam([{"params":[latent,root],"lr":0.01},{"params":[transl],"lr":0.003}],), args.base_steps, False)
    stage_a={"loss":s0,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage("stage_a_beta0")}
    # Stage B: freeze motion and optimize one shared beta only.
    for p in (latent,root,transl): p.requires_grad_(False)
    beta.requires_grad_(True)
    s1=run(torch.optim.Adam([{"params":[beta],"lr":0.0005}],), args.beta_steps, True)
    stage_b={"loss":s1,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage("stage_b_shared_beta")}
    # Stage C: joint refinement with beta learning rate ten times lower.
    for p in (latent,root,transl): p.requires_grad_(True)
    s2=run(torch.optim.Adam([{"params":[latent,root],"lr":0.003},{"params":[transl],"lr":0.001},{"params":[beta],"lr":0.0003}],), args.joint_steps, True)
    stage_c={"loss":s2,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage("stage_c_joint")}

    beta_before_d = beta.detach().clone()
    beta_frozen_during_stage_d = False
    stage_d = None
    want_surface_d = surface_mode and (
        args.surface_foot_contact_weight > 0 or args.surface_hand_contact_weight > 0
    )
    want_ankle_d = (not surface_mode) and args.contact_labels is not None and args.foot_contact_weight > 0
    if want_surface_d or want_ankle_d:
        contact_enabled = True
        # Real Stage C -> D continuation in the same process: freeze beta and
        # optimize only latent/root/transl from the Stage C memory state.
        beta.requires_grad_(False)
        beta_frozen_during_stage_d = True
        s3=run(torch.optim.Adam([{"params":[latent,root],"lr":0.001},{"params":[transl],"lr":0.0005}],), args.contact_steps, False)
        if not torch.equal(beta.detach(), beta_before_d.detach()):
            raise RuntimeError("beta changed during Stage D despite requires_grad_(False)")
        if want_surface_d:
            foot_on = args.surface_foot_contact_weight > 0
            hand_on = args.surface_hand_contact_weight > 0
            tag = ("stage_d_surface_foot_hand" if (foot_on and hand_on)
                   else "stage_d_surface_hand" if hand_on else "stage_d_surface_foot")
        else:
            tag = "stage_d_contact"
        stage_d={"loss":s3,"beta":beta.detach().cpu().numpy()[0].tolist(),"metrics":save_stage(tag),
                 "surface_mode": bool(want_surface_d)}
        if want_surface_d:
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
        result,pose,jc,pl,pr,_,_,_,lfoot,lhand=losses(beta)
    verts=result.vertices.cpu().numpy(); pred=jc.cpu().numpy(); roots=root.detach().cpu().numpy(); trans=transl.detach().cpu().numpy(); pose_np=pose.detach().cpu().numpy(); beta_np=beta.detach().cpu().numpy()[0]
    d3=np.linalg.norm(pred-target_np,axis=-1)*1000.0; eL=np.linalg.norm(pl.cpu().numpy()-left[:,:,:2],axis=-1); eR=np.linalg.norm(pr.cpu().numpy()-right[:,:,:2],axis=-1)
    mask=accepted & np.isfinite(d3)
    eLmed=eL[valid2d.cpu().numpy()]; eRmed=eR[valid2d.cpu().numpy()]
    metrics={"status":"completed_staged_single_frame_vposer_shared_beta","window":window,"frames":n,"full_sequence_fit":False,
             "stage_c_to_d_same_process":bool(stage_d is not None),"beta_frozen_during_stage_d":bool(beta_frozen_during_stage_d),"stage_a_beta_zero":stage_a,"stage_b_shared_beta":stage_b,"stage_c_joint":stage_c,"stage_d_contact":stage_d,"contact":{"enabled":bool(stage_d is not None),"foot_weight":float(args.foot_contact_weight),"hand_weight":float(args.hand_contact_weight),"label_file":str(args.contact_labels.resolve()) if args.contact_labels else None,"scene_file":str(args.scene_transforms.resolve()) if args.scene_transforms else None,"final_foot_loss":float(lfoot.detach()),"final_hand_loss":float(lhand.detach())},"2d":{"median_px":float(np.median(np.r_[eLmed,eRmed])),"p95_px":float(np.percentile(np.r_[eLmed,eRmed],95)),"left_p95_px":float(np.percentile(eLmed,95)),"right_p95_px":float(np.percentile(eRmed,95))},"3d":{"median_mm":float(np.median(d3[mask])) if mask.any() else None,"p95_mm":float(np.percentile(d3[mask],95)) if mask.any() else None,"joint_p95_mm":{NAMES[j]:float(np.percentile(d3[:,j][mask[:,j]],95)) if mask[:,j].any() else None for j in range(17)}},"beta":{"values":beta_np.tolist(),"shared":True,"at_boundary":bool(np.any(np.isclose(np.abs(beta_np),1.5,atol=1e-3)))},"right_knee_accepted":int(accepted[:,14].sum()),"vposer":{"latent_dim":int(vp_cfg.model_params.latentD),"checkpoint":str(vp_ckpt.resolve())},"source_audit":{"raw_left":str(args.left.resolve()),"raw_right":str(args.right.resolve()),"old_fit_inputs_read":False,"stored_triangulation_read":False,"temporal_prefit_read":False}}
    np.savez_compressed(out/"result.npz",vertices=verts,faces=np.asarray(model.faces),predicted_coco=pred,betas=np.repeat(beta_np[None,:],n,axis=0),body_pose=pose_np,global_orient=roots,transl=trans,raw_triangulated_points=raw,temporally_processed_points=raw,joint_confidence=q,accepted_mask=accepted,reject_reason=reason)
    (out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
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

if __name__=="__main__": raise SystemExit(main())
