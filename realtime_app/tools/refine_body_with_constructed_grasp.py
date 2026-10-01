#!/usr/bin/env python3
"""Optional SMPL-H refinement of an immutable existing fit.

Rollback: use the source result and legacy fit_smplh_wilor_sequence.py.
This tool never changes them. Short windows are probes, not stitched outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "realtime_app"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source-result", "grasp", "walker-model", "scene-transforms",
                 "contact-labels", "contact-vertex-sets", "left-raw", "right-raw",
                 "output-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--stop", type=int, default=91, help="exclusive full-sequence index")
    ap.add_argument("--upper-steps", type=int, default=200)
    ap.add_argument("--body-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=.01)
    ap.add_argument("--wrist-weight", type=float, default=1.)
    ap.add_argument("--orientation-weight", type=float, default=1.)
    ap.add_argument("--body-scale-m", type=float, default=.05)
    ap.add_argument("--grasp-mesh", type=Path, help="matching constructed NPZ; default JSON path with .npz suffix")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError("refuse existing output directory")
    if min(args.upper_steps, args.body_steps) < 0 or not np.isfinite(
            [args.lr, args.wrist_weight, args.orientation_weight, args.body_scale_m]).all() or args.lr <= 0 or args.body_scale_m <= 0 or min(
                args.wrist_weight, args.orientation_weight) < 0:
        raise ValueError("invalid optimizer configuration")

    import torch
    import smplx
    from smplx.utils import Struct
    from smplx.lbs import batch_rodrigues
    from scipy.spatial.transform import Rotation
    from pose_app.smpl_coco_observation import (
        _install_legacy_smpl_pickle_compatibility, load_coco17_regressor,
        regress_coco17_torch)
    from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_torch
    from pose_app import smpl_surface_contact as surf
    from pose_app.constructed_grasp_refinement import (
        UPPER_JOINTS, TORSO_JOINTS, load_grasp, validate_rotation, compose_body,
        wrist_rotations, masked_mean, foot_terms, frozen_surface_states)

    torch.set_num_threads(2)
    torch.manual_seed(20261001)
    device = torch.device(args.device)
    def tensor(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    def write(name, value):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    _install_legacy_smpl_pickle_compatibility()
    z = np.load(args.source_result, allow_pickle=False)
    grasp = load_grasp(args.grasp)
    walker = json.loads(args.walker_model.read_text(encoding="utf-8"))
    scene = np.load(args.scene_transforms, allow_pickle=True)
    labels = np.load(args.contact_labels, allow_pickle=True)
    total_frames = len(z["body_pose"])
    if not 0 <= args.start < args.stop <= total_frames or args.stop - args.start < 3:
        raise ValueError("window must contain at least three valid sequence indices")
    sl = slice(args.start, args.stop)
    n = args.stop - args.start
    Rg = np.asarray(scene["rotation_ground_from_left"])
    tg = np.asarray(scene["translation_ground_from_left_mm"]) / 1000
    fw = np.asarray(labels["foot_contact_weight"])
    if Rg.shape != (total_frames, 3, 3) or tg.shape != (total_frames, 3) or fw.shape != (total_frames, 2):
        raise ValueError("scene/contact sequence shape mismatch")
    validate_rotation(Rg)
    if not np.isfinite(tg).all() or not np.isfinite(fw).all() or np.any(fw < 0):
        raise ValueError("invalid ground/foot weights")
    Rc = np.asarray(walker["rotation_left_camera_from_walker"])
    tc = np.asarray(walker["translation_left_camera_from_walker_mm"]) / 1000
    validate_rotation(Rc)
    if tc.shape != (3,) or not np.isfinite(tc).all():
        raise ValueError("invalid camera-walker translation")
    if not np.allclose(z["betas"], grasp["betas"], atol=1e-6):
        raise ValueError("constructed grasp beta differs from frozen source beta")
    sets = json.loads(args.contact_vertex_sets.read_text(encoding="utf-8"))["sets"]
    soles = [np.unique(sum(sets[f"{s}_sole_surface_candidate"].values(), [])) for s in ("left", "right")]
    if any(len(ids) == 0 or np.min(ids) < 0 or np.max(ids) >= 6890 for ids in soles):
        raise ValueError("invalid sole surface vertex IDs")
    frame_ok_full = np.asarray(z["body_accepted"]).any(1) & np.asarray(scene["accepted"]).any(1)
    vg_source = np.einsum("nij,nvj->nvi", Rg, z["vertices"]) + tg[:, None]
    support_full, states_full, reasons_full = frozen_surface_states(vg_source, soles, frame_ok_full)
    support = torch.as_tensor(support_full[sl], device=device)
    frame_ok = torch.as_tensor(frame_ok_full[sl], device=device)
    if not frame_ok.any():
        raise ValueError("no valid scene/body frames")
    Rgt, tgt, Rct, tct = tensor(Rg[sl]), tensor(tg[sl]), tensor(Rc), tensor(tc)

    model_path = ROOT / "third_party/WiLoR/mano_data/models/SMPLH_male.pkl"
    with model_path.open("rb") as f:
        asset = pickle.load(f, encoding="latin1")
    data = dict(asset)
    data.update(hands_componentsl=np.eye(45), hands_componentsr=np.eye(45),
                hands_meanl=np.zeros(45), hands_meanr=np.zeros(45))
    model = smplx.SMPLHLayer(str(model_path), data_struct=Struct(**data),
                        use_pca=False, flat_hand_mean=True, batch_size=n).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    beta = tensor(z["betas"]).expand(n, -1)
    base = batch_rodrigues(tensor(z["body_pose"][sl]).reshape(-1, 3)).reshape(n, 21, 3, 3)
    root = batch_rodrigues(tensor(z["global_orient"][sl])).reshape(n, 1, 3, 3)
    transl = tensor(z["transl"][sl])
    # Matrix layer must reproduce the original axis-angle forward, including
    # mean-once convention, before using any constructed targets.
    with torch.no_grad():
        original_hands = [batch_rodrigues(tensor(z[f"{s}_hand_pose"][sl]).reshape(-1, 3)).reshape(n, 15, 3, 3) for s in ("left", "right")]
        replay = model(betas=beta, global_orient=root, body_pose=base,
                       left_hand_pose=original_hands[0], right_hand_pose=original_hands[1], transl=transl)
        replay_max = float((replay.vertices - tensor(z["vertices"][sl])).abs().max())
        if replay_max > 1e-5:
            raise ValueError(f"matrix replay differs from source: {replay_max} m")
    hand_pose, hand_rot = {}, {}
    for side in ("left", "right"):
        h = grasp["hands"][side]
        with (ROOT / f"third_party/WiLoR/mano_data/models/MANO_{side.upper()}.pkl").open("rb") as f:
            mano = pickle.load(f, encoding="latin1")
        comp = np.asarray(mano["hands_components"][:12])
        decoded = np.asarray(mano["hands_mean"]) + np.asarray(h["hand_pca"]) / np.linalg.norm(comp, axis=1) @ comp
        if not np.allclose(decoded, h["hand_pose_axis_angle"], atol=1e-5):
            raise ValueError("PCA/mean convention audit failed")
        hand_pose[side] = tensor(decoded).expand(n, -1)
        hand_rot[side] = batch_rodrigues(hand_pose[side].reshape(-1, 3)).reshape(n, 15, 3, 3)
    target_p = tensor([grasp["hands"][s]["wrist_walker_m"] for s in ("left", "right")])
    target_R = tensor([grasp["hands"][s]["rotation_walker_from_wrist"] for s in ("left", "right")])
    reg = load_coco17_regressor(ROOT / "models/smpl/J_regressor_coco.npy")
    target = tensor(np.nan_to_num(z["raw_triangulated_points"][sl] / 1000))
    body_mask = torch.as_tensor(z["body_accepted"][sl] & np.isfinite(z["raw_triangulated_points"][sl]).all(-1), device=device)
    quality = tensor(z["body_quality"][sl]).clamp_min(0)
    if not body_mask.any() or not torch.isfinite(quality).all() or quality[body_mask].sum() <= 0:
        raise ValueError("no valid weighted body observations")
    # Reuse the authoritative raw reader; raw fish-eye pixels, no WiLoR pixels.
    spec = importlib.util.spec_from_file_location("raw_grasp", ROOT / "research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py")
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    obs = []
    for path, side in ((args.left_raw, "left"), (args.right_raw, "right")):
        rows = raw.raw_side(path, side)
        if sorted(rows) != list(range(total_frames)):
            raise ValueError("raw frame identity mismatch")
        points = np.stack([rows[i] for i in range(args.start, args.stop)])
        valid = np.isfinite(points[..., :2]).all(-1) & (points[..., 2] > 0)
        obs.append((tensor(np.nan_to_num(points[..., :2])), tensor(np.clip(np.nan_to_num(points[..., 2]), 0, 1)) * tensor(valid)))
    cal = load_stereo_fisheye(ROOT / "realtime_app/calibration/results")
    R01, t01 = tensor(cal.R_cam0_to_cam1), tensor(cal.T_cam0_to_cam1_mm / 1000)
    upper = torch.nn.Parameter(torch.zeros(n, len(UPPER_JOINTS), 3, device=device))
    torso = torch.nn.Parameter(torch.zeros(n, len(TORSO_JOINTS), 3, device=device))
    frozen_lower = [j - 1 for j in range(1, 22) if j not in UPPER_JOINTS + TORSO_JOINTS]
    mesh_path = args.grasp_mesh or args.grasp.with_suffix(".npz")
    with np.load(mesh_path, allow_pickle=False) as mesh:
        hand_indices = [np.asarray(mesh[f"{s}_vertex_indices"]) for s in ("left", "right")]
        for s in ("left", "right"):
            if not np.allclose(mesh[f"{s}_hand_pca"], grasp["hands"][s]["hand_pca"], atol=1e-6):
                raise ValueError("constructed mesh/JSON parameter mismatch")
    if any(ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= 6890 for ids in hand_indices):
        raise ValueError("invalid hand surface indices")
    ends = [tensor([np.asarray(walker["nodes_walker_mm"][key]) / 1000 for key in walker["handle_segments"][s]]) for s in ("left", "right")]
    coefficients = {"body_3d": 1., "body_2d": .15, "pose_anchor": .05,
                    "body_temporal": .02, "wrist_position": args.wrist_weight,
                    "wrist_rotation": args.orientation_weight, "hand_penetration": 20.,
                    "foot_contact": 1., "foot_nonpenetration": 1., "foot_tangential": .001}
    args.output_dir.mkdir(parents=True)
    write("run_metadata.json", {"inputs": {k: str(v.resolve()) for k, v in vars(args).items() if isinstance(v, Path)},
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "coefficients": coefficients, "coordinate_frame": "camera0 fit; walker-local grip; same-source ground feet",
        "scope": "constructed_grasp_body_refinement_engineering_probe", "fps_assumed": 30,
        "frozen": ["beta", "finger_PCA", "root", "translation", "lower_limb", "calibration", "scene"],
        "source_matrix_replay_max_error_m": replay_max,
        "grasp_mesh_source": str(mesh_path.resolve()),
        "body_3d_scale_m": args.body_scale_m,
        "support_source": "frozen same-source baseline sole surface state machine; not measured support",
        "foot_tangent_unavailable_reason": None if support_full[sl].any() else "no_sticking_surface_candidates"})
    np.savez_compressed(args.output_dir / "frozen_foot_states.npz", pair_id=np.arange(args.start, args.stop),
        support=support_full[sl], state=states_full[sl], reason=reasons_full[sl], frame_ok=frame_ok_full[sl])

    def forward():
        body = compose_body(base, upper, UPPER_JOINTS)
        body = compose_body(body, torso, TORSO_JOINTS)
        out = model(betas=beta, global_orient=root, body_pose=body,
                    left_hand_pose=hand_rot["left"], right_hand_pose=hand_rot["right"],
                    transl=transl, return_verts=True)
        coco = regress_coco17_torch(out.vertices, reg)
        vg = torch.einsum("nij,nvj->nvi", Rgt, out.vertices) + tgt[:, None]
        vw = (out.vertices - tct) @ Rct
        wp = (out.joints[:, [20, 21]] - tct) @ Rct
        wr = Rct.T @ wrist_rotations(root[:, 0], body, model.parents)
        pos = torch.linalg.vector_norm(wp - target_p, dim=-1)
        rot_diff = wr - target_R
        # Frobenius chordal rotation residual ~ angle near zero, no acos gradient.
        terms = {"wrist_position": ((pos / .005) ** 2).mean(),
                 "wrist_rotation": (rot_diff.square().sum((-2, -1)) / (2 * np.deg2rad(5) ** 2)).mean(),
                 "pose_anchor": (upper.square().mean() + torso.square().mean()),
                 "body_3d": ((coco - target).square().sum(-1) * quality)[body_mask].sum() / quality[body_mask].sum().clamp_min(1e-6) / args.body_scale_m ** 2}
        projected = [fisheye_project_torch(coco, cal.K0, cal.D0),
                     fisheye_project_torch(coco @ R01.T + t01, cal.K1, cal.D1)]
        terms["body_2d"] = torch.stack([((surf.robust_scalar(torch.linalg.vector_norm(p - xy, dim=-1) / 100, 1.) * w).sum() / w.sum().clamp_min(1e-6)) for p, (xy, w) in zip(projected, obs)]).mean()
        rel = coco - (coco[:, 11:12] + coco[:, 12:13]) / 2
        acceleration = rel[2:] - 2 * rel[1:-1] + rel[:-2]
        valid3 = body_mask[2:] & body_mask[1:-1] & body_mask[:-2] & (frame_ok[2:] & frame_ok[1:-1] & frame_ok[:-2])[:, None]
        terms["body_temporal"] = masked_mean(surf.robust_scalar(torch.linalg.vector_norm(acceleration, dim=-1), .03), valid3)
        feet, active = foot_terms(vg, soles, tensor(fw[sl]), support, frame_ok)
        terms.update(feet)
        hand_pen = []
        for ids, (a, b) in zip(hand_indices, ends):
            gap = surf.capsule_surface_residual(vw[:, ids], a, b, .016)
            hand_pen.append((torch.relu(-gap) / .003).square().mean())
        terms["hand_penetration"] = torch.stack(hand_pen).mean()
        return sum(coefficients[k] * v for k, v in terms.items()), terms, out, body, wp, wr, vg, active

    def metrics(state):
        _, terms, out, body, wp, wr, vg, active = state
        pos = torch.linalg.vector_norm(wp - target_p, dim=-1).detach().cpu().numpy() * 1000
        trace = torch.einsum("nsij,sij->ns", wr, target_R)
        angle = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1))).detach().cpu().numpy()
        return {"terms": {k: float(v.detach()) for k, v in terms.items()},
                "wrist_error_mm": {s: {"median": float(np.median(pos[:, j])), "p95": float(np.percentile(pos[:, j], 95)), "max": float(pos[:, j].max())} for j, s in enumerate(("left", "right"))},
                "wrist_angle_deg": {s: {"median": float(np.median(angle[:, j])), "p95": float(np.percentile(angle[:, j], 95))} for j, s in enumerate(("left", "right"))},
                "foot_tangential_active_pairs": active,
                "body_3d_rms_mm": float(torch.sqrt(terms["body_3d"]).detach()) * args.body_scale_m * 1000,
                "sole_minimum_z_mm": {s: float(vg[:, idx, 2].min().detach()) * 1000 for s, idx in zip(("left", "right"), soles)},
                "lower_body_unchanged": bool(torch.equal(body[:, frozen_lower], base[:, frozen_lower]))}

    initial = forward()
    initial_metrics = metrics(initial)
    gradients = {}
    for key, term in initial[1].items():
        g = torch.autograd.grad(coefficients[key] * term, (upper, torso), retain_graph=True, allow_unused=True)
        gradients[key] = [None if v is None else float(v.norm().detach()) for v in g]
    write("gradient_audit.json", {"parameters": ["upper", "torso"], "weighted_gradient_norms": gradients})
    history, checkpoints = [], {"initial": initial_metrics}
    for stage, steps in (("upper_only", args.upper_steps), ("upper_and_torso", args.body_steps)):
        torso.requires_grad_(stage == "upper_and_torso")
        optimizer = torch.optim.Adam([upper] + ([torso] if torso.requires_grad else []), lr=args.lr)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            state = forward()
            loss = state[0]
            if not torch.isfinite(loss):
                write("FAILURE.json", {"reason": "nonfinite_loss", "stage": stage, "step": step})
                raise RuntimeError("nonfinite refinement loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in (upper, torso)):
                raise RuntimeError("nonfinite parameter gradient")
            optimizer.step()
            if step % 25 == 0 or step == steps - 1:
                history.append({"stage": stage, "step": step, "loss": float(loss.detach()), "terms": {k: float(v.detach()) for k, v in state[1].items()}})
                print(stage, step, float(loss.detach()), flush=True)
        with torch.no_grad():
            state = forward()
            checkpoints[stage] = metrics(state)
            np.savez_compressed(args.output_dir / f"{stage}_parameters.npz",
                pair_id=np.arange(args.start, args.stop),
                upper_joint_ids=np.asarray(UPPER_JOINTS), torso_joint_ids=np.asarray(TORSO_JOINTS),
                upper_corrections=upper.detach().cpu().numpy(), torso_corrections=torso.detach().cpu().numpy(),
                body_rotation_matrices=state[3].cpu().numpy())
    with torch.no_grad():
        final = forward()
        _, _, out, body, wp, wr, vg, _ = final
        body_aa = Rotation.from_matrix(body.cpu().numpy().reshape(-1, 3, 3)).as_rotvec().reshape(n, 63)
        exported = {"pair_id": np.arange(args.start, args.stop), "vertices": out.vertices.cpu().numpy(),
            "faces": np.asarray(asset["f"]), "betas": z["betas"], "global_orient": z["global_orient"][sl],
            "transl": z["transl"][sl], "body_pose": body_aa, "body_rotation_matrices": body.cpu().numpy(),
            "predicted_coco": regress_coco17_torch(out.vertices, reg).cpu().numpy(),
            "smplh_joints": out.joints.cpu().numpy(), "wrist_walker_m": wp.cpu().numpy(),
            "wrist_rotation_walker": wr.cpu().numpy(), "vertices_ground_m": vg.cpu().numpy(),
            "pose_constructed": np.asarray(True), "accepted_for_main_fit": np.asarray(False)}
        for side in ("left", "right"):
            exported[f"{side}_hand_pose"] = hand_pose[side].cpu().numpy()
            exported[f"{side}_hand_pca"] = np.repeat(grasp["hands"][side]["hand_pca"], n, axis=0)
        np.savez_compressed(args.output_dir / "result.npz", **exported)
    write("fit_summary.json", {"status": "engineering_probe_requires_grasp_and_image_review",
        "checkpoints": checkpoints, "trace": history,
        "accepted_for_main_fit": False, "rollback": str(args.source_result.resolve()),
        "scope": "upper_and_torso_refinement; root/lower/fingers/beta fixed; no world-stationary hand constraint"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
