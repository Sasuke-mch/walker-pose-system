#!/usr/bin/env python3
"""Optional SMPL-H refinement of an immutable existing fit.

Rollback: use the source result and legacy fit_smplh_wilor_sequence.py.
This tool never changes them. Short windows are probes, not stitched outputs.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import PROJECT_ROOT as _tool_project_root
_tool_prepare_imports()

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = _tool_project_root


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
    ap.add_argument("--surface-refine", action="store_true", help="opt-in shared PCA + full-mesh self collision/shape preservation")
    ap.add_argument("--full-body", action="store_true", help="VPoser32 body latent; root released in joint phase, beta fixed")
    ap.add_argument("--balanced-stages", action="store_true", help="opt-in grouped observations, guarded updates and dedicated leg temporal terms")
    ap.add_argument("--constrained-update", action="store_true", help="project frame-local Adam body increments; exact full-window gates remain authoritative")
    ap.add_argument("--projection-trust-radius", type=float, default=.2, help="per-frame normalized increment radius; latent/radian scales 1, translation scale .05m")
    ap.add_argument("--nonlinear-correction-steps", type=int, default=0, help="opt-in relinearized feasibility recovery at rejected projected trials")
    ap.add_argument("--wrist-direction-diagnostics", action="store_true", help="read-only frame-local wrist descent probes at body checkpoints")
    ap.add_argument("--stage2-static-assumption", action="store_true", help="explicit user assumption: heel/ball patches static within each Stage2 segment")
    ap.add_argument("--vposer-dir", type=Path, help="required for full-body: frozen eval decoder with live latent gradients")
    ap.add_argument("--pose-reference-result", type=Path, help="immutable original VPoser result; required for full-body, including resumed runs")
    ap.add_argument("--vposer-prior-weight", type=float, default=.02)
    ap.add_argument("--body-anchor-weight", type=float, default=1.)
    ap.add_argument("--rotation-temporal-weight", type=float, default=.2)
    ap.add_argument("--collision-refresh", type=int, default=100)
    ap.add_argument("--wrist-bound-m", type=float, default=.01, help="surface-refine tail barrier; not a feasibility certificate")
    ap.add_argument("--wrist-bound-scale-m", type=float, default=.001, help="soft wrist-bound residual scale; does not change the bound or final gate")
    ap.add_argument("--body-polish-steps", type=int, default=0, help="body/wrist refinement with shared fingers frozen and self-collision diagnostic detached")
    ap.add_argument("--hand-polish-steps", type=int, default=0, help="full-surface finger refinement with body/wrists frozen")
    ap.add_argument("--lock-grasp-orientation", action="store_true", help="solve local wrists by parent FK to keep palm walker orientation fixed")
    args = ap.parse_args()
    if args.balanced_stages and not args.full_body:
        raise ValueError("balanced stages require live VPoser full-body")
    if args.constrained_update and not args.balanced_stages:
        raise ValueError("constrained updates require balanced full-body stages")
    if args.nonlinear_correction_steps < 0 or (args.nonlinear_correction_steps and not args.constrained_update):
        raise ValueError("nonlinear correction requires constrained updates and nonnegative steps")
    if args.wrist_direction_diagnostics and not args.balanced_stages:
        raise ValueError("wrist direction diagnostics require balanced full-body")
    if not np.isfinite(args.projection_trust_radius) or args.projection_trust_radius <= 0:
        raise ValueError("invalid projection trust radius")
    if not np.isfinite(args.wrist_bound_scale_m) or args.wrist_bound_scale_m <= 0:
        raise ValueError("invalid wrist bound scale")
    if args.stage2_static_assumption and not args.balanced_stages:
        raise ValueError("Stage2 assumption requires balanced stages")
    if args.output_dir.exists():
        raise ValueError("refuse existing output directory")
    # Never silently fall back from the latent body graph to legacy free SO3.
    if args.full_body and (args.vposer_dir is None or args.pose_reference_result is None):
        raise ValueError("unrestricted full-body refinement is disabled: active VPoser body parameterization and prior are required")
    if args.full_body and args.lock_grasp_orientation:
        raise ValueError("VPoser full-body forbids overriding decoded wrist rotations; use the soft orientation loss")
    if not np.isfinite([args.vposer_prior_weight, args.body_anchor_weight, args.rotation_temporal_weight]).all() or min(args.vposer_prior_weight, args.body_anchor_weight) <= 0 or args.rotation_temporal_weight < 0:
        raise ValueError("invalid body prior weights")
    if args.hand_polish_steps and not args.surface_refine:
        raise ValueError("hand polish requires --surface-refine")
    if args.wrist_bound_m <= 0 or args.collision_refresh <= 0 or min(args.upper_steps, args.body_steps, args.body_polish_steps, args.hand_polish_steps) < 0 or not np.isfinite(
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
    from pose_app.balanced_grasp import (GROUPS, LEG_ROTATIONS, grouped_observation_loss,
        observation_guard, leg_temporal_terms, build_stage2_anchors, stage2_terms, guarded_adam_step)
    from pose_app.constrained_grasp import constrained_adam_step, probe_local_goal_descent, wrist_bound_loss
    from pose_app.constructed_grasp_refinement import (
        UPPER_JOINTS, TORSO_JOINTS, load_grasp, validate_rotation, compose_body,
        wrist_rotations, masked_mean, foot_terms, frozen_surface_states,
        triangle_separation_loss, screened_sequence_pairs, align_wrist_rotations,
        validate_full_sequence_ids, raw_observation_weights)

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
    validate_full_sequence_ids(z.get("pair_id"), total_frames, "source", allow_legacy=True)
    validate_full_sequence_ids(labels.get("pair_id"), total_frames, "contact labels")
    reference = np.load(args.pose_reference_result, allow_pickle=False) if args.full_body else z
    validate_full_sequence_ids(reference.get("pair_id"), total_frames, "reference", allow_legacy=True)
    if reference["vertices"].shape != z["vertices"].shape or not np.allclose(reference["betas"], z["betas"], atol=1e-6):
        raise ValueError("immutable reference frame/beta mismatch")
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
    frame_ok_full = np.asarray(reference["body_accepted"]).any(1) & np.asarray(scene["accepted"]).any(1)
    # Resuming must not relabel support using the newly optimized mesh.
    vg_source = np.einsum("nij,nvj->nvi", Rg, reference["vertices"]) + tg[:, None]
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
    latent = None
    if args.full_body:
        from pose_app.smplx_fitting import load_vposer_explicit
        from pose_app.vposer_grasp_body import decode_body_rotations, rotation_anchor_loss, rotation_temporal_loss
        vposer, _, vp_checkpoint = load_vposer_explicit(args.vposer_dir, args.device)
        if "vposer_latent" not in z.files or "vposer_latent" not in reference.files:
            raise ValueError("source/reference requires original VPoser latent; rejected SO3 fits cannot initialize this route")
        if reference["body_pose"].shape != z["body_pose"].shape or not np.allclose(reference["betas"], z["betas"], atol=1e-6):
            raise ValueError("immutable pose reference frame/beta mismatch")
        latent = torch.nn.Parameter(tensor(z["vposer_latent"][sl]).clone())
        reference_latent = tensor(reference["vposer_latent"][sl])
        with torch.no_grad():
            reference_body = decode_body_rotations(vposer, reference_latent)
            decoded_source = decode_body_rotations(vposer, latent)
        if not torch.allclose(decoded_source, base, atol=1e-5) or not torch.allclose(reference_body, batch_rodrigues(tensor(reference["body_pose"][sl]).reshape(-1, 3)).reshape(n,21,3,3), atol=1e-5):
            raise ValueError("saved body pose is not the declared VPoser decode")
        reference_root = batch_rodrigues(tensor(reference["global_orient"][sl])).reshape(n,1,3,3)
        reference_translation = tensor(reference["transl"][sl])
    # Matrix layer must reproduce the original axis-angle forward, including
    # mean-once convention, before using any constructed targets.
    with torch.no_grad():
        original_hands = [batch_rodrigues(tensor(z[f"{s}_hand_pose"][sl]).reshape(-1, 3)).reshape(n, 15, 3, 3) for s in ("left", "right")]
        replay = model(betas=beta, global_orient=root, body_pose=base,
                       left_hand_pose=original_hands[0], right_hand_pose=original_hands[1], transl=transl)
        replay_max = float((replay.vertices - tensor(z["vertices"][sl])).abs().max())
        if replay_max > 1e-5:
            raise ValueError(f"matrix replay differs from source: {replay_max} m")
    hand_pose, hand_rot, hand_bases, hand_coeff = {}, {}, {}, {}
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
        hand_bases[side] = (tensor(np.asarray(mano["hands_mean"])), tensor(comp / np.linalg.norm(comp, axis=1)[:, None]))
        initial_coeff = z[f"{side}_hand_pca"][0:1] if args.surface_refine and "pose_constructed" in z.files and bool(z["pose_constructed"]) else h["hand_pca"]
        hand_coeff[side] = torch.nn.Parameter(tensor(initial_coeff).clone(), requires_grad=args.surface_refine)
    target_p = tensor([grasp["hands"][s]["wrist_walker_m"] for s in ("left", "right")])
    target_R = tensor([grasp["hands"][s]["rotation_walker_from_wrist"] for s in ("left", "right")])
    reg = load_coco17_regressor(ROOT / "models/smpl/J_regressor_coco.npy")
    # Reuse the authoritative raw reader; raw fish-eye pixels, no WiLoR pixels.
    from pose_app import body_observations as raw
    obs, observation_reasons = [], []
    cal = load_stereo_fisheye(ROOT / "realtime_app/calibration/results")
    for path, side in ((args.left_raw, "left"), (args.right_raw, "right")):
        rows = raw.raw_side(path, side)
        if sorted(rows) != list(range(total_frames)):
            raise ValueError("raw frame identity mismatch")
        points = np.stack([rows[i] for i in range(args.start, args.stop)])
        weights, reasons = raw_observation_weights(points, cal.image_width, cal.image_height)
        observation_reasons.append(reasons)
        # Mask before tensor conversion: nan_to_num(+inf) can overflow float32,
        # and multiplying an infinite residual by zero still yields NaN.
        obs.append((tensor(np.where(weights[..., None] > 0, points[..., :2], 0.)), tensor(weights)))
    # Recompute observations from this run's original paired rows. Source fit
    # supplies initialization only; previous triangulation is never supervision.
    left_rows, right_rows = raw.raw_side(args.left_raw, "left"), raw.raw_side(args.right_raw, "right")
    current_tri, _, _, _, _, _, current_accepted, _, current_quality, _ = raw.raw_triangulate(
        np.stack([left_rows[i] for i in range(total_frames)]).astype(np.float32),
        np.stack([right_rows[i] for i in range(total_frames)]).astype(np.float32), cal)
    target = tensor(np.nan_to_num(np.asarray(current_tri)[sl] / 1000))
    body_mask = torch.as_tensor(np.asarray(current_accepted)[sl] & np.isfinite(np.asarray(current_tri)[sl]).all(-1), device=device)
    quality = tensor(np.asarray(current_quality)[sl]).clamp_min(0)
    if not body_mask.any() or not torch.isfinite(quality).all() or quality[body_mask].sum() <= 0:
        raise ValueError("no valid current weighted body observations")
    scene_raw = np.asarray(scene["raw_triangulated_points_mm"])
    if scene_raw.shape != np.asarray(current_tri).shape or not np.allclose(scene_raw, current_tri, equal_nan=True, atol=.001):
        raise ValueError("scene/current observation source or frame alignment mismatch")
    R01, t01 = tensor(cal.R_cam0_to_cam1), tensor(cal.T_cam0_to_cam1_mm / 1000)
    upper = torch.nn.Parameter(torch.zeros(n, len(UPPER_JOINTS), 3, device=device), requires_grad=not args.full_body)
    torso = torch.nn.Parameter(torch.zeros(n, len(TORSO_JOINTS), 3, device=device), requires_grad=not args.full_body)
    frozen_lower = [j - 1 for j in range(1, 22) if j not in UPPER_JOINTS + TORSO_JOINTS]
    lower = torch.nn.Parameter(torch.zeros(n, len(frozen_lower), 3, device=device), requires_grad=False)
    root_delta = torch.nn.Parameter(torch.zeros(n, 1, 3, device=device))
    translation_delta = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    mesh_path = args.grasp_mesh or args.grasp.with_suffix(".npz")
    with np.load(mesh_path, allow_pickle=False) as mesh:
        hand_indices = [np.asarray(mesh[f"{s}_vertex_indices"]) for s in ("left", "right")]
        hand_faces = [torch.as_tensor(mesh[f"{s}_faces"], dtype=torch.long, device=device) for s in ("left", "right")]
        hand_local_reference = [tensor(mesh[f"{s}_vertices_local_m"]) for s in ("left", "right")]
        for s in ("left", "right"):
            if not np.allclose(mesh[f"{s}_hand_pca"], grasp["hands"][s]["hand_pca"], atol=1e-6):
                raise ValueError("constructed mesh/JSON parameter mismatch")
    if any(ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= 6890 for ids in hand_indices):
        raise ValueError("invalid hand surface indices")
    ends = [tensor([np.asarray(walker["nodes_walker_mm"][key]) / 1000 for key in walker["handle_segments"][s]]) for s in ("left", "right")]
    region_groups, palm_groups = [], []
    dom = np.asarray(asset["weights"]).argmax(1)
    rest, restj = np.asarray(asset["v_template"]), np.asarray(asset["J"])
    for idx, start, wrist in zip(hand_indices, (22, 37), (20, 21)):
        region_groups.append([np.flatnonzero((dom[idx] >= start + 3 * i + 1) & (dom[idx] < start + 3 * i + 3)) for i in range(5)])
        center = .5 * (restj[wrist] + restj[[start, start + 3, start + 6, start + 9]].mean(0))
        palm_groups.append(np.flatnonzero((dom[idx] == wrist) & (np.linalg.norm(rest[idx] - center, axis=1) < .03)))
    coefficients = {"body_3d": 1., "body_2d": .15, "pose_anchor": .05,
                    "body_temporal": .02, "wrist_position": args.wrist_weight,
                    "wrist_rotation": args.orientation_weight, "hand_penetration": 20.,
                    "foot_contact": 1., "foot_nonpenetration": 1., "foot_tangential": .001}
    if args.surface_refine:
        coefficients.update(hand_self_collision=50., hand_surface_shape=.1, hand_pose_reference=.5,
                            hand_region_contact=3., hand_palm_contact=2., hand_opposition=10., wrist_bound=10.)
    if args.full_body:
        coefficients.update(root_reference=1.)
        coefficients.update(pose_anchor=args.body_anchor_weight, vposer_prior=args.vposer_prior_weight,
                            rotation_temporal=args.rotation_temporal_weight)
    active_collision_pairs = [torch.empty((0, 3), dtype=torch.long, device=device) for _ in range(2)]
    phase = "initial"
    wrist_ramp = 1.
    if args.balanced_stages:
        coefficients.update(leg_rotation_temporal=.2, leg_ground_acceleration=.05, upper_leg_anchor=1.)
    if args.stage2_static_assumption:
        anchor_ids, anchor_targets, anchor_segments, anchor_records = build_stage2_anchors(
            vg_source, scene["stage"], frame_ok_full,
            [sets[f"{s}_sole_surface_candidate"] for s in ("left", "right")])
        anchor_ids_t = torch.as_tensor(anchor_ids[sl], dtype=torch.long, device=device)
        anchor_targets_t = tensor(anchor_targets[sl])
        anchor_segments_t = torch.as_tensor(anchor_segments[sl], device=device)
        coefficients.update(stage2_position=1., stage2_velocity=.1)
    args.output_dir.mkdir(parents=True)
    write("run_metadata.json", {"inputs": {k: str(v.resolve()) for k, v in vars(args).items() if isinstance(v, Path)},
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "coefficients": coefficients, "coordinate_frame": "camera0 fit; walker-local grip; same-source ground feet",
        "scope": "constructed_grasp_body_refinement_engineering_probe", "fps_assumed": 30,
        "frozen": ["beta", "calibration", "scene"] + ([] if args.surface_refine else ["finger_PCA"]) + ([] if args.full_body else ["root", "translation", "lower_limb"]),
        "foot_loss_scale_m": .01 if args.full_body else None,
        "foot_nonpenetration_margin_m": 0. if args.balanced_stages else .003,
        "source_matrix_replay_max_error_m": replay_max,
        "grasp_mesh_source": str(mesh_path.resolve()),
        "body_3d_scale_m": args.body_scale_m,
        "body_parameterization": "VPoser32_decode_all21_no_override" if args.full_body else "legacy_SO3_diagnostic_only",
        "vposer_checkpoint": str(vp_checkpoint) if args.full_body else None,
        "support_source": "frozen same-source baseline sole surface state machine; not measured support",
        "support_reference_result": str((args.pose_reference_result or args.source_result).resolve()),
        "collision_screen_scope": "all frames in every active optimization stage",
        "foot_tangent_unavailable_reason": None if support_full[sl].any() else "no_sticking_surface_candidates"})
    np.savez_compressed(args.output_dir / "frozen_foot_states.npz", pair_id=np.arange(args.start, args.stop),
        support=support_full[sl], state=states_full[sl], reason=reasons_full[sl], frame_ok=frame_ok_full[sl])
    np.savez_compressed(args.output_dir / "body_2d_observation_audit.npz", pair_id=np.arange(args.start, args.stop),
                       left_reason=observation_reasons[0], right_reason=observation_reasons[1],
                       left_weight=obs[0][1].cpu().numpy(), right_weight=obs[1][1].cpu().numpy())
    if args.stage2_static_assumption:
        write("stage2_anchor_assumption.json", {"records": anchor_records,
            "source": "immutable reference representative foot frame; individual patch vertices; rigid XY shift and Z lift to plane",
            "schema_version": 2,
            "boundary_source": "frozen automatic scene stage; not independent verification",
            "measured_contact": False})
        np.savez_compressed(args.output_dir / "stage2_anchors.npz", pair_id=np.arange(args.start,args.stop),
            patch_ids=anchor_ids[sl], targets_m=anchor_targets[sl], segment=anchor_segments[sl])

    def forward():
        if args.full_body:
            body = decode_body_rotations(vposer, latent)
        else:
            body = compose_body(base, upper, UPPER_JOINTS)
            body = compose_body(body, torso, TORSO_JOINTS)
        current_root = root @ batch_rodrigues(root_delta.reshape(-1, 3)).reshape(n, 1, 3, 3) if args.full_body else root
        current_translation = transl + translation_delta if args.full_body else transl
        if args.lock_grasp_orientation:
            body = align_wrist_rotations(current_root[:, 0], body, model.parents, Rct @ target_R)
        if args.surface_refine:
            for s in ("left", "right"):
                mean, basis = hand_bases[s]
                hand_pose[s] = (mean + hand_coeff[s] @ basis).expand(n, -1)
                hand_rot[s] = batch_rodrigues(hand_pose[s].reshape(-1, 3)).reshape(n, 15, 3, 3)
        out = model(betas=beta, global_orient=current_root, body_pose=body,
                    left_hand_pose=hand_rot["left"], right_hand_pose=hand_rot["right"],
                    transl=current_translation, return_verts=True)
        coco = regress_coco17_torch(out.vertices, reg)
        vg = torch.einsum("nij,nvj->nvi", Rgt, out.vertices) + tgt[:, None]
        vw = (out.vertices - tct) @ Rct
        wp = (out.joints[:, [20, 21]] - tct) @ Rct
        wr = Rct.T @ wrist_rotations(current_root[:, 0], body, model.parents)
        pos = torch.linalg.vector_norm(wp - target_p, dim=-1)
        rot_diff = wr - target_R
        # Frobenius chordal rotation residual ~ angle near zero, no acos gradient.
        terms = {"wrist_position": ((pos / .005) ** 2).mean(),
                 "wrist_rotation": (rot_diff.square().sum((-2, -1)) / (2 * np.deg2rad(5) ** 2)).mean(),
                 "pose_anchor": rotation_anchor_loss(body, reference_body) if args.full_body else (upper.square().mean() + torso.square().mean()),
                 "body_3d": ((coco - target).square().sum(-1) * quality)[body_mask].sum() / quality[body_mask].sum().clamp_min(1e-6) / args.body_scale_m ** 2}
        if args.surface_refine:
            terms["wrist_bound"] = wrist_bound_loss(pos, args.wrist_bound_m, args.wrist_bound_scale_m)
        projected = [fisheye_project_torch(coco, cal.K0, cal.D0),
                     fisheye_project_torch(coco @ R01.T + t01, cal.K1, cal.D1)]
        terms["body_2d"] = torch.stack([((surf.robust_scalar(torch.linalg.vector_norm(p - xy, dim=-1) / 100, 1.) * w).sum() / w.sum().clamp_min(1e-6)) for p, (xy, w) in zip(projected, obs)]).mean()
        rel = coco - (coco[:, 11:12] + coco[:, 12:13]) / 2
        acceleration = rel[2:] - 2 * rel[1:-1] + rel[:-2]
        valid3 = body_mask[2:] & body_mask[1:-1] & body_mask[:-2] & (frame_ok[2:] & frame_ok[1:-1] & frame_ok[:-2])[:, None]
        terms["body_temporal"] = masked_mean(surf.robust_scalar(torch.linalg.vector_norm(acceleration, dim=-1), .03), valid3)
        if args.full_body:
            # Dimensionless 30mm second-difference scale; the old meter-valued
            # scalar made temporal weight .02 almost ineffective.
            terms["body_temporal"] = terms["body_temporal"] / .03 ** 2
            terms["rotation_temporal"] = rotation_temporal_loss(body, frame_ok)
            terms["vposer_prior"] = latent.square().mean()
        if args.balanced_stages:
            terms["body_3d"], _ = grouped_observation_loss(coco, target, quality, body_mask, args.body_scale_m)
            ground_coco = torch.einsum("nij,nkj->nki", Rgt, coco) + tgt[:,None]
            terms["leg_rotation_temporal"], terms["leg_ground_acceleration"] = leg_temporal_terms(body, ground_coco, frame_ok)
            terms["upper_leg_anchor"] = rotation_anchor_loss(body[:,LEG_ROTATIONS], reference_body[:,LEG_ROTATIONS]) if phase == "upper_only" else body.sum()*0
        if args.stage2_static_assumption:
            terms["stage2_position"], terms["stage2_velocity"] = stage2_terms(vg, anchor_ids_t, anchor_targets_t, anchor_segments_t)
        feet, active = foot_terms(vg, soles, tensor(fw[sl]), support, frame_ok,
            nonpenetration_margin_m=0. if args.balanced_stages else .003)
        terms.update(feet)
        if args.full_body:
            terms["foot_contact"] = terms["foot_contact"] / .01 ** 2
            terms["foot_nonpenetration"] = terms["foot_nonpenetration"] / .01 ** 2
            terms["root_reference"] = ((current_translation-reference_translation) / .05).square().mean() + rotation_anchor_loss(current_root, reference_root, scale_rad=.15)
        hand_pen, region_contact, palm_contact, opposition = [], [], [], []
        for ids, (a, b), groups, palm_group in zip(hand_indices, ends, region_groups, palm_groups):
            gap = surf.capsule_surface_residual(vw[:, ids], a, b, .016)
            pen = (torch.relu(-gap) / .003).square()
            hand_pen.append(pen.mean() + (pen.amax(1).mean() if args.surface_refine else pen.mean() * 0))
            if args.surface_refine:
                region_contact.append(torch.stack([gap[:, g].square().amin(1) / .005 ** 2 for g in groups]).mean())
                palm_contact.append((gap[:, palm_group].square().amin(1) / .005 ** 2).mean())
                points = vw[:, ids]
                chosen = torch.stack([points[torch.arange(n, device=device), torch.as_tensor(g, device=device)[gap[:, g].square().argmin(1)]] for g in groups], dim=1)
                axis = torch.nn.functional.normalize(b - a, dim=0)
                radial = chosen - a - ((chosen - a) * axis).sum(-1, keepdim=True) * axis
                radial = torch.nn.functional.normalize(radial, dim=-1)
                other = torch.nn.functional.normalize(radial[:, :4].mean(1), dim=-1)
                opposition.append(torch.relu((radial[:, 4] * other).sum(-1) - .2).square().mean())
        terms["hand_penetration"] = torch.stack(hand_pen).mean()
        if args.surface_refine:
            local_surfaces = [(vw[:, ids] - wp[:, j, None]) @ wr[:, j] for j, ids in enumerate(hand_indices)]
            terms["hand_surface_shape"] = torch.stack([((q - ref) / .005).square().mean() for q, ref in zip(local_surfaces, hand_local_reference)]).mean()
            terms["hand_self_collision"] = torch.stack([triangle_separation_loss(q, faces, pairs) for q, faces, pairs in zip(local_surfaces, hand_faces, active_collision_pairs)]).mean()
            terms["hand_pose_reference"] = torch.stack([(hand_coeff[s] - tensor(grasp["hands"][s]["hand_pca"])).square().mean() for s in ("left", "right")]).mean()
            terms["hand_region_contact"] = torch.stack(region_contact).mean()
            terms["hand_palm_contact"] = torch.stack(palm_contact).mean()
            terms["hand_opposition"] = torch.stack(opposition).mean()
        return sum(coefficients[k] * v * (wrist_ramp if k in ("wrist_position", "wrist_rotation", "wrist_bound") else 1.) for k, v in terms.items()), terms, out, body, wp, wr, vg, active

    def metrics(state):
        _, terms, out, body, wp, wr, vg, active = state
        pos = torch.linalg.vector_norm(wp - target_p, dim=-1).detach().cpu().numpy() * 1000
        trace = torch.einsum("nsij,sij->ns", wr, target_R)
        angle = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1))).detach().cpu().numpy()
        predicted = regress_coco17_torch(out.vertices, reg)
        error = torch.linalg.vector_norm(predicted-target, dim=-1)
        groups = {key: {"rms_mm": float(torch.sqrt(error[:,ids][body_mask[:,ids]].square().mean()).detach())*1000,
                        "max_mm": float(error[:,ids][body_mask[:,ids]].max().detach())*1000} if body_mask[:,ids].any() else None for key,ids in GROUPS.items()}
        return {"terms": {k: float(v.detach()) for k, v in terms.items()}, "body_groups": groups,
                "wrist_error_mm": {s: {"median": float(np.median(pos[:, j])), "p95": float(np.percentile(pos[:, j], 95)), "max": float(pos[:, j].max())} for j, s in enumerate(("left", "right"))},
                "wrist_angle_deg": {s: {"median": float(np.median(angle[:, j])), "p95": float(np.percentile(angle[:, j], 95))} for j, s in enumerate(("left", "right"))},
                "foot_tangential_active_pairs": active,
                "body_3d_rms_mm": float(torch.sqrt((error.square()*quality)[body_mask].sum()/quality[body_mask].sum().clamp_min(1e-6)).detach()) * 1000,
                "sole_minimum_z_mm": {s: float(vg[:, idx, 2].min().detach()) * 1000 for s, idx in zip(("left", "right"), soles)},
                "lower_body_unchanged": bool(torch.equal(body[:, frozen_lower], base[:, frozen_lower]))}

    initial = forward()
    if args.balanced_stages:
        # Reconstruct the same immutable baseline even when source is a resume.
        # Use original constructed PCA, not the resumed optimized fingers.
        with torch.no_grad():
            reference_hands = [batch_rodrigues(tensor(grasp['hands'][s]['hand_pose_axis_angle']).reshape(-1,3)).reshape(1,15,3,3).expand(n,-1,-1,-1) for s in ('left','right')]
            reference_out = model(betas=beta,global_orient=reference_root,body_pose=reference_body,
                left_hand_pose=reference_hands[0],right_hand_pose=reference_hands[1],transl=reference_translation)
            baseline_coco = regress_coco17_torch(reference_out.vertices,reg).detach().clone()
    else:
        baseline_coco = regress_coco17_torch(initial[2].vertices, reg).detach().clone()
    baseline_error = torch.linalg.vector_norm(baseline_coco-target, dim=-1)
    # Every accepted triangulated point is guarded; quality only weights losses.
    guard_mask = body_mask.clone()
    def local_constraints():
        # Constraints are frame-local even though the objective couples frames.
        local = forward()
        predicted = regress_coco17_torch(local[2].vertices, reg)
        safe_target = torch.where(guard_mask[...,None],target,predicted.detach())
        error = torch.linalg.vector_norm(predicted-safe_target,dim=-1)
        observation = torch.where(guard_mask,error-baseline_error-.01-1e-6,-torch.ones_like(error))
        if phase == "upper_only":
            leg = torch.linalg.vector_norm(predicted[:,11:]-baseline_coco[:,11:],dim=-1)-.005-1e-6
            return torch.cat((observation,leg),dim=1)
        return observation

    def wrist_goals():
        return (forward()[4]-target_p).square().sum(dim=-1)

    def save_direction_probe(name, selected):
        nonlocal phase
        if args.wrist_direction_diagnostics:
            scales = [(.05 if p is translation_delta else 1.) for p in selected]
            saved_phase = phase
            if phase == 'initial':
                phase = 'upper_only'
            try:
                probe = probe_local_goal_descent(selected,scales,local_constraints,wrist_goals)
                probe['constraint_phase'] = phase
            finally:
                phase = saved_phase
            probe['parameter_blocks'] = ['translation' if p is translation_delta else 'root' if p is root_delta else 'latent' for p in selected]
            for record in probe.get('frames',[]):
                record['pair_id'] = record['frame_index']+args.start
            write(f'wrist_direction_{name}.json',probe)

    def feasibility_diagnostic(state):
        predicted = regress_coco17_torch(state[2].vertices, reg)
        joints = state[2].joints
        target_camera = target_p @ Rct.T + tct
        records = []
        for side,ids,coco_id,column in (("left",(16,18,20),9,0),("right",(17,19,21),10,1)):
            shoulder,elbow,wrist = [joints[:,i] for i in ids]
            l1 = torch.linalg.vector_norm(elbow-shoulder,dim=-1)
            l2 = torch.linalg.vector_norm(wrist-elbow,dim=-1)
            reach = torch.linalg.vector_norm(target_camera[column]-shoulder,dim=-1)
            semantic = torch.linalg.vector_norm(wrist-predicted[:,coco_id],dim=-1)
            for i in range(n):
                records.append(dict(pair_id=i+args.start,side=side,upper_arm_m=float(l1[i]),forearm_m=float(l2[i]),
                    shoulder_to_target_m=float(reach[i]),outer_reach_excess_m=float(torch.relu(reach[i]-l1[i]-l2[i])),
                    inner_reach_excess_m=float(torch.relu((l1[i]-l2[i]).abs()-reach[i])),
                    model_joint_vs_coco_wrist_m=float(semantic[i])))
        error = torch.linalg.vector_norm(predicted-target,dim=-1)
        return dict(scope="current-shoulder two-link necessary reach condition; not global infeasibility or anatomical feasibility",
                    wrist_semantics="SMPL-H joint20/21 vs surface COCO9/10; offsets are diagnostics, not correction labels",
                    global_feasibility="unknown",records=records,
                    near_observation_boundary_pair_joint=[[int(i)+args.start,int(j)] for i,j in torch.nonzero(guard_mask & (error-baseline_error>.009)).cpu().tolist()])
    with torch.no_grad():
        write("feasibility_initial.json",feasibility_diagnostic(initial))
    np.savez_compressed(args.output_dir / "observation_guard_reference.npz", pair_id=np.arange(args.start,args.stop),
        baseline_error_m=baseline_error.cpu().numpy(), predicted_coco_m=baseline_coco.cpu().numpy(),
        valid=guard_mask.cpu().numpy(), allowance_m=.01)
    if args.balanced_stages:
        initial_ok, initial_excess = observation_guard(regress_coco17_torch(initial[2].vertices,reg),target,guard_mask,baseline_error)
        if not initial_ok:
            write('FAILURE.json', {'reason':'resume_outside_immutable_observation_guard','maximum_excess_m':initial_excess})
            raise ValueError('source outside immutable reference observation budget; refusing to reset budget')
    initial_metrics = metrics(initial)
    gradients = {}
    audit_parameters = (latent, root_delta, translation_delta) if args.full_body else (upper, torso)
    for key, term in initial[1].items():
        g = torch.autograd.grad(coefficients[key] * term, audit_parameters, retain_graph=True, allow_unused=True)
        gradients[key] = [None if v is None else float(v.norm().detach()) for v in g]
    write("gradient_audit.json", {"parameters": ["vposer_latent", "root", "translation"] if args.full_body else ["upper", "torso"], "weighted_gradient_norms": gradients})
    history, checkpoints, transactions = [], {"initial": initial_metrics}, []
    del initial
    # The reversible probe changes tensor version counters. Run only after
    # all derivatives of the initial graph have been consumed.
    save_direction_probe('initial',[latent])
    for stage, steps in (("upper_only", args.upper_steps), ("upper_and_torso", args.body_steps),
                         ("body_wrist_polish", args.body_polish_steps), ("hand_surface_polish", args.hand_polish_steps)):
        if steps == 0:
            continue
        phase = stage
        upper.requires_grad_(not args.full_body and stage != "hand_surface_polish")
        torso.requires_grad_(not args.full_body and stage in ("upper_and_torso", "body_wrist_polish"))
        for p in (root_delta, translation_delta):
            p.requires_grad_(args.full_body and stage in ("upper_and_torso", "body_wrist_polish"))
        for p in hand_coeff.values():
            p.requires_grad_(args.surface_refine and (stage == "hand_surface_polish" if args.balanced_stages else stage != "body_wrist_polish"))
        if args.full_body:
            latent.requires_grad_(stage != "hand_surface_polish")
            params = ([latent] if latent.requires_grad else []) + [p for p in (root_delta, translation_delta) if p.requires_grad]
        else:
            params = ([upper] if upper.requires_grad else []) + ([torso] if torso.requires_grad else [])
        params += [p for p in hand_coeff.values() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr * (.25 if args.balanced_stages and stage == "body_wrist_polish" else 1.))
        for step in range(steps):
            wrist_ramp = (.1 + .9*(step+1)/steps) if args.balanced_stages and stage == "upper_only" else 1.
            optimizer.zero_grad(set_to_none=True)
            state = forward()
            if args.surface_refine and step % args.collision_refresh == 0:
                for j, ids in enumerate(hand_indices):
                    points = state[2].vertices[:, ids].detach().cpu().numpy()
                    pairs = screened_sequence_pairs(points, hand_faces[j].cpu().numpy())
                    active_collision_pairs[j] = torch.as_tensor(pairs, dtype=torch.long, device=device).reshape(-1, 3)
                state = forward()
            loss = state[0]
            if args.surface_refine and stage == "body_wrist_polish" and not args.full_body:
                collision = coefficients["hand_self_collision"] * state[1]["hand_self_collision"]
                loss = loss - collision + collision.detach()
            if not torch.isfinite(loss):
                write("FAILURE.json", {"reason": "nonfinite_loss", "stage": stage, "step": step})
                raise RuntimeError("nonfinite refinement loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError("nonfinite parameter gradient")
            if args.balanced_stages:
                before_wrist_squared = (state[4].detach()-target_p).square().sum(dim=-1).sum(dim=0).cpu().tolist()
                def evaluate_trial():
                    trial = forward()
                    predicted = regress_coco17_torch(trial[2].vertices, reg)
                    ok, excess = observation_guard(predicted, target, guard_mask, baseline_error)
                    excess_all = torch.where(guard_mask, torch.linalg.vector_norm(predicted-target,dim=-1)-baseline_error-.01,
                                             torch.full_like(baseline_error,-float("inf")))
                    bad = torch.nonzero(excess_all > 1e-6).cpu().tolist()
                    leg_shift = float(torch.linalg.vector_norm(predicted[:,11:]-baseline_coco[:,11:], dim=-1).max())
                    if stage == "upper_only":
                        ok = ok and leg_shift <= .005 + 1e-6
                    pca_ok = all(bool(((hand_coeff[s]-tensor(grasp["hands"][s]["hand_pca"])).abs() <= 2.).all()) for s in ("left","right"))
                    trial_trace = torch.einsum("nsij,sij->ns", trial[5], target_R)
                    trial_angle = torch.rad2deg(torch.acos(((trial_trace-1)/2).clamp(-1,1)))
                    return trial[0], ok and pca_ok, {"observation_excess_m": excess, "leg_shift_m": leg_shift,
                        "pca_ok": pca_ok, "violating_pair_joint": [[i+args.start,j] for i,j in bad],
                        "wrist_squared_error_sum_m2": (trial[4]-target_p).square().sum(dim=-1).sum(dim=0).cpu().tolist(),
                        "wrist_angle_mean_deg": trial_angle.mean(dim=0).cpu().tolist(),
                        "weighted_terms": {key: float(coefficients[key]*value*(wrist_ramp if key in ("wrist_position","wrist_rotation","wrist_bound") else 1.)) for key,value in trial[1].items()},
                        "upper_leg_shift_guard_passed": stage != "upper_only" or leg_shift <= .005+1e-6}
                try:
                    if args.constrained_update and stage != "hand_surface_polish":
                        scales = [(.05 if p is translation_delta else 1.) for p in params]
                        transaction = constrained_adam_step(optimizer,params,evaluate_trial,loss.detach(),
                            local_constraints,scales,trust_radius=args.projection_trust_radius,
                            correction_steps=args.nonlinear_correction_steps)
                        for record in transaction['projection']['frames']:
                            record['pair_id'] = record['frame_index']+args.start
                    else:
                        transaction = guarded_adam_step(optimizer, params, evaluate_trial, loss.detach())
                except Exception as exc:
                    write('FAILURE.json', {'reason':'guarded_update_exception', 'stage':stage,
                        'step':step, 'exception_type':type(exc).__name__, 'message':str(exc),
                        'parameters_and_optimizer_restored':True})
                    write('update_transactions_before_failure.json', {'updates':transactions})
                    raise
                transactions.append({"stage":stage, "step":step, "wrist_ramp":wrist_ramp,
                                     "before_wrist_squared_error_sum_m2":before_wrist_squared,
                                     "before_loss":float(loss.detach()),
                                     "before_weighted_terms":{key:float(coefficients[key]*value.detach()*(wrist_ramp if key in ("wrist_position","wrist_rotation","wrist_bound") else 1.)) for key,value in state[1].items()},
                                     **transaction})
            else:
                optimizer.step()
            if args.surface_refine and not args.balanced_stages:
                with torch.no_grad():
                    for s in ("left", "right"):
                        reference = tensor(grasp["hands"][s]["hand_pca"])
                        hand_coeff[s].copy_(torch.maximum(torch.minimum(hand_coeff[s], reference + 2.), reference - 2.))
            if step % 25 == 0 or step == steps - 1:
                history.append({"stage": stage, "step": step, "loss": float(loss.detach()),
                    "evaluation_point": "before_proposed_update", "wrist_ramp": wrist_ramp,
                    "terms": {k: float(v.detach()) for k, v in state[1].items()}})
                print(stage, step, float(loss.detach()), flush=True)
        if stage != 'hand_surface_polish':
            save_direction_probe(stage,params)
        with torch.no_grad():
            state = forward()
            checkpoints[stage] = metrics(state)
            write(f"feasibility_{stage}.json",feasibility_diagnostic(state))
            np.savez_compressed(args.output_dir / f"{stage}_parameters.npz",
                pair_id=np.arange(args.start, args.stop),
                upper_joint_ids=np.asarray(UPPER_JOINTS), torso_joint_ids=np.asarray(TORSO_JOINTS),
                upper_corrections=upper.detach().cpu().numpy(), torso_corrections=torso.detach().cpu().numpy(),
                body_rotation_matrices=state[3].cpu().numpy(),
                root_delta=root_delta.detach().cpu().numpy(), translation_delta=translation_delta.detach().cpu().numpy(),
                lower_corrections=lower.detach().cpu().numpy(),
                left_hand_pca=hand_coeff["left"].detach().cpu().numpy(), right_hand_pca=hand_coeff["right"].detach().cpu().numpy())
            if args.full_body:
                np.savez_compressed(args.output_dir / f"{stage}_vposer_parameters.npz",
                                    pair_id=np.arange(args.start,args.stop), vposer_latent=latent.detach().cpu().numpy(),
                                    body_rotation_matrices=state[3].cpu().numpy())
    with torch.no_grad():
        final = forward()
        _, _, out, body, wp, wr, vg, _ = final
        if args.full_body:
            decoded_final = decode_body_rotations(vposer, latent)
            if not torch.equal(decoded_final, body):
                raise RuntimeError("body rotations were modified after VPoser decode")
        body_aa = Rotation.from_matrix(body.cpu().numpy().reshape(-1, 3, 3)).as_rotvec().reshape(n, 63)
        exported = {"pair_id": np.arange(args.start, args.stop), "vertices": out.vertices.cpu().numpy(),
            "faces": np.asarray(asset["f"]), "betas": z["betas"], "global_orient": Rotation.from_matrix((root @ batch_rodrigues(root_delta.reshape(-1, 3)).reshape(n, 1, 3, 3) if args.full_body else root).cpu().numpy().reshape(-1, 3, 3)).as_rotvec().astype(np.float32),
            "transl": (transl + translation_delta if args.full_body else transl).cpu().numpy(), "body_pose": body_aa.astype(np.float32), "body_rotation_matrices": body.cpu().numpy(),
            "predicted_coco": regress_coco17_torch(out.vertices, reg).cpu().numpy(),
            "smplh_joints": out.joints.cpu().numpy(), "wrist_walker_m": wp.cpu().numpy(),
            "wrist_rotation_walker": wr.cpu().numpy(), "vertices_ground_m": vg.cpu().numpy(),
            "pose_constructed": np.asarray(True), "accepted_for_main_fit": np.asarray(False)}
        from pose_app.smplh_hand_observation import hand21
        exported.update(raw_triangulated_points=np.asarray(current_tri)[sl], body_accepted=np.asarray(current_accepted)[sl], body_quality=np.asarray(current_quality)[sl],
                        hand_points_left=hand21(out.joints, out.vertices, "left").cpu().numpy(), hand_points_right=hand21(out.joints, out.vertices, "right").cpu().numpy())
        for side in ("left", "right"):
            exported[f"{side}_hand_pose"] = hand_pose[side].cpu().numpy()
            exported[f"{side}_hand_pca"] = np.repeat(hand_coeff[side].detach().cpu().numpy(), n, axis=0)
        if args.full_body:
            exported["vposer_latent"] = latent.detach().cpu().numpy()
            exported["body_parameterization"] = np.asarray("VPoser32_decode_all21_no_override")
        np.savez_compressed(args.output_dir / "result.npz", **exported)
    write("fit_summary.json", {"status": "engineering_probe_requires_grasp_and_image_review",
        "checkpoints": checkpoints, "trace": history,
        "accepted_for_main_fit": False, "rollback": str(args.source_result.resolve()),
        "scope": "full_body_with_shared_PCA_and_surface_constraints" if args.full_body else "upper_and_torso_refinement",
        "full_body": args.full_body, "shared_PCA_refinement": args.surface_refine,
        "wrist_target_frame": "walker_rigid_local", "beta_fixed": True})
    if args.balanced_stages:
        write("update_transactions.json", {"updates": transactions,
            "observation_allowance_m": .01, "upper_phase_leg_shift_limit_m": .005,
            "guard": "every accepted current triangulated point vs immutable pose-reference with original constructed fingers",
            "reference_result": str(args.pose_reference_result.resolve()),
            "rejected_updates_restore_parameters_and_Adam_moments": True,
            "accepted_backtracks_keep_proposed_Adam_moments": True,
            "body_update_rule": ("projected_Adam_nonlinear_recovery_exact_guard" if args.nonlinear_correction_steps else
                                 "projected_Adam_exact_nonlinear_guard") if args.constrained_update else "guarded_Adam",
            "projection_constraint_columns": "0..16 COCO observation budget; upper_only 17..22 COCO11..16 leg displacement"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
