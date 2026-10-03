"""SMPL-H run orchestration: input audits, model setup, optimization and output.

run_fit accepts the existing parsed CLI namespace. It preserves stage order,
artifact schemas and failure gates; numerical components live beside it.
"""
from __future__ import annotations

import inspect
import json

import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _value in {"bool": np.bool_, "int": np.int64, "float": np.float64,
                      "complex": np.complex128, "object": np.object_,
                      "unicode": np.str_, "str": np.str_}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)


from pose_app.fisheye_camera import fisheye_project_torch, load_stereo_fisheye
from pose_app.smpl_coco_observation import load_coco17_regressor, regress_coco17_torch
from pose_app.smplx_fitting import load_vposer_explicit
from pose_app import smpl_surface_contact as surface_contact

from pose_app import body_observations as raw_clean


from ..smplh_hand_observation import read_wilor, finite_pixel_values
from ..project_paths import PROJECT_ROOT as ROOT
from .initialization import initialize_rigid_body
from .stages import (build_stage_schedule, set_trainable_parameters,
                     validate_optimization_options, require_finite_tensors)
from .losses import (body_reprojection_loss, hand_temporal_loss, body_temporal_loss,
                     weighted_hand_residual)
from .contact import load_surface_targets, surface_contact_losses


def run_fit(args) -> int:
    validate_optimization_options(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refuse non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import smplx
    import pickle
    from smplx.utils import Struct
    from scipy.spatial.transform import Rotation
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.vposer_dir is None:
        raise ValueError("--vposer-dir is required: SMPL-H body pose is parameterized by VPoser")
    if args.surface_hand_contact_weight < 0 or args.surface_foot_contact_weight < 0:
        raise ValueError("surface contact weights must be non-negative")
    if args.global_hand_handle_weight < 0:
        raise ValueError("global hand-handle weight must be non-negative")
    if args.body_temporal_weight < 0 or args.body_reprojection_weight < 0:
        raise ValueError("body temporal/reprojection weights must be non-negative")
    if not np.isfinite([args.mano_pose_weight, args.hand_2d_weight]).all() or min(args.mano_pose_weight, args.hand_2d_weight) < 0:
        raise ValueError("MANO pose and hand 2D weights must be finite and non-negative")
    if args.shared_hand_pose and not (args.mano_pose_init and args.mano_pose_weight > 0):
        raise ValueError("shared hand solve requires native MANO init and a positive pose weight")
    if args.body_reprojection_scale_px <= 0:
        raise ValueError("--body-reprojection-scale-px must be positive")
    if not np.isfinite(args.wrist_reference_weight) or args.wrist_reference_weight < 0:
        raise ValueError("invalid wrist reference weight")
    if (args.wrist_reference is not None) != (args.wrist_reference_weight > 0):
        raise ValueError("wrist reference and positive weight must be supplied together")
    wrist_target = None
    wrist_audit = {"enabled": False}
    if args.wrist_reference is not None:
        from pose_app.independent_wrist_reference import load_reference, wrist_position_loss
        wrist_np, wrist_audit = load_reference(args.wrist_reference, args.calibration_dir,
                                             args.allow_diagnostic_wrist_reference)
        wrist_target = torch.tensor(wrist_np, device=device)
        wrist_audit.update(enabled=True, weight=args.wrist_reference_weight, stages=["A", "B", "C"])
        (args.output_dir / "wrist_reference_audit.json").write_text(
            json.dumps(wrist_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_args = (args.contact_labels, args.scene_transforms,
                    args.contact_vertex_sets, args.walker_topology)
    if any(v is not None for v in contact_args) and not all(v is not None for v in contact_args):
        raise ValueError("contact mode requires contact-labels, scene-transforms, contact-vertex-sets and walker-topology together")
    if all(v is not None for v in contact_args) and (args.surface_hand_contact_weight + args.surface_foot_contact_weight) <= 0.0:
        raise ValueError("contact inputs supplied but both surface contact weights are zero")
    if args.global_hand_handle_pose is not None and not all(v is not None for v in contact_args):
        raise ValueError("global hand-handle prior requires contact inputs")

    cal = load_stereo_fisheye(args.calibration_dir)
    reg = load_coco17_regressor(args.regressor)
    left_rows = raw_clean.raw_side(args.left_raw, "left")
    right_rows = raw_clean.raw_side(args.right_raw, "right")
    if set(left_rows) != set(right_rows):
        raise ValueError("left/right raw frame sets differ; refusing silent frame deletion")
    ids = sorted(left_rows)
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
    hand_ll, hand_ll_valid, hand_ll_bounds, hand_ll_weights, audit_left_cam_left = read_wilor(
        args.wilor_left, n, "left", left, image_size, "left", args.hand_2d_weight > 0)
    hand_lr, hand_lr_valid, hand_lr_bounds, hand_lr_weights, audit_left_cam_right = read_wilor(
        args.wilor_left, n, "right", left, image_size, "left", args.hand_2d_weight > 0)
    hand_rl, hand_rl_valid, hand_rl_bounds, hand_rl_weights, audit_right_cam_left = read_wilor(
        args.wilor_right, n, "left", right, image_size, "right", args.hand_2d_weight > 0)
    hand_rr, hand_rr_valid, hand_rr_bounds, hand_rr_weights, audit_right_cam_right = read_wilor(
        args.wilor_right, n, "right", right, image_size, "right", args.hand_2d_weight > 0)
    (args.output_dir / "wilor_association_audit.json").write_text(
        json.dumps({"left_camera": audit_left_cam_left + audit_left_cam_right,
                    "right_camera": audit_right_cam_left + audit_right_cam_right},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    hand_l, hand_l_valid, hand_l_bounds = hand_ll, hand_ll_valid, hand_ll_bounds
    hand_r, hand_r_valid, hand_r_bounds = hand_rr, hand_rr_valid, hand_rr_bounds
    # Keep the complete WiLoR 21-point order.  The first point is the wrist;
    # each following finger contributes three internal joints and one surface
    # fingertip.  Bounds are a soft quality signal, never a deletion gate.
    hand_l_full, hand_r_full = hand_l.copy(), hand_r.copy()
    hand_l_full_valid, hand_r_full_valid = hand_l_valid.copy(), hand_r_valid.copy()
    hand_l_full_bounds, hand_r_full_bounds = hand_l_bounds.copy(), hand_r_bounds.copy()
    hand_l_mask, hand_r_mask = hand_l_valid, hand_r_valid
    hand_l_weight_np = hand_ll_weights.astype(np.float32)
    hand_r_weight_np = hand_rr_weights.astype(np.float32)

    # Geometry audit is deliberately separate from fitting masks.  It tests
    # whether the two raw-fisheye WiLoR views can form a valid stereo point;
    # no point is deleted or promoted to a hand-3D observation here.
    from pose_app.smplh_hand_observation import audit_cross_view_geometry
    if args.hand_2d_weight > 0:
        hand_geom_left, hand_geom_left_summary = audit_cross_view_geometry(
            hand_ll, hand_rl, cal.K0, cal.D0, cal.K1, cal.D1,
            cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm, image_size)
        hand_geom_right, hand_geom_right_summary = audit_cross_view_geometry(
            hand_lr, hand_rr, cal.K0, cal.D0, cal.K1, cal.D1,
            cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm, image_size)
    else:
        # Missing-by-configuration is not a failed triangulation measurement.
        hand_geom_left = hand_geom_right = []
        hand_geom_left_summary = hand_geom_right_summary = {
            "status": "unavailable_2d_not_consumed", "accepted": None,
            "acceptance_rate": None, "unavailable_frames": list(range(n))}
    (args.output_dir / "wilor_hand_geometry_audit.jsonl").write_text(
        "\n".join(json.dumps({"hand": "left", **r}, ensure_ascii=False) for r in hand_geom_left)
        + "\n" + "\n".join(json.dumps({"hand": "right", **r}, ensure_ascii=False) for r in hand_geom_right)
        + "\n", encoding="utf-8")
    (args.output_dir / "wilor_hand_geometry_summary.json").write_text(
        json.dumps({"left": hand_geom_left_summary, "right": hand_geom_right_summary,
                    "pixels_consumed": bool(args.hand_2d_weight > 0),
                    "status": "diagnostic_only" if args.hand_2d_weight > 0 else "unavailable_2d_not_consumed",
                    "used_for_fitting": False,
                    "gate": {"raw_bounds": True, "positive_depth": True,
                              "reprojection_error_px_each_view": 10.0}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if not (1 <= args.hand_pca_comps <= 45):
        raise ValueError("--hand-pca-comps must be in 1..45")
    profile_for_dim = {12: "pca12", 24: "pca24", 45: "full45"}.get(args.hand_pca_comps)
    if args.hand_pca_profile is not None and args.hand_pca_profile != profile_for_dim:
        raise ValueError("--hand-pca-profile must agree with --hand-pca-comps (12/24/45)")
    hand_pca_profile = args.hand_pca_profile or profile_for_dim or f"pca{args.hand_pca_comps}"
    # The SMPL-H file shipped with WiLoR lacks optional PCA fields. Load the
    # matching MANO assets explicitly; identity matrices are not a hand prior.
    with args.smplh_model.open("rb") as handle:
        model_data = pickle.load(handle, encoding="latin1")
    with args.mano_left.open("rb") as handle:
        mano_left_data = pickle.load(handle, encoding="latin1")
    with args.mano_right.open("rb") as handle:
        mano_right_data = pickle.load(handle, encoding="latin1")
    for name, data in (("left", mano_left_data), ("right", mano_right_data)):
        if np.asarray(data["hands_components"]).shape != (45, 45) or np.asarray(data["hands_mean"]).shape != (45,):
            raise ValueError(f"MANO {name} PCA asset must contain components (45,45) and mean (45,)")
    model_data["hands_componentsl"] = np.asarray(mano_left_data["hands_components"], np.float32)
    model_data["hands_componentsr"] = np.asarray(mano_right_data["hands_components"], np.float32)
    model_data["hands_meanl"] = np.asarray(mano_left_data["hands_mean"], np.float32)
    model_data["hands_meanr"] = np.asarray(mano_right_data["hands_mean"], np.float32)
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
    hand_pca_dim = int(args.hand_pca_comps)
    mano_l_components = torch.tensor(model_data["hands_componentsl"][:hand_pca_dim], dtype=torch.float32, device=device)
    mano_r_components = torch.tensor(model_data["hands_componentsr"][:hand_pca_dim], dtype=torch.float32, device=device)
    mano_l_mean = torch.tensor(model_data["hands_meanl"], dtype=torch.float32, device=device)
    mano_r_mean = torch.tensor(model_data["hands_meanr"], dtype=torch.float32, device=device)
    # Normalize coefficients by the empirical MANO coefficient scales. This
    # keeps the prior dimensionless and prevents the first PCA directions from
    # dominating solely because their raw units are larger.
    mano_l_scale = torch.linalg.vector_norm(mano_l_components, dim=1).clamp_min(1e-6)
    mano_r_scale = torch.linalg.vector_norm(mano_r_components, dim=1).clamp_min(1e-6)
    hand_parameter_frames = 1 if args.shared_hand_pose else n
    lhand = torch.nn.Parameter(torch.zeros(hand_parameter_frames, hand_pca_dim, device=device))
    rhand = torch.nn.Parameter(torch.zeros(hand_parameter_frames, hand_pca_dim, device=device))

    def decode_hand_pose(coeff, components, mean, scale):
        normalized = coeff / scale[None, :]
        pose = mean[None, :] + normalized @ components
        return pose.expand(n, -1) if args.shared_hand_pose else pose

    # Native local MANO information is independent of the pinhole-derived
    # keypoints. The detector box + body wrist associates the hypotheses.
    # Keep both views rather than averaging incompatible axis-angle vectors.
    mano_enabled = args.mano_pose_init or args.mano_pose_weight > 0
    mano_targets = mano_weights = mano_initial = None
    if mano_enabled:
        from pose_app.wilor_mano_prior import audit_assets, read_parameter_view, encode_pca
        with (ROOT / "third_party/WiLoR/mano_data/MANO_RIGHT.pkl").open("rb") as handle:
            canonical_right = pickle.load(handle, encoding="latin1")
        asset_audit = audit_assets(model_data, mano_left_data, mano_right_data, canonical_right)
        rotations0, weights0, audit0 = read_parameter_view(args.wilor_left, left, image_size)
        rotations1, weights1, audit1 = read_parameter_view(args.wilor_right, right, image_size)
        rot = np.stack([rotations0, rotations1], axis=1)  # N, views, side, 15,3,3
        w = np.stack([weights0, weights1], axis=1)
        mano_targets, mano_weights, mano_initial = {}, {}, {}
        reconstruction = {}
        disagreement = {}
        for side, h, components, mean, scale in (
            ("left", 0, mano_l_components, mano_l_mean, mano_l_scale),
            ("right", 1, mano_r_components, mano_r_mean, mano_r_scale)):
            if not (w[:, :, h] > 0).any():
                # Audit all rejected records before stopping, including legacy
                # files that have no native parameters. Never reconstruct pose
                # from the old 21 model-projected points.
                (args.output_dir / "wilor_mano_parameter_audit.json").write_text(
                    json.dumps({"left_camera": audit0, "right_camera": audit1,
                                "failure": f"no native MANO coverage for {side}"}, indent=2), encoding="utf-8")
                raise ValueError(f"no accepted native MANO parameters for {side}; regenerate WiLoR JSONL")
            coeff, errors = encode_pca(rot[:, :, h], components.detach().cpu().numpy(),
                                      mean.detach().cpu().numpy(), scale.detach().cpu().numpy())
            # Choose highest box-confidence view for initialization only.
            # The fitting prior retains BOTH independent camera hypotheses.
            best = np.argmax(w[:, :, h], axis=1)
            initial = coeff[np.arange(n), best]
            covered = w[:, :, h].max(axis=1) > 0
            initial[~covered] = 0  # explicit unavailable, not interpolation
            if args.shared_hand_pose:
                # This is only a linear PCA-space starting point. The actual
                # shared solution minimizes rotation loss over both views and
                # every accepted frame, not an average axis-angle pose.
                initial = ((coeff * w[:, :, h, None]).sum(axis=(0, 1)) /
                           w[:, :, h].sum())[None, :]
            mano_initial[side] = torch.tensor(initial, dtype=torch.float32, device=device)
            mano_targets[side] = torch.tensor(rot[:, :, h], dtype=torch.float32, device=device)
            mano_weights[side] = torch.tensor(w[:, :, h], dtype=torch.float32, device=device)
            valid_errors = np.degrees(errors[w[:, :, h] > 0])
            paired = (w[:, 0, h] > 0) & (w[:, 1, h] > 0)
            delta = rot[paired, 0, h].swapaxes(-1, -2) @ rot[paired, 1, h]
            view_angles = np.degrees(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude()) if paired.any() else np.array([])
            disagreement[side] = {"paired_frames": int(paired.sum()),
                                 "median_deg": float(np.median(view_angles)) if len(view_angles) else None,
                                 "p95_deg": float(np.percentile(view_angles,95)) if len(view_angles) else None}
            reconstruction[side] = {"covered_frames": int(covered.sum()),
                                    "unavailable_frames": np.flatnonzero(~covered).tolist(),
                                    "pca_rotation_error_median_deg": float(np.median(valid_errors)),
                                    "pca_rotation_error_p95_deg": float(np.percentile(valid_errors, 95))}
        (args.output_dir / "wilor_mano_parameter_audit.json").write_text(
            json.dumps({"assets": asset_audit, "left_camera": audit0, "right_camera": audit1,
                        "pca_reconstruction": reconstruction,
                        "cross_view_local_rotation_disagreement": disagreement,
                        "pose_weight": args.mano_pose_weight, "init": args.mano_pose_init,
                        "hand_2d_weight": args.hand_2d_weight,
                        "status": "model_pose_hypothesis_not_truth"}, indent=2), encoding="utf-8")

    def decode_body_pose(z):
        pose = vposer.decode(z)["pose_body"].reshape(n, 63)
        if pose.shape != (n, 63) or not bool(torch.isfinite(pose).all()):
            raise RuntimeError(f"VPoser decoded pose has invalid shape/values: {tuple(pose.shape)}")
        return pose

    # Build a model-space body basis before any optimization.  The basis is
    # defined by semantic COCO points, so it does not assume that SMPL-H's
    # local +X/+Y/+Z already match the calibrated camera axes.
    initialization = initialize_rigid_body(
        model=model, reg=reg, decode_body_pose=decode_body_pose, n=n,
        latent_dim=latent_dim, device=device, tri_m=tri_m, body_mask=body_mask,
        root=root, transl=transl)
    root_init_np = initialization.root_init_np
    root_init_source = initialization.root_init_source
    init_source = initialization.init_source
    body_bones = initialization.body_bones
    ref_bone_lengths = initialization.ref_bone_lengths
    root_anchor_target = initialization.root_anchor_target
    init_body_rms_mm = initialization.init_body_rms_mm
    init_report = {
        "body_rms_mm": init_body_rms_mm,
        "valid_root_basis_frames": int(np.sum(root_init_source == "shoulder_hip_rigid_basis")),
        "total_frames": n,
        "max_init_body_rms_mm": float(args.max_init_body_rms_mm),
        "root_init_source_counts": {name: int(np.sum(root_init_source == name)) for name in np.unique(root_init_source)},
    }
    (args.output_dir / "initialization_audit.json").write_text(
        json.dumps(init_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not np.isfinite(init_body_rms_mm) or init_body_rms_mm > args.max_init_body_rms_mm:
        raise RuntimeError(f"rigid initialization failed body RMS gate: {init_body_rms_mm:.1f} mm")
    if init_report["valid_root_basis_frames"] != n:
        raise RuntimeError("rigid initialization lacks accepted shoulder/hip basis on some frames")
    target = torch.tensor(tri_m, device=device)
    raw_left_2d = torch.tensor(finite_pixel_values(left[:, :, :2]), device=device)
    raw_right_2d = torch.tensor(finite_pixel_values(right[:, :, :2]), device=device)
    raw_left_conf = torch.tensor(np.clip(finite_pixel_values(left[:, :, 2]), 0.0, 1.0), device=device)
    raw_right_conf = torch.tensor(np.clip(finite_pixel_values(right[:, :, 2]), 0.0, 1.0), device=device)
    raw_left_valid = torch.isfinite(torch.tensor(left[:, :, :2], device=device)).all(dim=-1) & (raw_left_conf > 0)
    raw_right_valid = torch.isfinite(torch.tensor(right[:, :, :2], device=device)).all(dim=-1) & (raw_right_conf > 0)
    obs_ll = torch.tensor(finite_pixel_values(hand_ll), device=device)
    obs_lr = torch.tensor(finite_pixel_values(hand_lr), device=device)
    obs_rl = torch.tensor(finite_pixel_values(hand_rl), device=device)
    obs_rr = torch.tensor(finite_pixel_values(hand_rr), device=device)
    mask_body = torch.tensor(body_mask, device=device)
    mask_ll = torch.tensor(hand_ll_valid, device=device); mask_lr = torch.tensor(hand_lr_valid, device=device)
    mask_rl = torch.tensor(hand_rl_valid, device=device); mask_rr = torch.tensor(hand_rr_valid, device=device)
    weight_ll = torch.tensor(hand_ll_weights, device=device)
    weight_lr = torch.tensor(hand_lr_weights, device=device)
    weight_rl = torch.tensor(hand_rl_weights, device=device)
    weight_rr = torch.tensor(hand_rr_weights, device=device)
    hand_order_weights_np = np.asarray(
        [1.00, 0.95, 0.82, 0.68, 0.48,
         0.95, 0.82, 0.68, 0.48, 0.95, 0.82, 0.68, 0.48,
         0.95, 0.82, 0.68, 0.48, 0.95, 0.82, 0.68, 0.48], np.float32)
    order_w = torch.tensor(hand_order_weights_np, device=device)[None, :]
    weight_ll = weight_ll * order_w; weight_lr = weight_lr * order_w
    weight_rl = weight_rl * order_w; weight_rr = weight_rr * order_w
    contact_enabled = args.contact_labels is not None
    contact = load_surface_targets(args, n, device)
    R01 = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=device)
    T01 = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=device)
    K0, D0 = cal.K0, cal.D0
    K1, D1 = cal.K1, cal.D1

    with torch.no_grad():
        _ = decode_body_pose(latent)
    history = []
    from pose_app.smplh_hand_observation import HAND_JOINTS, validate_smplh, hand21
    validate_smplh(model)
    # Named SMPL-H internal-joint maps; hand21() emits the verified WiLoR
    # OpenPose order and appends five surface fingertip vertices.
    hand_map_l = HAND_JOINTS["left"]
    hand_map_r = HAND_JOINTS["right"]
    assert hand_map_l == [20, 34, 35, 36, 22, 23, 24, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    assert hand_map_r == [21, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45]
    # WiLoR OpenPose order -> SMPL-H joints verified above; tips via hand21 vertices.
    stage_names, stage_train, stage_steps = build_stage_schedule(args, contact_enabled)
    param_map = {"beta": beta, "root": root, "transl": transl,
                 "latent": latent, "lhand": lhand, "rhand": rhand}
    stage_history = []
    np.savez_compressed(args.output_dir / "cold_initialization.npz",
                        betas=beta.detach().cpu().numpy(), global_orient=root.detach().cpu().numpy(),
                        transl=transl.detach().cpu().numpy(), vposer_latent=latent.detach().cpu().numpy(),
                        pair_id=np.asarray(ids, dtype=np.int64))
    global_step = 0
    for stage_index, (stage_name, train_names, n_stage) in enumerate(zip(stage_names, stage_train, stage_steps)):
        # Do not let MANO initialization alter A/B/C body or shared-beta fits.
        if stage_index == 3 and args.mano_pose_init:
            with torch.no_grad():
                lhand.copy_(mano_initial["left"])
                rhand.copy_(mano_initial["right"])
        set_trainable_parameters(param_map, train_names)
        lr_scale = {0: 1.0, 1: 0.10, 2: 0.25, 3: 0.50, 4: 0.25, 5: 0.10}[stage_index]
        stage_params = [param_map[k] for k in train_names]
        optim = torch.optim.Adam(stage_params, lr=args.lr * lr_scale)
        for local_step in range(n_stage):
            step = global_step
            global_step += 1
            optim.zero_grad(set_to_none=True)
            body_pose = decode_body_pose(latent)
            lhand_pose = decode_hand_pose(lhand, mano_l_components, mano_l_mean, mano_l_scale)
            rhand_pose = decode_hand_pose(rhand, mano_r_components, mano_r_mean, mano_r_scale)
            out = model(betas=beta.expand(n, -1), global_orient=root,
                        body_pose=body_pose, left_hand_pose=lhand_pose,
                        right_hand_pose=rhand_pose, transl=transl, return_verts=True)
            coco = regress_coco17_torch(out.vertices, reg)
            body_res = (coco - target).pow(2).sum(-1)
            body_w = torch.tensor(np.asarray(quality, np.float32), device=device).clamp_min(0.0)
            body_loss = (body_res * body_w)[mask_body].sum() / body_w[mask_body].sum().clamp_min(1e-6) if mask_body.any() else body_res.mean() * 0.0
            model_cam1 = (R01 @ coco.transpose(1, 2)).transpose(1, 2) + T01
            proj_cam0 = fisheye_project_torch(coco, K0, D0)
            proj_cam1 = fisheye_project_torch(model_cam1, K1, D1)
            body_2d_loss, body_2d_px = body_reprojection_loss(
                args=args, stage_index=stage_index, body_loss=body_loss,
                proj_cam0=proj_cam0, proj_cam1=proj_cam1,
                raw_left_2d=raw_left_2d, raw_right_2d=raw_right_2d,
                raw_left_conf=raw_left_conf, raw_right_conf=raw_right_conf,
                raw_left_valid=raw_left_valid, raw_right_valid=raw_right_valid)
            joints = out.joints
            current_bone_lengths = torch.stack([
                torch.linalg.vector_norm(joints[:, a] - joints[:, b], dim=-1)
                for a, b in body_bones], dim=1)
            ref_bones_t = torch.tensor(ref_bone_lengths, dtype=torch.float32, device=device)[None, :]
            bone_loss = (current_bone_lengths - ref_bones_t).pow(2).mean()
            root_anchor = (root - root_anchor_target).pow(2).mean()
            jl = hand21(joints, out.vertices, "left")
            jr = hand21(joints, out.vertices, "right")
            left_in_cam0 = fisheye_project_torch(jl, K0, D0)
            right_in_cam0 = fisheye_project_torch(jr, K0, D0)
            left_in_cam1 = fisheye_project_torch((R01 @ jl.transpose(1, 2)).transpose(1, 2) + T01, K1, D1)
            right_in_cam1 = fisheye_project_torch((R01 @ jr.transpose(1, 2)).transpose(1, 2) + T01, K1, D1)
            # The two axes are camera identity and anatomical hand side.
            hand_ll_res = (left_in_cam0 - obs_ll).pow(2).sum(-1)
            hand_rl_res = (left_in_cam1 - obs_rl).pow(2).sum(-1)
            hand_lr_res = (right_in_cam0 - obs_lr).pow(2).sum(-1)
            hand_rr_res = (right_in_cam1 - obs_rr).pow(2).sum(-1)
            hand_l_loss = weighted_hand_residual(hand_ll_res, weight_ll, mask_ll) + weighted_hand_residual(hand_rl_res, weight_rl, mask_rl)
            hand_r_loss = weighted_hand_residual(hand_lr_res, weight_lr, mask_lr) + weighted_hand_residual(hand_rr_res, weight_rr, mask_rr)
            pose_reg = (args.vposer_prior_weight * latent.pow(2).mean()
                        + args.hand_pca_prior_weight * (lhand.pow(2).mean() + rhand.pow(2).mean()))
            hand_temporal = hand_temporal_loss(
                args=args, n=n, lhand=lhand, rhand=rhand,
                mask_ll=mask_ll, mask_rl=mask_rl, mask_lr=mask_lr, mask_rr=mask_rr,
                mano_enabled=mano_enabled, mano_weights=mano_weights)
            body_temporal = body_temporal_loss(
                args=args, n=n, stage_index=stage_index, body_loss=body_loss,
                coco=coco, mask_body=mask_body)
            if stage_index in (3, 4):
                body_term = body_loss.detach() * 0.0
            elif stage_index == 5:
                body_term = body_loss.detach() * 0.0
            else:
                body_term = body_loss
            # Pixel residuals are numerically much larger than metre-scale
            # body residuals. Keep hand fitting auxiliary and prevent it from
            # moving the body/root to explain WiLoR's model-derived pixels.
            hand_term = args.hand_2d_weight * (hand_l_loss + hand_r_loss) if stage_index >= 3 else (hand_l_loss + hand_r_loss).detach() * 0.0
            mano_pose_loss = (lhand_pose.sum() + rhand_pose.sum()) * 0.0
            if mano_enabled and stage_index >= 3:
                from pose_app.wilor_mano_prior import rotation_pose_loss
                mano_pose_loss = 0.5 * (
                    rotation_pose_loss(lhand_pose, mano_targets["left"], mano_weights["left"]) +
                    rotation_pose_loss(rhand_pose, mano_targets["right"], mano_weights["right"]))
            structure_term = (args.bone_weight * bone_loss if stage_index in (0, 2)
                              else bone_loss.detach() * 0.0)
            temporal_term = (args.hand_temporal_weight * hand_temporal
                             if stage_index >= 3 else hand_temporal.detach() * 0.0)
            root_term = (args.root_anchor_weight * root_anchor if stage_index == 2
                         else root_anchor.detach() * 0.0)
            contact_hand_loss, contact_foot_loss, global_hand_loss = surface_contact_losses(
                left_in_cam0=left_in_cam0, contact_enabled=contact_enabled,
                stage_index=stage_index, vertices=out.vertices,
                contact=contact)
            reproj_term = (args.body_reprojection_weight * body_2d_loss
                           if stage_index in (0, 1, 2) else body_2d_loss.detach() * 0.0)
            contact_total = (args.surface_hand_contact_weight * contact_hand_loss
                             + args.surface_foot_contact_weight * contact_foot_loss)
            if contact.global_hand_offset is not None and stage_index >= 4:
                contact_total = contact_total + args.global_hand_handle_weight * global_hand_loss
            contact_anchor = (args.root_anchor_weight * root_anchor
                              if contact_enabled and stage_index >= 4 else root_anchor.detach() * 0.0)
            wrist_loss = (wrist_position_loss(out.joints, wrist_target)
                          if wrist_target is not None else body_loss.detach() * 0.0)
            wrist_term = (args.wrist_reference_weight * wrist_loss
                          if stage_index in (0, 1, 2) else wrist_loss.detach() * 0.0)
            loss = body_term + reproj_term + structure_term + root_term + contact_anchor + temporal_term + \
                   args.body_temporal_weight * body_temporal + hand_term + args.mano_pose_weight * mano_pose_loss + pose_reg + 1e-3 * beta.pow(2).mean() + \
                   (contact_total if stage_index >= 4 else contact_total.detach() * 0.0) + wrist_term
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at {stage_name}:{local_step}")
            loss.backward()
            require_finite_tensors({name: param_map[name].grad for name in train_names},
                                   f"gradient at {stage_name}:{local_step}")
            optim.step()
            require_finite_tensors({name: param_map[name] for name in train_names},
                                   f"parameters after {stage_name}:{local_step}")
            if step % 10 == 0 or step == sum(stage_steps) - 1:
                row = {"stage": stage_name, "stage_step": local_step, "step": step, "loss": float(loss.detach()),
                            "body_m": float(torch.sqrt(body_loss.detach())),
                            "body_2d_px": float(body_2d_px.detach()),
                            "hand_l_px": float(torch.sqrt(hand_l_loss.detach())) if mask_ll.any() or mask_rl.any() else None,
                            "hand_r_px": float(torch.sqrt(hand_r_loss.detach())) if mask_lr.any() or mask_rr.any() else None,
                            "hand_l_points": int(mask_ll.sum() + mask_rl.sum()), "hand_r_points": int(mask_lr.sum() + mask_rr.sum()),
                            "contact_hand": float(contact_hand_loss.detach()),
                            "mano_pose_chordal": float(mano_pose_loss.detach()),
                            "contact_foot": float(contact_foot_loss.detach()),
                            "contact": float(contact_total.detach()),
                            "global_hand_handle": float(global_hand_loss.detach()),
                            "contact_active": bool(contact_enabled and stage_index >= 4)}
                row["bone_m"] = float(torch.sqrt(bone_loss.detach()))
                row["root_anchor"] = float(torch.sqrt(root_anchor.detach()))
                row["hand_temporal"] = float(torch.sqrt(hand_temporal.detach()))
                row["body_temporal"] = float(torch.sqrt(body_temporal.detach()))
                row["wrist_reference_rms_mm"] = float(torch.sqrt(wrist_loss.detach()) * 1000) if wrist_target is not None else None
                history.append(row)
                print(json.dumps(row), flush=True)
        stage_history.append({"stage": stage_name, "steps": n_stage, "trainable": train_names,
                              "hand_3d_term": False,
                              "mano_local_pose_term": bool(args.mano_pose_weight > 0 and stage_index >= 3),
                              "learning_rate": float(args.lr * lr_scale),
                              "hand_priority": "wrist_to_distal_fixed_weights" if stage_index >= 3 else "inactive",
                              "contact_term": bool(contact_enabled and stage_index >= 4),
                              "contact_weights_independent": True})

    with torch.no_grad():
        body_pose = decode_body_pose(latent)
        lhand_pose = decode_hand_pose(lhand, mano_l_components, mano_l_mean, mano_l_scale)
        rhand_pose = decode_hand_pose(rhand, mano_r_components, mano_r_mean, mano_r_scale)
        final = model(betas=beta.expand(n, -1), global_orient=root,
                      body_pose=body_pose, left_hand_pose=lhand_pose,
                      right_hand_pose=rhand_pose, transl=transl, return_verts=True)
        final_hand_l = hand21(final.joints, final.vertices, "left")
        final_hand_r = hand21(final.joints, final.vertices, "right")
        final_coco = regress_coco17_torch(final.vertices, reg)
        require_finite_tensors({**param_map, "vertices": final.vertices,
                               "joints": final.joints, "predicted_coco": final_coco,
                               "left_hand_pose": lhand_pose, "right_hand_pose": rhand_pose},
                              "final output")
        contact_diag = {}
        if mano_enabled:
            for side in ("left", "right"):
                contact_diag[f"mano_{side}_target_rotations"] = mano_targets[side].cpu().numpy()
                contact_diag[f"mano_{side}_view_weights"] = mano_weights[side].cpu().numpy()
                contact_diag[f"mano_{side}_initial_pca"] = mano_initial[side].cpu().numpy()
        if contact_enabled:
            vg = torch.einsum("nij,nvj->nvi", contact.rotation, final.vertices) + contact.translation[:, None, :]
            for side, j in (("left", 0), ("right", 1)):
                a, b = contact.handle[:, j, 0, :], contact.handle[:, j, 1, :]
                contact_diag[f"{side}_palm_ground_m"] = vg[:, contact.palm_indices[side], :].cpu().numpy()
                contact_diag[f"{side}_hand_capsule_residual_m"] = surface_contact.capsule_surface_residual(
                    vg[:, contact.palm_indices[side], :], a[:, None, :], b[:, None, :], 0.016).cpu().numpy()
            contact_diag["hand_contact_weight"] = contact.hand_weight.cpu().numpy()
    np.savez_compressed(args.output_dir / "result.npz",
                        pair_id=np.asarray(ids, dtype=np.int64),
                        vertices=final.vertices.cpu().numpy(), faces=np.asarray(model.faces),
                        predicted_coco=final_coco.cpu().numpy(),
                        betas=beta.detach().cpu().numpy(), global_orient=root.detach().cpu().numpy(),
                        transl=transl.detach().cpu().numpy(),
                        body_pose=body_pose.detach().cpu().numpy(),
                        vposer_latent=latent.detach().cpu().numpy(),
                        left_hand_pose=lhand_pose.detach().cpu().numpy(), right_hand_pose=rhand_pose.detach().cpu().numpy(),
                        left_hand_pca=lhand.detach().expand(n, -1).cpu().numpy(), right_hand_pca=rhand.detach().expand(n, -1).cpu().numpy(),
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
                        root_init=root_init_np,
                        root_init_source=root_init_source,
                        initialization_body_rms_mm=np.asarray(init_body_rms_mm, dtype=np.float32),
                        wilor_left_weight=hand_l_weight_np, wilor_right_weight=hand_r_weight_np,
                        smplh_joints=final.joints.detach().cpu().numpy(),
                        hand_points_left=final_hand_l.detach().cpu().numpy(),
                        hand_points_right=final_hand_r.detach().cpu().numpy(), **contact_diag)
    (args.output_dir / "fit_summary.json").write_text(json.dumps({
        "status": "engineering_candidate", "frames": n, "steps": int(sum(stage_steps)),
        "wrist_reference": wrist_audit,
        "initial_beta_source": "zeros_reoptimized_in_B_and_C",
        "model": str(args.smplh_model.resolve()), "vertices": 6890,
        "hand_observation_source": "WiLoR_native_MANO_local_rotations" if mano_enabled and args.hand_2d_weight == 0 else "WiLoR_model_projected_MANO_joints",
        "native_mano_prior": {
            "enabled": bool(mano_enabled), "weight": args.mano_pose_weight,
            "initialization": args.mano_pose_init, "hand_2d_weight": args.hand_2d_weight,
            "audit": "wilor_mano_parameter_audit.json" if mano_enabled else None,
            "loss": "SO3_chordal_1_minus_cos_angle",
            "trainable": "shared_MANO_PCA_only" if args.shared_hand_pose else "existing_MANO_PCA_only",
            "global_orient_transl_betas_used": False,
            "stage": "D1_D2_D3_only"},
        "hand_observation_points_retained": 21,
        "hand_fit_subset_points": 21 if args.hand_2d_weight > 0 else 0,
        "hand_bounds_are_diagnostic_only": True,
        "wilor_left_valid_points": int(hand_ll_valid.sum() + hand_rl_valid.sum()),
        "wilor_right_valid_points": int(hand_lr_valid.sum() + hand_rr_valid.sum()), "history": history,
        "association_audit": "wilor_association_audit.json",
        "wilor_observation_views": {"left_camera": ["left_hand", "right_hand"],
                                     "right_camera": ["left_hand", "right_hand"]},
        "stage_schedule": stage_history,
        "hand_confidence_source": "candidate_box_confidence_weighted_or_missing_neutral_weight",
        "vposer_checkpoint": str(vp_ckpt), "vposer_latent_dim": latent_dim,
        "mano_left": str(args.mano_left.resolve()), "mano_right": str(args.mano_right.resolve()),
        "hand_pose_parameterization": "MANO_PCA_decode_to_SMPLH_45D_axis_angle",
        "hand_pca_components": hand_pca_dim,
        "shared_hand_pose": bool(args.shared_hand_pose),
        "hand_pose_parameter_frames": hand_parameter_frames,
        "shared_hand_pose_scope": "local_finger_rotations_only_not_walker_relative_SE3" if args.shared_hand_pose else None,
        "hand_pca_profile": hand_pca_profile,
        "hand_pca_ablation_supported": ["pca12", "pca24", "full45"],
        "hand_pca_prior_weight": float(args.hand_pca_prior_weight),
        "hand_temporal_weight": float(args.hand_temporal_weight),
        "body_temporal_weight": float(args.body_temporal_weight),
        "body_temporal": {"mode": "model_coco_relative_pelvis_huber" if args.body_temporal_weight > 0 else "disabled",
                           "huber_delta_m": 0.03, "cross_gap": False,
                           "used_for_hand_3d": False},
        "body_reprojection": {"weight": float(args.body_reprojection_weight),
                               "scale_px": float(args.body_reprojection_scale_px),
                               "mode": "dual_fisheye_raw_pmpose_huber" if args.body_reprojection_weight > 0 else "disabled",
                               "camera_parameters_optimized": False,
                               "stages": ["A_body_vposer", "B_shared_beta", "C_body_vposer_refine"],
                               "uses_triangulated_3d_term": True},
        "hand_geometry_audit": {"path": "wilor_hand_geometry_audit.jsonl",
                                "summary": "wilor_hand_geometry_summary.json",
                                "used_for_fitting": False},
        "hand_priority_weights": hand_order_weights_np.tolist(),
        "contact_enabled": bool(contact_enabled),
        "contact_weighting": {"hand_surface": float(args.surface_hand_contact_weight),
                               "foot_surface": float(args.surface_foot_contact_weight),
                               "global_hand_handle": float(args.global_hand_handle_weight)},
        "contact_schedule": ["D1 hand observation",
                             "D2 surface hand+foot contact" if contact_enabled else "D2 hand-only refinement",
                             "D3 hand+foot contact refinement" if contact_enabled else "D3 skipped: no contact input"],
        "hand_3d_observation_used": False,
        "body_accepted_points": int(body_mask.sum()),
        "initialization": {
            "root_source": "SMPLH_template_to_triangulated_shoulder_hip_rigid_basis",
            "translation_source": "same_rotated_template_hip_midpoint",
            "body_rms_mm": init_body_rms_mm,
            "valid_root_basis_frames": int(np.sum(root_init_source == "shoulder_hip_rigid_basis")),
            "bone_pairs_smplh": [list(x) for x in body_bones],
            "bone_weight": float(args.bone_weight),
            "root_anchor_weight": float(args.root_anchor_weight),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0
