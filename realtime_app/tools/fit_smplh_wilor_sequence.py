#!/usr/bin/env python3
"""Fit PCA/VPoser SMPL-H from body stereo and optional native WiLoR pose.

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


def read_wilor(path: Path, n: int, side: str, body_points: np.ndarray,
               image_size: tuple[int, int], camera: str, consume_pixels: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Use the single audited WiLoR association implementation.

    The shared reader validates frame/image identity, duplicate frames and
    same-camera wrist association.  ``valid`` remains finite-only so rejected
    or out-of-bounds points are preserved for diagnostics and soft weighting.
    """
    from pose_app.smplh_hand_observation import read_view

    if not consume_pixels:
        # Native-MANO-only runs must not require projected points at all.
        # Original records remain in the input JSONL, not replaced with zero
        # observations. The unavailable arrays preserve downstream shapes.
        return (np.full((n,21,2), np.nan, np.float32), np.zeros((n,21), bool),
                np.zeros((n,21), bool), np.zeros((n,21), np.float32),
                [{"source": str(path), "camera": camera, "hand": side,
                  "reason": "hand_2d_disabled_not_consumed", "selected": False}])

    points, weights, audit = read_view(path, body_points, image_size, camera)
    hand_index = 0 if side == "left" else 1
    obs = points[:, hand_index]
    valid = np.isfinite(obs).all(axis=-1)
    width, height = image_size
    bounds_ok = (valid & (obs[:, :, 0] >= 0) & (obs[:, :, 0] < width)
                 & (obs[:, :, 1] >= 0) & (obs[:, :, 1] < height))
    return obs, valid, bounds_ok, weights[:, hand_index], audit


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
    ap.add_argument("--root-anchor-weight", type=float, default=0.01,
                    help="Stage C/D3 penalty for drifting from the triangulation-initialized root")
    ap.add_argument("--max-init-body-rms-mm", type=float, default=300.0,
                    help="stop before optimization when rigid initialization is inconsistent")
    ap.add_argument("--hand-pca-comps", type=int, default=12,
                    help="number of real MANO PCA hand-pose coefficients per hand")
    ap.add_argument("--hand-pca-profile", choices=("pca12", "pca24", "full45"), default=None,
                    help="explicit ablation label; must agree with --hand-pca-comps")
    ap.add_argument("--hand-pca-prior-weight", type=float, default=1e-3,
                    help="quadratic prior on normalized MANO PCA coefficients")
    ap.add_argument("--hand-temporal-weight", type=float, default=2e-2,
                    help="second-order temporal prior on MANO PCA coefficients")
    ap.add_argument("--mano-pose-weight", type=float, default=0.0,
                    help="native WiLoR local rotation soft prior (SO3 chordal); enabled only in D1/D2/D3")
    ap.add_argument("--mano-pose-init", action="store_true",
                    help="initialize hand PCA at the start of D1 from accepted native MANO poses")
    ap.add_argument("--shared-hand-pose", action="store_true",
                    help="one PCA vector per anatomical hand shared by all frames; local articulation only")
    ap.add_argument("--hand-2d-weight", type=float, default=1e-7,
                    help="WiLoR model-derived 2D auxiliary weight; set 0 for native-MANO-only hand information")
    ap.add_argument("--body-temporal-weight", type=float, default=0.0,
                    help="model-COCO relative-to-pelvis second-order prior; 0 preserves the audited baseline")
    ap.add_argument("--body-reprojection-weight", type=float, default=0.0,
                    help="dual-fisheye body 2-D reprojection term; 0 preserves the audited baseline")
    ap.add_argument("--body-reprojection-scale-px", type=float, default=100.0,
                    help="pixel scale used to make the body reprojection Huber term dimensionless")
    ap.add_argument("--mano-left", type=Path, default=ROOT / "third_party/WiLoR/mano_data/models/MANO_LEFT.pkl")
    ap.add_argument("--mano-right", type=Path, default=ROOT / "third_party/WiLoR/mano_data/models/MANO_RIGHT.pkl")
    ap.add_argument("--bone-weight", type=float, default=0.0,
                    help="optional beta-zero bone-length prior; default off because it biases shared beta")
    ap.add_argument("--contact-labels", type=Path, default=None)
    ap.add_argument("--scene-transforms", type=Path, default=None)
    ap.add_argument("--contact-vertex-sets", type=Path, default=None)
    ap.add_argument("--walker-topology", type=Path, default=None,
                    help="validated walker topology; used to check handle semantics")
    ap.add_argument("--surface-hand-contact-weight", type=float, default=0.0)
    ap.add_argument("--surface-foot-contact-weight", type=float, default=0.0)
    ap.add_argument("--global-hand-handle-pose", type=Path, default=None)
    ap.add_argument("--global-hand-handle-weight", type=float, default=0.0)
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
    contact_args = (args.contact_labels, args.scene_transforms,
                    args.contact_vertex_sets, args.walker_topology)
    if any(v is not None for v in contact_args) and not all(v is not None for v in contact_args):
        raise ValueError("contact mode requires contact-labels, scene-transforms, contact-vertex-sets and walker-topology together")
    if all(v is not None for v in contact_args) and (args.surface_hand_contact_weight + args.surface_foot_contact_weight) <= 0.0:
        raise ValueError("contact inputs supplied but both surface contact weights are zero")
    if args.global_hand_handle_pose is not None and not all(v is not None for v in contact_args):
        raise ValueError("global hand-handle prior requires contact inputs")

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
    def make_body_basis(points: np.ndarray) -> np.ndarray | None:
        p = np.asarray(points, dtype=np.float64)
        needed = (5, 6, 11, 12)
        if not np.isfinite(p[list(needed)]).all():
            return None
        left = p[5] - p[6]
        up = 0.5 * (p[5] + p[6]) - 0.5 * (p[11] + p[12])
        nl = np.linalg.norm(left)
        if nl < 1e-8:
            return None
        left = left / nl
        up = up - left * float(left @ up)
        nu = np.linalg.norm(up)
        if nu < 1e-8:
            return None
        up = up / nu
        forward = np.cross(left, up)
        nf = np.linalg.norm(forward)
        if nf < 1e-8:
            return None
        forward = forward / nf
        basis = np.column_stack((left, up, forward))
        if np.linalg.det(basis) < 0:
            basis[:, 2] *= -1.0
        return basis

    # Decode the VPoser mean and use it for the template basis.  A zero axis
    # angle is not treated as an observation-based body orientation.
    with torch.no_grad():
        template_latent = torch.zeros(n, latent_dim, device=device)
        template_body_pose = decode_body_pose(template_latent)
        template = model(
            betas=torch.zeros(n, 10, device=device),
            global_orient=torch.zeros(n, 3, device=device),
            body_pose=template_body_pose,
            left_hand_pose=torch.zeros(n, 45, device=device),
            right_hand_pose=torch.zeros(n, 45, device=device),
            transl=torch.zeros(n, 3, device=device), return_verts=True)
        template_coco = regress_coco17_torch(template.vertices, reg).detach().cpu().numpy()
        template_joints = template.joints.detach().cpu().numpy()
    model_basis = make_body_basis(template_coco[0])
    if model_basis is None:
        raise RuntimeError("cannot construct a valid SMPL-H template shoulder/hip basis")
    root_init_np = np.zeros((n, 3), dtype=np.float32)
    root_init_source = np.full(n, "basis_unavailable", dtype="U32")
    from scipy.spatial.transform import Rotation
    for i in range(n):
        observed_basis = make_body_basis(tri_m[i]) if body_mask[i, [5, 6, 11, 12]].all() else None
        if observed_basis is None:
            root_init_np[i] = np.zeros(3, dtype=np.float32)
            continue
        global_rotation = observed_basis @ model_basis.T
        if not np.isfinite(global_rotation).all() or np.linalg.det(global_rotation) <= 0:
            continue
        root_init_np[i] = Rotation.from_matrix(global_rotation).as_rotvec().astype(np.float32)
        root_init_source[i] = "shoulder_hip_rigid_basis"

    # Translation is computed from the same rotated template used for root
    # initialization.  This prevents a zero-rotation translation from being
    # paired with a later nonzero global orientation.
    with torch.no_grad():
        root_init_t = torch.tensor(root_init_np, dtype=torch.float32, device=device)
        rotated_template = model(
            betas=torch.zeros(n, 10, device=device), global_orient=root_init_t,
            body_pose=template_body_pose,
            left_hand_pose=torch.zeros(n, 45, device=device),
            right_hand_pose=torch.zeros(n, 45, device=device),
            transl=torch.zeros(n, 3, device=device), return_verts=True)
        rotated_template_coco = regress_coco17_torch(rotated_template.vertices, reg)
        rotated_template_np = rotated_template_coco.detach().cpu().numpy()
    init_source = []
    t0 = np.zeros((n, 3), np.float32)
    for i in range(n):
        hl = tri_m[i, 11] if np.isfinite(tri_m[i, 11]).all() and body_mask[i, 11] else None
        hr = tri_m[i, 12] if np.isfinite(tri_m[i, 12]).all() and body_mask[i, 12] else None
        if hl is not None and hr is not None:
            t0[i] = (hl + hr) / 2 - (rotated_template_np[i, 11] + rotated_template_np[i, 12]) / 2
            init_source.append("rotated_template_hips_midpoint")
        elif hl is not None:
            t0[i] = hl - rotated_template_np[i, 11]; init_source.append("rotated_template_left_hip")
        elif hr is not None:
            t0[i] = hr - rotated_template_np[i, 12]; init_source.append("rotated_template_right_hip")
        else:
            t0[i] = np.zeros(3, np.float32); init_source.append("translation_init_unavailable")
    with torch.no_grad():
        root[:] = root_init_t
        transl[:] = torch.tensor(t0, device=device)

    # SMPL-H supplies the reference lengths.  The triangulated skeleton is
    # not used as a bone-length truth source.
    body_bones = ((16, 17), (1, 2), (16, 18), (18, 20), (17, 19), (19, 21),
                  (1, 4), (4, 7), (2, 5), (5, 8))
    ref_bone_lengths = np.asarray([
        np.linalg.norm(template_joints[0, a] - template_joints[0, b])
        for a, b in body_bones], dtype=np.float32)
    root_anchor_target = root_init_t.detach().clone()
    init_coco_after_translation = rotated_template_np + t0[:, None, :]
    init_res = init_coco_after_translation - tri_m
    init_valid = body_mask & np.isfinite(init_res).all(axis=-1)
    init_body_rms_mm = float(np.sqrt(np.mean(np.sum(init_res[init_valid] ** 2, axis=-1))) * 1000.0) if init_valid.any() else float("nan")
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
    raw_left_2d = torch.tensor(np.nan_to_num(left[:, :, :2], nan=0.0), device=device)
    raw_right_2d = torch.tensor(np.nan_to_num(right[:, :, :2], nan=0.0), device=device)
    raw_left_conf = torch.tensor(np.clip(np.nan_to_num(left[:, :, 2], nan=0.0), 0.0, 1.0), device=device)
    raw_right_conf = torch.tensor(np.clip(np.nan_to_num(right[:, :, 2], nan=0.0), 0.0, 1.0), device=device)
    raw_left_valid = torch.isfinite(torch.tensor(left[:, :, :2], device=device)).all(dim=-1) & (raw_left_conf > 0)
    raw_right_valid = torch.isfinite(torch.tensor(right[:, :, :2], device=device)).all(dim=-1) & (raw_right_conf > 0)
    obs_ll = torch.tensor(np.nan_to_num(hand_ll, nan=0.0), device=device)
    obs_lr = torch.tensor(np.nan_to_num(hand_lr, nan=0.0), device=device)
    obs_rl = torch.tensor(np.nan_to_num(hand_rl, nan=0.0), device=device)
    obs_rr = torch.tensor(np.nan_to_num(hand_rr, nan=0.0), device=device)
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
    contact_hand_weight = contact_foot_weight = contact_handle = contact_R = contact_T = palm_idx = sole_idx = None
    global_hand_offset = None
    global_hand_surface = global_camera_R = global_camera_t = None
    if contact_enabled:
        labels = np.load(args.contact_labels, allow_pickle=True)
        scene = np.load(args.scene_transforms, allow_pickle=True)
        if int(labels["handle_ends_ground_m"].shape[0]) != n:
            raise ValueError("contact labels frame count does not match raw input")
        if tuple(labels["hand_contact_weight"].shape) != (n, 2):
            raise ValueError("hand_contact_weight must have shape (frames,2)")
        if tuple(labels["foot_contact_weight"].shape) != (n, 2):
            raise ValueError("foot_contact_weight must have shape (frames,2)")
        if tuple(labels["handle_ends_ground_m"].shape) != (n, 2, 2, 3):
            raise ValueError("handle_ends_ground_m must have shape (frames,2,2,3)")
        if tuple(scene["rotation_ground_from_left"].shape) != (n, 3, 3):
            raise ValueError("rotation_ground_from_left must have shape (frames,3,3)")
        if tuple(scene["translation_ground_from_left_mm"].shape) != (n, 3):
            raise ValueError("translation_ground_from_left_mm must have shape (frames,3)")
        if not np.isfinite(labels["hand_contact_weight"]).all() or np.any(labels["hand_contact_weight"] < 0):
            raise ValueError("hand_contact_weight must be finite and non-negative")
        if not np.isfinite(labels["foot_contact_weight"]).all() or np.any(labels["foot_contact_weight"] < 0):
            raise ValueError("foot_contact_weight must be finite and non-negative")
        if not np.isfinite(labels["handle_ends_ground_m"]).all():
            raise ValueError("handle_ends_ground_m contains non-finite values")
        contact_handle = torch.tensor(labels["handle_ends_ground_m"], dtype=torch.float32, device=device)
        contact_hand_weight = torch.tensor(labels["hand_contact_weight"], dtype=torch.float32, device=device)
        contact_foot_weight = torch.tensor(labels["foot_contact_weight"], dtype=torch.float32, device=device)
        contact_R = torch.tensor(scene["rotation_ground_from_left"], dtype=torch.float32, device=device)
        contact_T = torch.tensor(scene["translation_ground_from_left_mm"] / 1000.0, dtype=torch.float32, device=device)
        sets = json.loads(args.contact_vertex_sets.read_text(encoding="utf-8"))["sets"]
        topology = json.loads(args.walker_topology.read_text(encoding="utf-8"))
        handles = topology.get("handle_segments", {})
        if set(handles) != {"left", "right"} or any(len(v) != 2 for v in handles.values()):
            raise ValueError("walker topology must expose exactly two endpoints for left/right handles")
        palm_idx = {s: torch.tensor(sum(sets[f"{s}_palm_surface_candidate"].values(), []), dtype=torch.long, device=device)
                    for s in ("left", "right")}
        sole_idx = {s: torch.tensor(sum(sets[f"{s}_sole_surface_candidate"].values(), []), dtype=torch.long, device=device)
                    for s in ("left", "right")}
        if any(int(v.numel()) == 0 for v in palm_idx.values()):
            raise ValueError("contact vertex sets contain no palm vertices")
        if any(int(v.numel()) == 0 for v in sole_idx.values()):
            raise ValueError("contact vertex sets contain no sole vertices")
        if any(int(v.min()) < 0 or int(v.max()) >= 6890 for v in palm_idx.values()):
            raise ValueError("contact vertex index is outside the 6890-vertex SMPL-H topology")
        if float(np.asarray(labels["hand_contact_weight"]).sum() + np.asarray(labels["foot_contact_weight"]).sum()) <= 0.0:
            raise ValueError("contact mode requested but all contact weights are zero")
        if args.global_hand_handle_pose is not None:
            prior = json.loads(args.global_hand_handle_pose.read_text(encoding="utf-8"))
            if prior.get("assumption") != "hand_static_relative_to_walker_for_entire_video":
                raise ValueError("global hand-handle prior assumption mismatch")
            if prior.get("status") != "engineering_candidate" or prior.get("geometry_source") != "camera_rigid_mount_assumption":
                raise ValueError("global prior requires validated camera-rigid solver provenance")
            global_camera_R = torch.tensor(topology["rotation_left_camera_from_walker"], dtype=torch.float32, device=device)
            global_camera_t = torch.tensor(topology["translation_left_camera_from_walker_mm"], dtype=torch.float32, device=device) / 1000
            global_hand_surface = {}
            for s in ("left", "right"):
                indices = prior["hands"][s]["vertex_indices"]
                palm_idx[s] = torch.tensor(indices, dtype=torch.long, device=device)
                global_hand_surface[s] = torch.tensor(prior["hands"][s]["shared_surface_walker_m"], dtype=torch.float32, device=device)
            global_hand_offset = torch.tensor(
                [prior["hands"][s]["palm_offset_handle_m"] for s in ("left", "right")],
                dtype=torch.float32, device=device)
    R01 = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=device)
    T01 = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=device)
    K0, D0 = cal.K0, cal.D0
    K1, D1 = cal.K1, cal.D1

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
    if not contact_enabled:
        # Without a physical contact target, D3 must not release the whole
        # body's root/translation to explain model-derived WiLoR pixels.
        stage_steps[5] = 0
    stage_names = ["A_body_vposer", "B_shared_beta", "C_body_vposer_refine",
                   "D1_hand_proximal",
                   "D2_hand_foot_surface_contact" if contact_enabled else "D2_hand_refine_no_contact",
                   "D3_hand_foot_contact_refine"]
    # Contact refinement must be able to move the body/feet as well as the
    # hands.  Keep the no-contact route unchanged: WiLoR observations are
    # model-derived and must not release the body root in that route.
    contact_body_train = ["root", "transl", "latent"] if contact_enabled else []
    stage_train = [["transl", "latent"], ["beta"],
                   ["beta", "root", "transl", "latent"], ["lhand", "rhand"],
                   ["lhand", "rhand"] + contact_body_train,
                   ["lhand", "rhand"] + contact_body_train]
    param_map = {"beta": beta, "root": root, "transl": transl,
                 "latent": latent, "lhand": lhand, "rhand": rhand}
    def set_stage(train):
        for p in (beta, root, transl, latent, lhand, rhand): p.requires_grad_(False)
        for k in train: param_map[k].requires_grad_(True)
    stage_history = []
    global_step = 0
    for stage_index, (stage_name, train_names, n_stage) in enumerate(zip(stage_names, stage_train, stage_steps)):
        # Do not let MANO initialization alter A/B/C body or shared-beta fits.
        if stage_index == 3 and args.mano_pose_init:
            with torch.no_grad():
                lhand.copy_(mano_initial["left"])
                rhand.copy_(mano_initial["right"])
        set_stage(train_names)
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
            body_2d_loss = body_loss.detach() * 0.0
            body_2d_px = body_loss.detach() * 0.0
            if args.body_reprojection_weight > 0 and stage_index in (0, 1, 2):
                e0 = torch.linalg.vector_norm(proj_cam0 - raw_left_2d, dim=-1) / args.body_reprojection_scale_px
                e1 = torch.linalg.vector_norm(proj_cam1 - raw_right_2d, dim=-1) / args.body_reprojection_scale_px
                delta = 1.0
                h0 = torch.where(e0 <= delta, 0.5 * e0.pow(2), delta * (e0 - 0.5 * delta))
                h1 = torch.where(e1 <= delta, 0.5 * e1.pow(2), delta * (e1 - 0.5 * delta))
                d0 = raw_left_conf * raw_left_valid
                d1 = raw_right_conf * raw_right_valid
                body_2d_loss = ((h0 * d0).sum() / d0.sum().clamp_min(1e-6) +
                                (h1 * d1).sum() / d1.sum().clamp_min(1e-6)) * 0.5
                body_2d_px = ((torch.linalg.vector_norm(proj_cam0 - raw_left_2d, dim=-1) * d0).sum() /
                              d0.sum().clamp_min(1e-6) +
                              (torch.linalg.vector_norm(proj_cam1 - raw_right_2d, dim=-1) * d1).sum() /
                              d1.sum().clamp_min(1e-6)) * 0.5
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
            def weighted(res, w, mask):
                return (res * w).sum() / w.sum().clamp_min(1e-6) if mask.any() else res.sum() * 0.0
            hand_l_loss = weighted(hand_ll_res, weight_ll, mask_ll) + weighted(hand_rl_res, weight_rl, mask_rl)
            hand_r_loss = weighted(hand_lr_res, weight_lr, mask_lr) + weighted(hand_rr_res, weight_rr, mask_rr)
            pose_reg = (args.vposer_prior_weight * latent.pow(2).mean()
                        + args.hand_pca_prior_weight * (lhand.pow(2).mean() + rhand.pow(2).mean()))
            hand_temporal = (lhand.sum() + rhand.sum()) * 0.0
            if n >= 3 and not args.shared_hand_pose:
                l_valid_frame = mask_ll.any(dim=1) | mask_rl.any(dim=1)
                r_valid_frame = mask_lr.any(dim=1) | mask_rr.any(dim=1)
                if mano_enabled and args.hand_2d_weight == 0:
                    l_valid_frame = mano_weights["left"].sum(dim=1) > 0
                    r_valid_frame = mano_weights["right"].sum(dim=1) > 0
                elif mano_enabled:
                    l_valid_frame = l_valid_frame | (mano_weights["left"].sum(dim=1) > 0)
                    r_valid_frame = r_valid_frame | (mano_weights["right"].sum(dim=1) > 0)
                l_edge = l_valid_frame[2:] & l_valid_frame[1:-1] & l_valid_frame[:-2]
                r_edge = r_valid_frame[2:] & r_valid_frame[1:-1] & r_valid_frame[:-2]
                l_acc = (lhand[2:] - 2 * lhand[1:-1] + lhand[:-2]).pow(2).mean(dim=1)
                r_acc = (rhand[2:] - 2 * rhand[1:-1] + rhand[:-2]).pow(2).mean(dim=1)
                l_term = l_acc[l_edge].mean() if l_edge.any() else l_acc.mean() * 0.0
                r_term = r_acc[r_edge].mean() if r_edge.any() else r_acc.mean() * 0.0
                hand_temporal = 0.5 * (l_term + r_term)
            body_temporal = body_loss.detach() * 0.0
            if args.body_temporal_weight > 0 and n >= 3 and stage_index in (0, 2):
                pelvis = 0.5 * (coco[:, 11] + coco[:, 12])
                rel = coco - pelvis[:, None, :]
                acc = rel[2:] - 2.0 * rel[1:-1] + rel[:-2]
                body_valid_frame = mask_body.all(dim=1)
                edge = body_valid_frame[2:] & body_valid_frame[1:-1] & body_valid_frame[:-2]
                acc_norm = torch.linalg.vector_norm(acc, dim=-1)
                # Huber in metres: preserve genuine motion while suppressing
                # one-frame detector spikes. Delta is fixed and recorded.
                delta = 0.03
                robust = torch.where(acc_norm <= delta,
                                     0.5 * acc_norm.pow(2),
                                     delta * (acc_norm - 0.5 * delta))
                body_temporal = robust[edge].mean() if edge.any() else robust.mean() * 0.0
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
            contact_hand_loss = left_in_cam0.sum() * 0.0
            contact_foot_loss = left_in_cam0.sum() * 0.0
            global_hand_loss = left_in_cam0.sum() * 0.0
            if contact_enabled and stage_index >= 4:
                vg = torch.einsum("nij,nvj->nvi", contact_R, out.vertices) + contact_T[:, None, :]
                hand_terms = []
                foot_terms = []
                for side, j in (("left", 0), ("right", 1)):
                    pts = vg[:, palm_idx[side], :]
                    a, b = contact_handle[:, j, 0, :], contact_handle[:, j, 1, :]
                    hand_terms.append(surface_contact.hand_surface_loss(pts, a[:, None, :], b[:, None, :], 0.016)
                                      * contact_hand_weight[:, j])
                for side, j in (("left", 0), ("right", 1)):
                    sole_z = vg[:, sole_idx[side], 2]
                    foot_terms.append(surface_contact.foot_surface_loss(sole_z) * contact_foot_weight[:, j])
                hand_den = contact_hand_weight.sum().clamp_min(1e-6)
                foot_den = contact_foot_weight.sum().clamp_min(1e-6)
                contact_hand_loss = torch.stack(hand_terms, dim=1).sum() / hand_den
                contact_foot_loss = torch.stack(foot_terms, dim=1).sum() / foot_den
                if global_hand_offset is not None:
                    terms = []
                    for side in ("left", "right"):
                        pc = out.vertices[:, palm_idx[side], :]
                        pw = torch.einsum("ij,nvj->nvi", global_camera_R.T, pc-global_camera_t)
                        err = torch.linalg.vector_norm(pw-global_hand_surface[side],dim=-1)/0.03
                        terms.append(torch.where(err <= 1, .5*err.pow(2),err-.5).mean())
                    global_hand_loss = torch.stack(terms).mean()
            reproj_term = (args.body_reprojection_weight * body_2d_loss
                           if stage_index in (0, 1, 2) else body_2d_loss.detach() * 0.0)
            contact_total = (args.surface_hand_contact_weight * contact_hand_loss
                             + args.surface_foot_contact_weight * contact_foot_loss)
            if global_hand_offset is not None and stage_index >= 4:
                contact_total = contact_total + args.global_hand_handle_weight * global_hand_loss
            contact_anchor = (args.root_anchor_weight * root_anchor
                              if contact_enabled and stage_index >= 4 else root_anchor.detach() * 0.0)
            loss = body_term + reproj_term + structure_term + root_term + contact_anchor + temporal_term + \
                   args.body_temporal_weight * body_temporal + hand_term + args.mano_pose_weight * mano_pose_loss + pose_reg + 1e-3 * beta.pow(2).mean() + \
                   (contact_total if stage_index >= 4 else contact_total.detach() * 0.0)
            loss.backward()
            optim.step()
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
                history.append(row)
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
        contact_diag = {}
        if mano_enabled:
            for side in ("left", "right"):
                contact_diag[f"mano_{side}_target_rotations"] = mano_targets[side].cpu().numpy()
                contact_diag[f"mano_{side}_view_weights"] = mano_weights[side].cpu().numpy()
                contact_diag[f"mano_{side}_initial_pca"] = mano_initial[side].cpu().numpy()
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


if __name__ == "__main__":
    raise SystemExit(main())
