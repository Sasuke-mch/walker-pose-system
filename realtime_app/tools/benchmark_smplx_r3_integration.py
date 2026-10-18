#!/usr/bin/env python3
"""Benchmark SMPL-X R3-integration routes on frozen people_1 windows.

Routes (frozen definitions; all share window 60-90, fixed shared betas, the
same raw-fisheye projection, COCO offset observation model, robust kernel and
temporal/foot/hand terms; no display smoothing):

  C-prefit     prefit-based initialization; formal loss = L/R 2D + existing
               prefit soft 3D + temporal + priors + contacts.  Re-run common
               baseline (never references the old C1000 summary as a result).
  C-R3-init    R3 high-quality joints initialize root translation/rotation and
               VPoser latent through a short pre-optimisation; the R3 loss is
               fully removed during formal optimisation (loss identical
               item-by-item with C-prefit).
  B-R3-init    same R3 initialization; formal optimisation cancels ALL 3D data
               terms (prefit and R3), keeping only L/R 2D, temporal,
               VPoser/shape priors, foot-ground and hand-rail terms.
  C-R3-soft    R3 initialization; formal loss keeps L/R 2D and prefit soft 3D
               plus an independent high-quality R3 robust soft 3D term
               (normalised by valid joint count) at one small weight from
               {0.10, 0.25, 0.50, 1.00, 2.00}.

Usage:
  python benchmark_smplx_r3_integration.py --experiment-dir <dir> [--mode auto|dev|validate|render|aggregate]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.fisheye_camera import (  # noqa: E402
    fisheye_project_numpy,
    fisheye_project_torch,
    load_stereo_fisheye,
    world_to_cameras_torch_batch,
)
from pose_app.joint_offset_model import JointOffsetModel  # noqa: E402
from pose_app.reprojection_common import (  # noqa: E402
    build_2d_observations,
    build_view_2d_weights_from_obs,
    robust_gm,
)
from pose_app.smplx_fitting import (  # noqa: E402
    BODY_COCO_INDICES,
    COCO17_TO_SMPLX,
    initial_world_from_model_rotation,
    load_vposer_explicit,
    observation_weights,
    read_jsonl,
)
from pose_app.smplx_r3_integration import (  # noqa: E402
    ALL_ROUTES,
    R3_INIT_ITERATIONS,
    R3_INIT_LEARNING_RATE,
    ROUTE_B_R3_INIT,
    ROUTE_C_PREFIT,
    ROUTE_C_R3_INIT,
    ROUTE_C_R3_SOFT,
    R3_SOFT_WEIGHT_GRID,
    R3ContractError,
    build_r3_soft_weights,
    load_r3_jsonl,
    r3_quality_mask,
    r3_rejection_reasons,
    r3_soft_loss_torch,
    r3_targets_m,
)
from pose_app.temporal_body_prefit import (  # noqa: E402
    TemporalBodyPrefitConfig,
    build_contact_targets,
)

ITERATIONS = 400
CHECKPOINTS = (0, 10, 25, 50, 100, 200, 400)
TIMED_RUNS = 3
LEARNING_RATE = 0.02
HOLDOUT_JOINTS = (9, 16)  # COCO left wrist / right ankle (frozen)
AUDIT_FRAMES = (60, 75, 90)
DEV_WINDOW = (60, 90)
VALIDATION_WINDOWS = ((129, 159), (278, 308), (373, 403))

WEIGHTS = {
    "data3d": 120.0,
    "data2d": 40.0,
    "temporal": 220.0,
    "foot_contact": 100.0,
    "hand_contact": 30.0,
    "latent": 0.02,
    "betas": 0.04,
}
GM_SCALE_PX = 20.0
GM_SCALE_M = 0.03

JOINT_NAMES = {
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}


@dataclass(frozen=True)
class FitRouteConfig:
    name: str
    init: str  # "prefit" | "r3"
    data3d: bool  # prefit soft 3D term in the formal loss
    r3soft_weight: float | None  # C-R3-soft only
    temporal_weight: float | None = None  # None keeps the frozen v1 weight
    temporal_mode: str = "legacy"  # legacy | factorized
    root_temporal_weight: float | None = None
    local_temporal_weight: float | None = None
    # Explicit point-space contract for newer isolated routes.  Defaults retain
    # the archived C-prefit behaviour exactly.
    temporal_point_space: str = "body"  # body | coco
    include_head_observations: bool = False
    include_head_temporal: bool = False
    pelvis_anchor_weight: float = 0.0
    pelvis_net_weight: float = 0.0
    pelvis_direction_weight: float = 0.0
    anatomical_temporal_weight: float = 0.0
    # Optional mechanism-isolation route: auxiliary world-space targets may
    # update translation/pose but must not rotate the global body against 2D.
    auxiliary_detach_root_orientation: bool = False
    auxiliary_detach_root_joints: tuple[int, ...] = ()
    foot_contact_weight: float = WEIGHTS["foot_contact"]
    hand_contact_weight: float = WEIGHTS["hand_contact"]
    preserve_translation_endpoints: bool = False
    preserve_translation_net_delta: bool = False
    preserve_coco_pelvis_net_delta: bool = False
    structured_knot_stride: int = 5
    structured_interpolation: str = "linear"


def route_configs() -> list[FitRouteConfig]:
    configs = [
        FitRouteConfig(ROUTE_C_PREFIT, init="prefit", data3d=True, r3soft_weight=None),
        FitRouteConfig(ROUTE_C_R3_INIT, init="r3", data3d=True, r3soft_weight=None),
        FitRouteConfig(ROUTE_B_R3_INIT, init="r3", data3d=False, r3soft_weight=None),
    ]
    for weight in R3_SOFT_WEIGHT_GRID:
        configs.append(
            FitRouteConfig(f"{ROUTE_C_R3_SOFT}_w{weight:05.2f}", init="r3", data3d=True, r3soft_weight=weight)
        )
    return configs


def config_subdir(config: FitRouteConfig) -> str:
    return config.name


def gate_prefit_weights_by_accepted_observation(weights: np.ndarray, obs2d: dict) -> np.ndarray:
    """Fail closed: a soft-3D target requires an accepted observation in either view."""
    accepted_any = np.asarray(obs2d["accepted_left"], dtype=bool) | np.asarray(
        obs2d["accepted_right"], dtype=bool
    )
    if weights.shape != accepted_any.shape:
        raise ValueError(f"soft-3D weights {weights.shape} do not match observations {accepted_any.shape}")
    return np.asarray(weights).copy() * accepted_any


class FitEnv:
    """Shared inputs and tensors for one window."""

    def __init__(
        self,
        args: argparse.Namespace,
        start_frame: int,
        end_frame: int,
        device: str = "cpu",
    ) -> None:
        import torch
        from human_body_prior.tools.rotation_tools import matrot2aa

        self.torch = torch
        self.matrot2aa = matrot2aa
        self.args = args
        self.device = torch.device(device)

        motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
        prefit = read_jsonl(args.prefit_jsonl)
        strict_foot_labels = getattr(args, "foot_labels_jsonl", None)
        interactions = (
            [{} for _ in motion["frames"]]
            if strict_foot_labels
            else read_jsonl(args.interaction_jsonl)
        )
        prior_mode = getattr(args, "smplx_prior_mode", "inherited")
        if prior_mode not in ("inherited", "clean_zero"):
            raise ValueError(f"unsupported SMPL-X prior mode: {prior_mode}")
        initial_fit = (
            json.loads(args.initial_fit_summary.read_text(encoding="utf-8"))
            if prior_mode == "inherited"
            else None
        )
        stereo_rows = read_jsonl(args.stereo_jsonl)
        stereo_by_pair = {int(row["pair_id"]): row for row in stereo_rows}
        cal = load_stereo_fisheye(args.calibration_dir)
        offset_model = (
            JointOffsetModel.from_audit_json(args.offset_audit_json)
            if prior_mode == "inherited"
            else JointOffsetModel({})
        )
        if strict_foot_labels:
            r3_records = {}
        else:
            try:
                r3_records = load_r3_jsonl(args.r3_jsonl)
            except R3ContractError as error:
                raise SystemExit(f"R3 data contract violation: {error}")

        frame_count = len(motion["frames"])
        if not (len(prefit) == len(interactions) == frame_count):
            raise ValueError("motion, prefit and interaction streams must align")
        if not (0 <= start_frame <= end_frame < frame_count):
            raise ValueError("invalid inclusive frame window")

        self.motion = motion
        self.frames = motion["frames"]
        self.walker = motion["walker"]
        self.prefit_rows = prefit
        self.interactions = interactions
        self.initial_fit = initial_fit
        self.stereo_by_pair = stereo_by_pair
        self.cal = cal
        self.offset_model = offset_model
        self.smplx_prior_mode = prior_mode
        self.r3_records = r3_records

        indices = np.arange(start_frame, end_frame + 1)
        self.indices = indices
        self.window_size = len(indices)
        self.window_frames = [self.frames[index] for index in indices]
        self.pair_ids = [int(self.frames[index]["pair"]) for index in indices]

        # Full-sequence prefit points (world mm, COCO 17).
        self.full_prefit_mm = np.asarray(
            [[row["points_ground_mm"][str(joint)] for joint in range(17)] for row in prefit],
            dtype=np.float64,
        )
        self.full_motion_mm = np.asarray(
            [[point if point is not None else [np.nan, np.nan, np.nan] for point in frame["points_mm"]]
             for frame in self.frames],
            dtype=np.float64,
        )
        self.errors = np.asarray(
            [[value if value is not None else np.nan for value in frame["errors_px"]] for frame in self.frames],
            dtype=np.float64,
        )
        self.strict = np.asarray([frame["strict"] for frame in self.frames], dtype=bool)

        if strict_foot_labels:
            contact_targets_mm = np.zeros_like(self.full_motion_mm)
            contact_weights = np.zeros((frame_count, 17), dtype=np.float64)
            contact_audit = {"source": "strict_foot_labels_only", "legacy_targets_loaded": False}
        else:
            contact_targets_mm, contact_weights, contact_audit = build_contact_targets(
                self.full_motion_mm, self.frames, interactions, motion["walker"], TemporalBodyPrefitConfig()
            )
        self.contact_targets_mm = contact_targets_mm
        self.contact_weights = contact_weights
        self.contact_audit = contact_audit

        body_model = self._create_smplx()
        self.body_model = body_model
        vposer, _, checkpoint_path = load_vposer_explicit(args.vposer_dir, args.device)
        self.vposer = vposer
        self.vposer_checkpoint = str(checkpoint_path)
        if prior_mode == "inherited":
            beta_record = json.loads(args.fixed_betas_json.read_text(encoding="utf-8"))
            self.initial_betas = np.asarray(
                beta_record["beta_statistics"]["coordinatewise_median_shared_beta"], dtype=np.float32
            )
            self.shape_mode = "fixed_cross_clip_coordinatewise_median"
            self.initial_latent = np.asarray(
                initial_fit["parameters"]["vposer_latent"], dtype=np.float32
            )
            self.prior_provenance = {
                "mode": prior_mode,
                "fixed_betas_json": str(args.fixed_betas_json),
                "offset_audit_json": str(args.offset_audit_json),
                "initial_fit_summary": str(args.initial_fit_summary),
            }
        else:
            self.initial_betas = np.zeros(10, dtype=np.float32)
            self.shape_mode = "clean_zero_unfitted"
            self.initial_latent = np.zeros(32, dtype=np.float32)
            self.prior_provenance = {
                "mode": prior_mode,
                "fixed_betas_json": None,
                "offset_audit_json": None,
                "initial_fit_summary": None,
            }

        self.coco_indices = np.asarray(BODY_COCO_INDICES, dtype=np.int64)
        self.smplx_indices = np.asarray([COCO17_TO_SMPLX[index] for index in BODY_COCO_INDICES], dtype=np.int64)
        self.all_smplx_indices = np.asarray(
            [COCO17_TO_SMPLX[index] for index in range(17)], dtype=np.longlong
        )
        self.betas_tensor = torch.as_tensor(
            self.initial_betas, dtype=torch.float32, device=self.device
        ).reshape(1, 10).expand(self.window_size, -1)
        self.per_frame_rotation = [np.asarray(self.frames[index]["rotation_world_from_left"], dtype=np.float64) for index in indices]
        self.per_frame_translation = [np.asarray(self.frames[index]["translation_world_from_left_mm"], dtype=np.float64) for index in indices]

        self._build_window_tensors()

    def _create_smplx(self):
        import smplx
        return smplx.create(
            str(self.args.model_root.resolve()), model_type="smplx",
            gender=getattr(self.args, "smplx_gender", "male"),
            ext="npz", use_pca=False, batch_size=self.window_size,
        ).to(self.device)

    def _build_window_tensors(self) -> None:
        torch = self.torch
        indices = self.indices
        prefit_mm = self.full_prefit_mm[indices]

        rotations = np.stack([initial_world_from_model_rotation(prefit_mm[index]) for index in range(self.window_size)])
        rotation_tensor = torch.as_tensor(rotations, dtype=torch.float32, device=self.device)
        latent_init = np.tile(self.initial_latent, (self.window_size, 1))

        # SMPL-X applies global_orient around the shaped root joint.  Rotating a
        # zero-pose pelvis around the world origin therefore gives the wrong
        # translation (historically about 0.4--0.5 m here).  Evaluate the exact
        # selected gender/beta/pose/root state at transl=0 and align the output
        # COCO hip midpoint instead.  This also keeps any explicit observation
        # offset inside the same forward contract as the later loss.
        with torch.no_grad():
            root_orient = self.matrot2aa(rotation_tensor).reshape(self.window_size, 3)
            latent_tensor = torch.as_tensor(
                latent_init, dtype=torch.float32, device=self.device
            )
            body_pose = self.vposer.decode(latent_tensor)["pose_body"].reshape(
                self.window_size, -1
            )
            zero_translation = torch.zeros(
                (self.window_size, 3), dtype=torch.float32, device=self.device
            )
            initial_result = self.body_model(
                betas=self.betas_tensor,
                body_pose=body_pose,
                global_orient=root_orient,
                transl=zero_translation,
                return_verts=False,
            )
            initial_coco = self.offset_model.predict_coco_torch(
                initial_result.joints[:, self.all_smplx_indices]
            )
            predicted_hip = 0.5 * (initial_coco[:, 11] + initial_coco[:, 12])
            target_hip = torch.as_tensor(
                prefit_mm[:, [11, 12]].mean(axis=1) / 1000.0,
                dtype=torch.float32,
                device=self.device,
            )
            translation_init = (target_hip - predicted_hip).cpu().numpy()

        self.base_rotations = rotation_tensor
        self.base_translation_init = np.asarray(translation_init, dtype=np.float64)
        self.base_latent_init = latent_init
        self.translation_initialization = "forward_coco_pelvis_alignment"

        # prefit soft 3D targets (12 body joints, metres) + weights.
        coco = self.coco_indices
        self.prefit_target_m = torch.as_tensor(
            prefit_mm[:, coco] / 1000.0, dtype=torch.float32, device=self.device
        )
        weights_np = np.stack([
            observation_weights(self.errors[index], self.strict[index])[coco] for index in indices
        ])
        self.prefit_weights_np = weights_np
        self.prefit_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=self.device)

        # contacts (17 joints, world mm -> m).
        self.contacts = torch.as_tensor(
            self.contact_targets_mm[indices] / 1000.0, dtype=torch.float32, device=self.device
        )
        self.contact_mask_np = self.contact_weights[indices].copy()
        self.foot_contact_mode = "legacy_target_proxy"
        foot_labels_path = getattr(self.args, "foot_labels_jsonl", None)
        if foot_labels_path:
            label_rows = [
                json.loads(line) for line in Path(foot_labels_path).read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            if len(label_rows) != frame_count:
                raise ValueError("foot label stream must align with the full motion stream")
            strict_foot_mask = np.zeros((frame_count, 17), dtype=np.float64)
            for frame_index, row in enumerate(label_rows):
                if row.get("frame_index") != frame_index:
                    raise ValueError("foot label frame indices must be contiguous and zero-based")
                for side, joint in (("left", 15), ("right", 16)):
                    label = (row.get(side) or {}).get("label")
                    weight = float((row.get(side) or {}).get("contact_weight") or 0.0)
                    if label not in {"support", "stage2_contact_candidate"}:
                        weight = 0.0
                    strict_foot_mask[frame_index, joint] = max(weight, 0.0)
            self.contact_mask_np = strict_foot_mask[indices]
            self.foot_contact_mode = "ground_z_pseudohuber_delta_25mm"
        self.contact_mask = torch.as_tensor(self.contact_mask_np, dtype=torch.float32, device=self.device)

        # 2D observations.
        obs2d = build_2d_observations(
            self.stereo_by_pair, self.frames, indices, self.coco_indices, self.cal.image_width, self.cal.image_height
        )
        self.obs2d = obs2d
        self.prefit_weights_np = gate_prefit_weights_by_accepted_observation(
            self.prefit_weights_np, obs2d
        )
        self.prefit_weights = torch.as_tensor(
            self.prefit_weights_np, dtype=torch.float32, device=self.device
        )
        self.kp_left = torch.as_tensor(np.nan_to_num(obs2d["kp_left_px"]), dtype=torch.float32, device=self.device)
        self.kp_right = torch.as_tensor(np.nan_to_num(obs2d["kp_right_px"]), dtype=torch.float32, device=self.device)
        self.kp_valid_l = torch.as_tensor(obs2d["accepted_left"], dtype=torch.float32, device=self.device)
        self.kp_valid_r = torch.as_tensor(obs2d["accepted_right"], dtype=torch.float32, device=self.device)
        w2d_left, w2d_right = build_view_2d_weights_from_obs(obs2d)
        self.w2d_left_np = w2d_left
        self.w2d_right_np = w2d_right
        self.w2d_left = torch.as_tensor(w2d_left, dtype=torch.float32, device=self.device)
        self.w2d_right = torch.as_tensor(w2d_right, dtype=torch.float32, device=self.device)

        # Full COCO-17 observations are constructed separately.  They are used
        # only by routes that explicitly enable quality-gated head observations.
        self.all_coco_indices = np.arange(17, dtype=np.int64)
        self.obs2d_all = build_2d_observations(
            self.stereo_by_pair, self.frames, indices, self.all_coco_indices,
            self.cal.image_width, self.cal.image_height,
        )
        w_all_l, w_all_r = build_view_2d_weights_from_obs(self.obs2d_all)
        # Head weights are frozen and only affect accepted observations.
        head_scale = np.asarray([0.50, 0.25, 0.25, 0.20, 0.20] + [1.0] * 12, dtype=np.float64)
        self.w2d_all_l = torch.as_tensor(w_all_l * head_scale[None, :], dtype=torch.float32, device=self.device)
        self.w2d_all_r = torch.as_tensor(w_all_r * head_scale[None, :], dtype=torch.float32, device=self.device)
        self.w2d_all_l_np = w_all_l * head_scale[None, :]
        self.w2d_all_r_np = w_all_r * head_scale[None, :]
        self.kp_all_left = torch.as_tensor(np.nan_to_num(self.obs2d_all["kp_left_px"]), dtype=torch.float32, device=self.device)
        self.kp_all_right = torch.as_tensor(np.nan_to_num(self.obs2d_all["kp_right_px"]), dtype=torch.float32, device=self.device)
        self.kp_all_valid_l = torch.as_tensor(self.obs2d_all["accepted_left"], dtype=torch.float32, device=self.device)
        self.kp_all_valid_r = torch.as_tensor(self.obs2d_all["accepted_right"], dtype=torch.float32, device=self.device)
        all_prefit_w = np.stack([observation_weights(self.errors[index], self.strict[index]) for index in indices])
        # A head target is never invented: both the upstream strict flag and a
        # finite prefit point are required before it can enter soft-3D.
        all_prefit_w = gate_prefit_weights_by_accepted_observation(all_prefit_w, self.obs2d_all)
        all_prefit_w[:, :5] *= np.isfinite(prefit_mm[:, :5]).all(axis=2)
        all_prefit_w[:, :5] *= head_scale[None, :5]
        self.prefit_all_target_m = torch.as_tensor(prefit_mm / 1000.0, dtype=torch.float32, device=self.device)
        self.prefit_all_weights = torch.as_tensor(all_prefit_w, dtype=torch.float32, device=self.device)

        # R3 soft weights + targets over the window.
        r3_weights, r3_audit = build_r3_soft_weights(
            self.r3_records, indices, self.coco_indices, set()
        )
        self.r3_weights_np = r3_weights
        self.r3_weights = torch.as_tensor(r3_weights, dtype=torch.float32, device=self.device)
        self.r3_target_m = torch.as_tensor(
            r3_targets_m(self.r3_records, indices, self.coco_indices, r3_weights),
            dtype=torch.float32, device=self.device,
        )
        self.r3_audit = r3_audit

        self.input_accel_p95 = self._acceleration_p95_mm(prefit_mm)
        self.pelvis_input = self._pelvis_net_displacement_mm(prefit_mm)

    # ------------------------------------------------------------------ utils
    def _acceleration_p95_mm(self, points_mm: np.ndarray) -> float:
        accel = points_mm[2:] - 2.0 * points_mm[1:-1] + points_mm[:-2]
        return float(np.percentile(np.linalg.norm(accel, axis=2), 95))

    def _pelvis_net_displacement_mm(self, points_mm: np.ndarray) -> float:
        pelvis = 0.5 * (points_mm[:, 11] + points_mm[:, 12])
        return float(np.linalg.norm(pelvis[-1] - pelvis[0]))

    def make_params(self, hold_out: set[int]):
        torch = self.torch
        n = self.window_size
        root_orient = torch.nn.Parameter(
            self.matrot2aa(self.base_rotations).reshape(n, 3).detach().clone()
        )
        translation = torch.nn.Parameter(
            torch.tensor(self.base_translation_init.copy(), dtype=torch.float32, device=self.device)
        )
        # The stored single-frame latent was fitted with all observed joints and
        # is therefore target-derived.  A strict hold-out run must not inherit
        # wrist/ankle information through that latent; start it from the neutral
        # VPoser code instead.  Ordinary runs keep the frozen common start.
        latent_init = (
            np.zeros_like(self.base_latent_init)
            if hold_out
            else self.base_latent_init
        )
        latent = torch.nn.Parameter(
            torch.tensor(np.asarray(latent_init).copy(), dtype=torch.float32, device=self.device)
        )
        return root_orient, translation, latent

    def masks_with_holdout(self, hold_out: set[int]):
        w2d_l = self.w2d_left_np.copy()
        w2d_r = self.w2d_right_np.copy()
        prefit_w = self.prefit_weights_np.copy()
        contact_mask = self.contact_mask_np.copy()
        r3_w = self.r3_weights_np.copy()
        w2d_all_l = self.w2d_all_l_np.copy()
        w2d_all_r = self.w2d_all_r_np.copy()
        prefit_all_w = self.prefit_all_weights.detach().cpu().numpy().copy()
        temporal_body = np.ones(len(BODY_COCO_INDICES), dtype=np.float32)
        temporal_all = np.ones(17, dtype=np.float32)
        for position, coco_index in enumerate(BODY_COCO_INDICES):
            if coco_index in hold_out:
                w2d_l[:, position] = 0.0
                w2d_r[:, position] = 0.0
                prefit_w[:, position] = 0.0
                contact_mask[:, coco_index] = 0.0
                r3_w[:, position] = 0.0
                temporal_body[position] = 0.0
        for coco_index in hold_out:
            if 0 <= coco_index < 17:
                w2d_all_l[:, coco_index] = 0.0
                w2d_all_r[:, coco_index] = 0.0
                prefit_all_w[:, coco_index] = 0.0
                contact_mask[:, coco_index] = 0.0
                temporal_all[coco_index] = 0.0
        torch = self.torch
        return {
            "w2d_l": torch.as_tensor(w2d_l, dtype=torch.float32, device=self.device),
            "w2d_r": torch.as_tensor(w2d_r, dtype=torch.float32, device=self.device),
            "prefit_w": torch.as_tensor(prefit_w, dtype=torch.float32, device=self.device),
            "contact_mask": torch.as_tensor(contact_mask, dtype=torch.float32, device=self.device),
            "r3_w": torch.as_tensor(r3_w, dtype=torch.float32, device=self.device),
            "w2d_all_l": torch.as_tensor(w2d_all_l, dtype=torch.float32, device=self.device),
            "w2d_all_r": torch.as_tensor(w2d_all_r, dtype=torch.float32, device=self.device),
            "prefit_all_w": torch.as_tensor(prefit_all_w, dtype=torch.float32, device=self.device),
            "temporal_body": torch.as_tensor(temporal_body, dtype=torch.float32, device=self.device),
            "temporal_all": torch.as_tensor(temporal_all, dtype=torch.float32, device=self.device),
        }

    # ---------------------------------------------------------------- forward
    def forward(self, root_orient, translation, latent, betas=None):
        body_pose = self.vposer.decode(latent)["pose_body"].reshape(self.window_size, -1)
        beta_values = self.betas_tensor if betas is None else betas.reshape(1, 10).expand(self.window_size, -1)
        result = self.body_model(
            betas=beta_values,
            body_pose=body_pose, global_orient=root_orient, transl=translation, return_verts=True,
        )
        predicted_coco = self.offset_model.predict_coco_torch(result.joints[:, self.all_smplx_indices])
        predicted_coco_body = predicted_coco[:, self.coco_indices]
        predicted_body = result.joints[:, self.smplx_indices]
        return result, predicted_coco, predicted_coco_body, predicted_body

    def loss_terms(self, root_orient, translation, latent, masks, config: FitRouteConfig) -> dict:
        torch = self.torch
        n = self.window_size
        result, predicted_coco, predicted_coco_body, predicted_body = self.forward(
            root_orient, translation, latent
        )
        if config.auxiliary_detach_root_orientation or config.auxiliary_detach_root_joints:
            auxiliary_result, auxiliary_coco, auxiliary_coco_body, _ = self.forward(
                root_orient.detach(), translation, latent
            )
            if config.auxiliary_detach_root_joints and not config.auxiliary_detach_root_orientation:
                detached = set(config.auxiliary_detach_root_joints)
                auxiliary_coco = torch.stack([
                    auxiliary_coco[:, j] if j in detached else predicted_coco[:, j]
                    for j in range(17)
                ], dim=1)
                auxiliary_coco_body = auxiliary_coco[:, self.coco_indices]
                auxiliary_predicted_all = auxiliary_coco
            else:
                auxiliary_predicted_all = auxiliary_coco
        else:
            auxiliary_result = result
            auxiliary_coco = predicted_coco
            auxiliary_coco_body = predicted_coco_body
            auxiliary_predicted_all = predicted_coco
        terms: dict[str, float] = {}

        if config.include_head_observations:
            data_points, kp_l, kp_r = predicted_coco, self.kp_all_left, self.kp_all_right
            valid_l, valid_r, w_l, w_r = self.kp_all_valid_l, self.kp_all_valid_r, masks["w2d_all_l"], masks["w2d_all_r"]
        else:
            data_points, kp_l, kp_r = predicted_coco_body, self.kp_left, self.kp_right
            valid_l, valid_r, w_l, w_r = self.kp_valid_l, self.kp_valid_r, masks["w2d_l"], masks["w2d_r"]
        left_m, right_m = world_to_cameras_torch_batch(data_points, self.per_frame_rotation, self.per_frame_translation, self.cal)
        proj_l = fisheye_project_torch(left_m, self.cal.K0, self.cal.D0)
        proj_r = fisheye_project_torch(right_m, self.cal.K1, self.cal.D1)
        residual_l = torch.linalg.vector_norm(proj_l - kp_l, dim=2) * valid_l
        residual_r = torch.linalg.vector_norm(proj_r - kp_r, dim=2) * valid_r
        scale_px2 = GM_SCALE_PX * GM_SCALE_PX
        loss_l = torch.sum(w_l * (residual_l * residual_l) / (residual_l * residual_l + scale_px2)) / torch.clamp(w_l.sum(), min=1.0)
        loss_r = torch.sum(w_r * (residual_r * residual_r) / (residual_r * residual_r + scale_px2)) / torch.clamp(w_r.sum(), min=1.0)
        terms["data2d"] = 0.5 * (loss_l + loss_r)

        if config.data3d:
            target3d, weight3d, points3d = (self.prefit_all_target_m, masks["prefit_all_w"], auxiliary_coco) if config.include_head_observations else (self.prefit_target_m, masks["prefit_w"], auxiliary_coco_body)
            residual3d = points3d - target3d
            data3d_loss = torch.sum(weight3d * robust_gm(residual3d, GM_SCALE_M)) / torch.clamp(weight3d.sum(), min=1.0)
            terms["data3d"] = data3d_loss

        if config.r3soft_weight is not None:
            r3_loss = r3_soft_loss_torch(predicted_coco_body, self.r3_target_m, masks["r3_w"])
            terms["r3soft"] = r3_loss

        temporal_points = predicted_body if config.temporal_point_space == "body" else (predicted_coco if config.include_head_temporal else predicted_coco_body)
        acceleration = temporal_points[2:] - 2.0 * temporal_points[1:-1] + temporal_points[:-2]
        temporal_mask = masks["temporal_all"] if config.include_head_temporal else masks["temporal_body"]
        temporal_squared = acceleration.square().sum(dim=2)
        terms["temporal"] = torch.sum(temporal_squared * temporal_mask[None, :]) / torch.clamp(temporal_mask.sum() * acceleration.shape[0] * 3.0, min=1.0)
        terms["temporal_coco_main"] = terms["temporal"] if config.temporal_point_space == "coco" else torch.zeros_like(terms["temporal"])
        anatomical_acceleration = predicted_body[2:] - 2.0 * predicted_body[1:-1] + predicted_body[:-2]
        terms["anatomical_temporal_regularizer"] = torch.mean(anatomical_acceleration.square())
        # BODY_COCO_INDICES is COCO 5..16, so local positions 6/7 are hips 11/12.
        # Separating the pelvis trajectory from pelvis-relative joints prevents
        # a stronger local smoothness prior from directly suppressing walking.
        pelvis = 0.5 * (predicted_body[:, 6] + predicted_body[:, 7])
        pelvis_acceleration = pelvis[2:] - 2.0 * pelvis[1:-1] + pelvis[:-2]
        relative_body = predicted_body - pelvis[:, None, :]
        local_acceleration = relative_body[2:] - 2.0 * relative_body[1:-1] + relative_body[:-2]
        terms["temporal_root"] = torch.mean(pelvis_acceleration.square())
        terms["temporal_local"] = torch.mean(local_acceleration.square())
        pelvis_coco = 0.5 * (predicted_coco[:, 11] + predicted_coco[:, 12])
        pelvis_prefit = 0.5 * (self.prefit_all_target_m[:, 11] + self.prefit_all_target_m[:, 12])
        terms["pelvis_frame_anchor"] = torch.mean(robust_gm(pelvis_coco - pelvis_prefit, GM_SCALE_M))
        pred_delta = pelvis_coco[-1] - pelvis_coco[0]
        ref_delta = pelvis_prefit[-1] - pelvis_prefit[0]
        terms["pelvis_net_displacement"] = torch.mean(robust_gm((pred_delta - ref_delta).reshape(1, 3), GM_SCALE_M))
        cosine = torch.sum(pred_delta * ref_delta) / torch.clamp(torch.linalg.vector_norm(pred_delta) * torch.linalg.vector_norm(ref_delta), min=1e-8)
        terms["pelvis_direction"] = 1.0 - cosine
        terms["latent"] = torch.mean(latent.square())

        predicted_all = auxiliary_predicted_all
        foot_mask = torch.zeros_like(masks["contact_mask"])
        foot_mask[:, 15:17] = masks["contact_mask"][:, 15:17]
        hand_mask = torch.zeros_like(masks["contact_mask"])
        hand_mask[:, 9:11] = masks["contact_mask"][:, 9:11]
        if self.foot_contact_mode == "ground_z_pseudohuber_delta_25mm":
            # The model-side point is the COCO observation regressor output,
            # already in the current ground frame. Stage-2 labels only provide
            # a low-weight engineering regularizer because the ground pose can
            # itself use ankle anchors.
            delta = 0.025
            distance = predicted_all[:, 15:17, 2]
            rho = delta * delta * (torch.sqrt(1.0 + (distance / delta).square()) - 1.0)
            foot_weights = masks["contact_mask"][:, 15:17]
            terms["foot_contact"] = torch.sum(foot_weights * rho) / torch.clamp(foot_weights.sum(), min=1.0)
        else:
            contact_residual = torch.linalg.vector_norm(predicted_all - self.contacts, dim=2)
            terms["foot_contact"] = torch.sum(foot_mask * contact_residual.square()) / torch.clamp(foot_mask.sum(), min=1.0)
        contact_residual = torch.linalg.vector_norm(predicted_all - self.contacts, dim=2)
        terms["hand_contact"] = torch.sum(hand_mask * contact_residual.square()) / torch.clamp(hand_mask.sum(), min=1.0)
        return terms

    def total_loss(self, terms: dict, config: FitRouteConfig):
        if config.temporal_mode == "factorized":
            if config.root_temporal_weight is None or config.local_temporal_weight is None:
                raise ValueError("factorized temporal mode requires root and local weights")
            temporal_total = (
                float(config.root_temporal_weight) * terms["temporal_root"]
                + float(config.local_temporal_weight) * terms["temporal_local"]
            )
        else:
            temporal_weight = (
                WEIGHTS["temporal"] if config.temporal_weight is None
                else float(config.temporal_weight)
            )
            temporal_total = temporal_weight * terms["temporal"]
        total = (
            temporal_total + WEIGHTS["latent"] * terms["latent"]
            + float(config.foot_contact_weight) * terms["foot_contact"]
            + float(config.hand_contact_weight) * terms["hand_contact"]
        )
        if "data3d" in terms:
            total = total + WEIGHTS["data3d"] * terms["data3d"]
        if "data2d" in terms:
            total = total + WEIGHTS["data2d"] * terms["data2d"]
        if "r3soft" in terms:
            total = total + float(config.r3soft_weight) * terms["r3soft"]
        total = total + float(config.anatomical_temporal_weight) * terms["anatomical_temporal_regularizer"]
        total = total + float(config.pelvis_anchor_weight) * terms["pelvis_frame_anchor"]
        total = total + float(config.pelvis_net_weight) * terms["pelvis_net_displacement"]
        total = total + float(config.pelvis_direction_weight) * terms["pelvis_direction"]
        return total

    @staticmethod
    def terms_to_float(terms: dict) -> dict:
        return {key: float(value.detach().cpu()) for key, value in terms.items()}

    # ---------------------------------------------------------------- metrics
    def snapshot(self, root_orient, translation, latent, masks, config: FitRouteConfig, hold_out: set[int]) -> dict:
        """Numpy metrics at the current parameters (no video smoothing)."""
        torch = self.torch
        with torch.no_grad():
            result, predicted_coco, predicted_coco_body, predicted_body = self.forward(
                root_orient, translation, latent
            )
        n = self.window_size
        predicted_coco_mm = predicted_coco.cpu().numpy() * 1000.0
        predicted_body_m = predicted_body.cpu().numpy()

        reproj = self._reprojection_metrics(predicted_coco_mm)
        accel_p95 = self._acceleration_p95_mm(predicted_coco_mm[:, self.coco_indices])
        fitted_pelvis = 0.5 * (predicted_coco_mm[:, 11] + predicted_coco_mm[:, 12])
        pelvis_fitted = float(np.linalg.norm(fitted_pelvis[-1] - fitted_pelvis[0]))
        pelvis_input = self.pelvis_input
        pelvis_change = float(abs(pelvis_fitted - pelvis_input) / max(pelvis_input, 1e-6))

        body_mm = predicted_coco_mm[:, self.coco_indices]
        nan_joint_count = int(np.count_nonzero(~np.isfinite(body_mm)))
        nan_all17 = int(np.count_nonzero(~np.isfinite(predicted_coco_mm)))
        neg_depth = 0
        for local in range(n):
            left_pts = (self.per_frame_rotation[local].T @ (body_mm[local] - self.per_frame_translation[local]).T).T
            right_pts = (self.cal.R_cam0_to_cam1 @ left_pts.T + self.cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
            neg_depth += int(np.count_nonzero(left_pts[:, 2] <= 0)) + int(np.count_nonzero(right_pts[:, 2] <= 0))

        # Algorithm-output availability, not input-observation availability.
        # A predicted joint is usable only when finite and in front of both
        # cameras.  This makes the baseline comparison sensitive to an actual
        # output degradation instead of returning the same input count for all
        # routes.
        available_joint_count = 0
        for local in range(n):
            left_pts = (
                self.per_frame_rotation[local].T
                @ (body_mm[local] - self.per_frame_translation[local]).T
            ).T
            right_pts = (
                self.cal.R_cam0_to_cam1 @ left_pts.T
                + self.cal.T_cam0_to_cam1_mm.reshape(3, 1)
            ).T
            finite = np.isfinite(body_mm[local]).all(axis=1)
            available_joint_count += int(
                np.count_nonzero(finite & (left_pts[:, 2] > 0) & (right_pts[:, 2] > 0))
            )

        # per-joint acceleration P95 (body 12) and root motion stats.
        accel = body_mm[2:] - 2.0 * body_mm[1:-1] + body_mm[:-2]
        per_joint_accel_p95 = np.percentile(np.linalg.norm(accel, axis=2), 95, axis=0)
        pelvis_accel = 0.5 * (accel[:, 6] + accel[:, 7])
        pelvis_accel_p95 = float(np.percentile(np.linalg.norm(pelvis_accel, axis=1), 95))
        root_orient_np = root_orient.detach().cpu().numpy()
        translation_np = translation.detach().cpu().numpy() * 1000.0
        root_orient_jumps = np.linalg.norm(np.diff(root_orient_np, axis=0), axis=1)
        translation_jumps = np.linalg.norm(np.diff(translation_np, axis=0), axis=1)

        snapshot = {
            "iteration": None,
            "reproj_left_strict_median_px": reproj["left_strict_median"],
            "reproj_left_strict_p95_px": reproj["left_strict_p95"],
            "reproj_right_strict_median_px": reproj["right_strict_median"],
            "reproj_right_strict_p95_px": reproj["right_strict_p95"],
            "reproj_left_relaxed_p95_px": reproj["left_relaxed_p95"],
            "reproj_right_relaxed_p95_px": reproj["right_relaxed_p95"],
            "reproj_left_rejected_p95_px": reproj["left_rejected_p95"],
            "reproj_right_rejected_p95_px": reproj["right_rejected_p95"],
            "accel_p95_mm_per_frame2": accel_p95,
            "pelvis_accel_p95_mm_per_frame2": pelvis_accel_p95,
            "per_joint_accel_p95_mm_per_frame2": per_joint_accel_p95.tolist(),
            "root_orient_jump_p95_rad": float(np.percentile(root_orient_jumps, 95)) if len(root_orient_jumps) else None,
            "root_orient_jump_max_rad": float(np.max(root_orient_jumps)) if len(root_orient_jumps) else None,
            "root_translation_jump_p95_mm": float(np.percentile(translation_jumps, 95)) if len(translation_jumps) else None,
            "root_translation_jump_max_mm": float(np.max(translation_jumps)) if len(translation_jumps) else None,
            "pelvis_net_displacement_input_mm": pelvis_input,
            "pelvis_net_displacement_fitted_mm": pelvis_fitted,
            "pelvis_change_fraction": pelvis_change,
            "nan_joint_count": nan_joint_count,
            "nan_joint_count_all17": nan_all17,
            "negative_depth_count": neg_depth,
            "available_joint_count": available_joint_count,
            "r3_high_quality_count": self.r3_audit["high_quality_count"],
            "r3_zero_weight_count": self.r3_audit["zero_weight_count"],
        }
        return snapshot

    def _reprojection_metrics(self, predicted_coco_mm: np.ndarray) -> dict:
        n = self.window_size
        coco = self.coco_indices
        err_l = np.full((n, len(coco)), np.nan)
        err_r = np.full_like(err_l, np.nan)
        for local in range(n):
            left_pts = (self.per_frame_rotation[local].T @ (predicted_coco_mm[local, coco] - self.per_frame_translation[local]).T).T
            right_pts = (self.cal.R_cam0_to_cam1 @ left_pts.T + self.cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
            proj_l = fisheye_project_numpy(left_pts, self.cal.K0, self.cal.D0)
            proj_r = fisheye_project_numpy(right_pts, self.cal.K1, self.cal.D1)
            err_l[local] = np.linalg.norm(proj_l - self.obs2d["kp_left_px"][local], axis=1)
            err_r[local] = np.linalg.norm(proj_r - self.obs2d["kp_right_px"][local], axis=1)
        reasons_l = self.obs2d["reasons_left"]
        reasons_r = self.obs2d["reasons_right"]
        strict_l = reasons_l == "accepted_strict"
        relaxed_l = reasons_l == "accepted_relaxed"
        strict_r = reasons_r == "accepted_strict"
        relaxed_r = reasons_r == "accepted_relaxed"
        rejected_l = ~(strict_l | relaxed_l)
        rejected_r = ~(strict_r | relaxed_r)
        return {
            "left_strict_median": float(np.nanmedian(err_l[strict_l])) if np.any(strict_l) else None,
            "left_strict_p95": float(np.nanpercentile(err_l[strict_l], 95)) if np.any(strict_l) else None,
            "right_strict_median": float(np.nanmedian(err_r[strict_r])) if np.any(strict_r) else None,
            "right_strict_p95": float(np.nanpercentile(err_r[strict_r], 95)) if np.any(strict_r) else None,
            "left_relaxed_p95": float(np.nanpercentile(err_l[relaxed_l], 95)) if np.any(relaxed_l) else None,
            "right_relaxed_p95": float(np.nanpercentile(err_r[relaxed_r], 95)) if np.any(relaxed_r) else None,
            "left_rejected_p95": float(np.nanpercentile(err_l[rejected_l], 95)) if np.any(rejected_l) else None,
            "right_rejected_p95": float(np.nanpercentile(err_r[rejected_r], 95)) if np.any(rejected_r) else None,
        }


def torch_as(torch, betas: np.ndarray, window_size: int, device) -> "Any":
    return torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(1, 10).expand(window_size, -1)


def run_r3_init(env: FitEnv, root_orient, translation, latent, masks) -> None:
    torch = env.torch
    parameters = [root_orient, translation, latent]
    optimizer = torch.optim.Adam(parameters, lr=R3_INIT_LEARNING_RATE)
    for _ in range(R3_INIT_ITERATIONS):
        optimizer.zero_grad()
        _, _, predicted_coco_body, _ = env.forward(root_orient, translation, latent)
        loss = r3_soft_loss_torch(predicted_coco_body, env.r3_target_m, masks["r3_w"])
        loss.backward()
        optimizer.step()


@dataclass
class RunResult:
    config: FitRouteConfig
    window: tuple[int, int]
    run_type: str  # convergence | timed | holdout | validation
    repeat: int
    warmup: bool
    init_seconds: float
    formal_seconds: float
    end_to_end_seconds: float
    snapshots: list[dict] = field(default_factory=list)  # convergence only
    final_metrics: dict = field(default_factory=dict)
    holdout_metrics: dict = field(default_factory=dict)
    per_frame_rows: list[dict] = field(default_factory=list)
    rejection_rows: list[dict] = field(default_factory=list)
    checkpoint_states: dict[int, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)


def capture_parameter_state(root_orient, translation, latent) -> dict[str, np.ndarray]:
    """Deep-copy one exact optimizer state point for metric/export identity."""
    return {
        "root_orient": root_orient.detach().cpu().numpy().copy(),
        "translation": translation.detach().cpu().numpy().copy(),
        "latent": latent.detach().cpu().numpy().copy(),
    }


def parameters_from_state(env: FitEnv, state: dict[str, np.ndarray]):
    torch = env.torch
    return tuple(
        torch.nn.Parameter(torch.tensor(state[name].copy(), dtype=torch.float32, device=env.device))
        for name in ("root_orient", "translation", "latent")
    )


def materialize_checkpoint_result(
    env: FitEnv,
    source: RunResult,
    iteration: int,
    hold_out: set[int] | None = None,
) -> RunResult:
    """Export exactly the state that produced a recorded checkpoint metric."""
    hold_out = set() if hold_out is None else set(hold_out)
    if iteration not in source.checkpoint_states:
        raise KeyError(f"checkpoint state {iteration} was not captured")
    root_orient, translation, latent = parameters_from_state(env, source.checkpoint_states[iteration])
    masks = env.masks_with_holdout(hold_out)
    final = env.snapshot(root_orient, translation, latent, masks, source.config, hold_out)
    final["iteration"] = int(iteration)
    terms = env.loss_terms(root_orient, translation, latent, masks, source.config)
    final["loss_terms"] = env.terms_to_float(terms)
    final["loss_total"] = float(env.total_loss(terms, source.config).detach().cpu())
    result = RunResult(
        config=source.config, window=source.window, run_type="selected_checkpoint",
        repeat=source.repeat, warmup=source.warmup, init_seconds=source.init_seconds,
        formal_seconds=source.formal_seconds, end_to_end_seconds=source.end_to_end_seconds,
        final_metrics=final,
    )
    result.per_frame_rows = build_per_frame_rows(
        env, source.config, root_orient, translation, latent, hold_out,
        result.run_type, result.repeat, result.warmup,
    )
    result.rejection_rows = build_rejection_rows(
        env, source.config, hold_out, result.run_type, result.repeat, result.warmup
    )
    return result


def run_fit(
    env: FitEnv,
    config: FitRouteConfig,
    hold_out: set[int],
    checkpoints: tuple[int, ...],
    run_type: str,
    repeat: int = 0,
    warmup: bool = False,
    iterations: int = ITERATIONS,
) -> RunResult:
    torch = env.torch
    root_orient, translation, latent = env.make_params(hold_out)
    masks = env.masks_with_holdout(hold_out)
    result = RunResult(config=config, window=(int(env.indices[0]), int(env.indices[-1])),
                       run_type=run_type, repeat=repeat, warmup=warmup,
                       init_seconds=0.0, formal_seconds=0.0, end_to_end_seconds=0.0)

    started = time.perf_counter()
    if config.init == "r3":
        run_r3_init(env, root_orient, translation, latent, masks)
    init_seconds = time.perf_counter() - started

    # Iteration 0 means "after the selected initialization, before any formal
    # optimization step".  The previous implementation accidentally captured
    # it after all 400 steps, invalidating time-to-baseline comparisons.
    checkpoint_set = set(checkpoints)
    if 0 in checkpoint_set:
        terms0 = env.loss_terms(root_orient, translation, latent, masks, config)
        snap0 = env.snapshot(root_orient, translation, latent, masks, config, hold_out)
        snap0["iteration"] = 0
        snap0["loss_terms"] = env.terms_to_float(terms0)
        snap0["loss_total"] = float(env.total_loss(terms0, config).detach().cpu())
        result.snapshots.append(snap0)
        result.checkpoint_states[0] = capture_parameter_state(root_orient, translation, latent)

    optimizer = torch.optim.Adam([root_orient, translation, latent], lr=LEARNING_RATE)
    formal_started = time.perf_counter()
    for step in range(1, iterations + 1):
        optimizer.zero_grad()
        terms = env.loss_terms(root_orient, translation, latent, masks, config)
        loss = env.total_loss(terms, config)
        loss.backward()
        optimizer.step()
        if step in checkpoint_set:
            checkpoint_terms = env.loss_terms(root_orient, translation, latent, masks, config)
            snap = env.snapshot(root_orient, translation, latent, masks, config, hold_out)
            snap["iteration"] = step
            snap["loss_terms"] = env.terms_to_float(checkpoint_terms)
            snap["loss_total"] = float(
                env.total_loss(checkpoint_terms, config).detach().cpu()
            )
            result.snapshots.append(snap)
            result.checkpoint_states[int(step)] = capture_parameter_state(root_orient, translation, latent)
    formal_seconds = time.perf_counter() - formal_started
    result.snapshots.sort(key=lambda item: item["iteration"])

    # Final state (iterations reached).
    final = env.snapshot(root_orient, translation, latent, masks, config, hold_out)
    final["iteration"] = iterations
    final_terms = env.loss_terms(root_orient, translation, latent, masks, config)
    final["loss_terms"] = env.terms_to_float(final_terms)
    final["loss_total"] = float(env.total_loss(final_terms, config).detach().cpu())
    result.final_metrics = final
    result.init_seconds = init_seconds
    result.formal_seconds = formal_seconds
    result.end_to_end_seconds = init_seconds + formal_seconds

    if run_type != "timed":
        result.per_frame_rows = build_per_frame_rows(env, config, root_orient, translation, latent, hold_out, run_type, repeat, warmup)
        result.rejection_rows = build_rejection_rows(env, config, hold_out, run_type, repeat, warmup)
    if hold_out:
        result.holdout_metrics = build_holdout_metrics(env, config, root_orient, translation, latent)
    return result


def fitted_joints_numpy(env: FitEnv, root_orient, translation, latent) -> np.ndarray:
    torch = env.torch
    with torch.no_grad():
        _, predicted_coco, _, _ = env.forward(root_orient, translation, latent)
    return predicted_coco.cpu().numpy() * 1000.0  # (N, 17, 3) world mm


def window_label(window: tuple[int, int]) -> str:
    return f"{int(window[0]):04d}_{int(window[1]):04d}"


def build_per_frame_rows(env: FitEnv, config: FitRouteConfig, root_orient, translation, latent,
                         hold_out: set[int], run_type: str, repeat: int, warmup: bool) -> list[dict]:
    predicted_mm = fitted_joints_numpy(env, root_orient, translation, latent)
    rows = []
    n = env.window_size
    coco = env.coco_indices
    err_l = np.full((n, 17), np.nan)
    err_r = np.full((n, 17), np.nan)
    depth_l = np.full((n, 17), np.nan)
    depth_r = np.full((n, 17), np.nan)
    for local in range(n):
        left_pts_all = (env.per_frame_rotation[local].T @ (predicted_mm[local] - env.per_frame_translation[local]).T).T
        right_pts_all = (env.cal.R_cam0_to_cam1 @ left_pts_all.T + env.cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
        proj_l_all = fisheye_project_numpy(left_pts_all, env.cal.K0, env.cal.D0)
        proj_r_all = fisheye_project_numpy(right_pts_all, env.cal.K1, env.cal.D1)
        err_l[local, coco] = np.linalg.norm(proj_l_all[coco] - env.obs2d["kp_left_px"][local], axis=1)
        err_r[local, coco] = np.linalg.norm(proj_r_all[coco] - env.obs2d["kp_right_px"][local], axis=1)
        depth_l[local] = left_pts_all[:, 2]
        depth_r[local] = right_pts_all[:, 2]
    for local, frame_index in enumerate(env.indices):
        record = env.r3_records.get(int(frame_index))
        r3_mask = r3_quality_mask(record) if record is not None else np.zeros(17, dtype=bool)
        frame_row = env.frames[frame_index]
        accepted_left_all = np.asarray(env.obs2d_all["accepted_left"][local], dtype=bool)
        accepted_right_all = np.asarray(env.obs2d_all["accepted_right"][local], dtype=bool)
        input_observed = accepted_left_all | accepted_right_all
        display_valid = np.asarray([
            bool(
                np.all(np.isfinite(predicted_mm[local, j]))
                and j not in hold_out
                and (j >= 5 or (config.include_head_observations and input_observed[j]))
            )
            for j in range(17)
        ], dtype=bool)
        rows.append({
            "frame_index": int(frame_index),
            "pair_id": int(frame_row["pair"]),
            "stage": frame_row["stage"],
            "route": config.name,
            "config": config_subdir(config),
            "window": f"{int(env.indices[0]):04d}_{int(env.indices[-1]):04d}",
            "run_type": run_type,
            "repeat": repeat,
            "warmup": warmup,
            "smplx_coco17_joints_ground_mm": np.round(predicted_mm[local], 3).tolist(),
            "reprojection_left_px": np.round(np.nan_to_num(err_l[local], nan=-1.0), 3).tolist(),
            "reprojection_right_px": np.round(np.nan_to_num(err_r[local], nan=-1.0), 3).tolist(),
            "depth_left_mm": np.round(np.nan_to_num(depth_l[local], nan=-1.0), 3).tolist(),
            "depth_right_mm": np.round(np.nan_to_num(depth_r[local], nan=-1.0), 3).tolist(),
            "r3_high_quality": r3_mask.tolist(),
            "r3_rejection_reasons": [r3_rejection_reasons(record, j) if record is not None else ["r3_frame_missing"] for j in range(17)],
            "hold_out_joints": sorted(int(j) for j in hold_out),
            "finite": [bool(np.all(np.isfinite(predicted_mm[local, j]))) for j in range(17)],
            # Explicit export contract.  A head without an accepted observation
            # is not represented as a measured, precise output.
            "point_space": "predicted_coco",
            "joint_definition": "COCO-17 observation points",
            "offset_model_used": True,
            "source": ["smplx_offset_model" for _ in range(17)],
            "valid": display_valid.tolist(),
            "observation_state": [
                {
                    "left_accepted": bool(accepted_left_all[j]),
                    "right_accepted": bool(accepted_right_all[j]),
                    "any_accepted": bool(input_observed[j]),
                    "used_by_route": bool(j >= 5 or config.include_head_observations),
                    "held_out": bool(j in hold_out),
                }
                for j in range(17)
            ],
            "rejection_reason": [
                ([] if display_valid[j] else
                 (["held_out"] if j in hold_out else
                  (["head_observations_disabled"] if j < 5 and not config.include_head_observations else
                   (["head_observation_unavailable_fail_closed"] if j < 5 else
                    ["non_finite_model_prediction"]))))
                for j in range(17)
            ],
        })
    return rows


def build_rejection_rows(env: FitEnv, config: FitRouteConfig, hold_out: set[int],
                         run_type: str, repeat: int, warmup: bool) -> list[dict]:
    rows = []
    reasons_l = env.obs2d["reasons_left"]
    reasons_r = env.obs2d["reasons_right"]
    for local, frame_index in enumerate(env.indices):
        record = env.r3_records.get(int(frame_index))
        for position, coco_joint in enumerate(BODY_COCO_INDICES):
            for view, reasons in (("left", reasons_l), ("right", reasons_r)):
                reason = reasons[local, position]
                if reason not in ("accepted_strict", "accepted_relaxed"):
                    rows.append({
                        "frame_index": int(frame_index),
                        "pair_id": int(env.frames[frame_index]["pair"]),
                        "joint_index": int(coco_joint),
                        "joint_name": JOINT_NAMES.get(int(coco_joint)),
                        "scope": f"2d_{view}",
                        "route": config.name,
                        "config": config_subdir(config),
                        "window": f"{int(env.indices[0]):04d}_{int(env.indices[-1]):04d}",
                        "run_type": run_type,
                        "repeat": repeat,
                        "warmup": warmup,
                        "reasons": [reason],
                    })
            if record is not None and not bool(r3_quality_mask(record)[int(coco_joint)]):
                rows.append({
                    "frame_index": int(frame_index),
                    "pair_id": int(env.frames[frame_index]["pair"]),
                    "joint_index": int(coco_joint),
                    "joint_name": JOINT_NAMES.get(int(coco_joint)),
                    "scope": "r3",
                    "route": config.name,
                    "config": config_subdir(config),
                    "window": f"{int(env.indices[0]):04d}_{int(env.indices[-1]):04d}",
                    "run_type": run_type,
                    "repeat": repeat,
                    "warmup": warmup,
                    "reasons": r3_rejection_reasons(record, int(coco_joint)) or ["not_high_quality"],
                })
            elif record is None:
                rows.append({
                    "frame_index": int(frame_index),
                    "pair_id": int(env.frames[frame_index]["pair"]),
                    "joint_index": int(coco_joint),
                    "joint_name": JOINT_NAMES.get(int(coco_joint)),
                    "scope": "r3",
                    "route": config.name,
                    "config": config_subdir(config),
                    "window": f"{int(env.indices[0]):04d}_{int(env.indices[-1]):04d}",
                    "run_type": run_type,
                    "repeat": repeat,
                    "warmup": warmup,
                    "reasons": ["r3_frame_missing"],
                })
    return rows


def build_holdout_metrics(env: FitEnv, config: FitRouteConfig, root_orient, translation, latent) -> dict:
    """Occlusion-joint audit: 3D deviation, recovery, depth consistency, L/R holdout reprojection."""
    predicted_mm = fitted_joints_numpy(env, root_orient, translation, latent)
    n = env.window_size
    prefit_mm = env.full_prefit_mm[env.indices]
    motion_mm = env.full_motion_mm[env.indices]
    out: dict = {"joints": {}}
    for coco_joint in HOLDOUT_JOINTS:
        position = list(BODY_COCO_INDICES).index(coco_joint)
        dev = np.linalg.norm(predicted_mm[:, coco_joint] - prefit_mm[:, coco_joint], axis=1)
        dev_finite = dev[np.isfinite(prefit_mm[:, coco_joint]).all(axis=1)]
        fitted_finite = np.all(np.isfinite(predicted_mm[:, coco_joint]), axis=1)
        depth_ok = np.ones(n, dtype=bool)
        for local in range(n):
            left_pts = env.per_frame_rotation[local].T @ (predicted_mm[local, coco_joint] - env.per_frame_translation[local])
            right_pts = env.cal.R_cam0_to_cam1 @ left_pts + env.cal.T_cam0_to_cam1_mm
            depth_ok[local] = bool(left_pts[2] > 0 and right_pts[2] > 0)
        recovery = float(np.mean(fitted_finite & depth_ok)) if n else 0.0

        depth_diff = np.abs(predicted_mm[:, coco_joint, 2] - motion_mm[:, coco_joint, 2])
        # Internal-depth reference is restricted to strict triangulations that
        # were not used by this hold-out fit.  Finite relaxed points are not
        # silently promoted to strict evidence.
        motion_finite = np.all(np.isfinite(motion_mm[:, coco_joint]), axis=1)
        strict_reference = motion_finite & env.strict[env.indices, coco_joint]
        depth_vals = depth_diff[strict_reference]

        err_l = np.full(n, np.nan)
        err_r = np.full(n, np.nan)
        accepted_l = env.obs2d["accepted_left"][:, position]
        accepted_r = env.obs2d["accepted_right"][:, position]
        for local in range(n):
            left_pts = env.per_frame_rotation[local].T @ (predicted_mm[local, coco_joint] - env.per_frame_translation[local])
            right_pts = env.cal.R_cam0_to_cam1 @ left_pts + env.cal.T_cam0_to_cam1_mm
            if accepted_l[local]:
                err_l[local] = float(np.linalg.norm(fisheye_project_numpy(left_pts[None], env.cal.K0, env.cal.D0)[0] - env.obs2d["kp_left_px"][local, position]))
            if accepted_r[local]:
                err_r[local] = float(np.linalg.norm(fisheye_project_numpy(right_pts[None], env.cal.K1, env.cal.D1)[0] - env.obs2d["kp_right_px"][local, position]))
        out["joints"][str(coco_joint)] = {
            "joint_index": int(coco_joint),
            "joint_name": JOINT_NAMES[int(coco_joint)],
            "deviation_median_mm": float(np.median(dev_finite)) if len(dev_finite) else None,
            "deviation_p95_mm": float(np.percentile(dev_finite, 95)) if len(dev_finite) else None,
            "recovery_available_fraction": recovery,
            "internal_depth_consistency_median_mm": float(np.median(depth_vals)) if len(depth_vals) else None,
            "internal_depth_consistency_p95_mm": float(np.percentile(depth_vals, 95)) if len(depth_vals) else None,
            "reprojection_left_median_px": float(np.nanmedian(err_l[accepted_l])) if np.any(accepted_l) else None,
            "reprojection_right_median_px": float(np.nanmedian(err_r[accepted_r])) if np.any(accepted_r) else None,
        }
    # root depth drift.
    fitted_pelvis = 0.5 * (predicted_mm[:, 11] + predicted_mm[:, 12])
    prefit_pelvis = 0.5 * (prefit_mm[:, 11] + prefit_mm[:, 12])
    out["root_depth_drift_input_mm"] = float(abs(prefit_pelvis[-1, 2] - prefit_pelvis[0, 2]))
    out["root_depth_drift_fitted_mm"] = float(abs(fitted_pelvis[-1, 2] - fitted_pelvis[0, 2]))
    return out


# --------------------------------------------------------------------------- gates

def baseline_target_checks(snapshot: dict, baseline: dict) -> dict:
    return {
        "strict_L_p95_le_base_plus_1px": (
            snapshot["reproj_left_strict_p95_px"] is not None
            and baseline["reproj_left_strict_p95_px"] is not None
            and snapshot["reproj_left_strict_p95_px"] <= baseline["reproj_left_strict_p95_px"] + 1.0
        ),
        "strict_R_p95_le_base_plus_1px": (
            snapshot["reproj_right_strict_p95_px"] is not None
            and baseline["reproj_right_strict_p95_px"] is not None
            and snapshot["reproj_right_strict_p95_px"] <= baseline["reproj_right_strict_p95_px"] + 1.0
        ),
        "accel_p95_le_base_x1_05": snapshot["accel_p95_mm_per_frame2"] <= baseline["accel_p95_mm_per_frame2"] * 1.05,
        "pelvis_change_le_base_plus_1pp": snapshot["pelvis_change_fraction"] <= baseline["pelvis_change_fraction"] + 0.01,
        "nan_not_worse": snapshot["nan_joint_count"] <= baseline["nan_joint_count"],
        "negative_depth_not_worse": snapshot["negative_depth_count"] <= baseline["negative_depth_count"],
        "available_joints_not_worse": snapshot["available_joint_count"] >= baseline["available_joint_count"],
    }


def dev_gates(snapshot: dict, env: FitEnv) -> dict:
    return {
        "nan_joint_count_zero": snapshot["nan_joint_count"] == 0,
        "negative_depth_count_zero": snapshot["negative_depth_count"] == 0,
        "strict_L_P95_at_most_50_px": snapshot["reproj_left_strict_p95_px"] is not None and snapshot["reproj_left_strict_p95_px"] <= 50.0,
        "strict_R_P95_at_most_50_px": snapshot["reproj_right_strict_p95_px"] is not None and snapshot["reproj_right_strict_p95_px"] <= 50.0,
        "accel_at_most_1_2x_input": snapshot["accel_p95_mm_per_frame2"] <= env.input_accel_p95 * 1.2 + 1e-9,
        "pelvis_change_at_most_25_percent": snapshot["pelvis_change_fraction"] <= 0.25,
    }


# --------------------------------------------------------------------------- persistence

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_summary(config: FitRouteConfig, window: tuple[int, int], result: RunResult,
                  env: FitEnv, hold_out: set[int], timing_stats: dict | None,
                  time_to_baseline: str | None, baseline_checks: dict | None,
                  dev_passed: bool | None, diagnostic_only: bool, experiment_dir: Path,
                  subdir: str | None = None) -> None:
    if subdir is None:
        config_dir = experiment_dir / "routes" / config_subdir(config)
    else:
        config_dir = experiment_dir / subdir / window_label(window) / config_subdir(config)
    config_dir.mkdir(parents=True, exist_ok=True)
    gates = dev_gates(result.final_metrics, env)
    summary = {
        "schema_version": "smplx_r3_integration_fit_v1",
        "route": config.name,
        "config": config_subdir(config),
        "init": config.init,
        "data3d": config.data3d,
        "r3soft_weight": config.r3soft_weight,
        "inclusive_frame_window": [int(window[0]), int(window[1])],
        "pair_window": [int(env.pair_ids[0]), int(env.pair_ids[-1])],
        "frame_count": env.window_size,
        "iterations": ITERATIONS,
        "device": str(env.device),
        "shape_mode": env.shape_mode,
        "hold_out_joints": sorted(int(j) for j in hold_out),
        "weights": {**WEIGHTS, "r3soft": config.r3soft_weight},
        "r3_init_budget": {"iterations": R3_INIT_ITERATIONS, "learning_rate": R3_INIT_LEARNING_RATE},
        "r3_audit": env.r3_audit,
        "timing": {
            "init_seconds": result.init_seconds,
            "formal_seconds": result.formal_seconds,
            "end_to_end_seconds": result.end_to_end_seconds,
            "timed_runs_stats": timing_stats,
            "warmup": result.warmup,
        },
        "vposer_checkpoint": env.vposer_checkpoint,
        "final_metrics": result.final_metrics,
        "gates": gates,
        "gates_passed": bool(all(gates.values())),
        "baseline_quality": {
            "time_to_baseline_quality": time_to_baseline,
            "checks": baseline_checks,
        },
        "dev_passed": dev_passed,
        "diagnostic_only": diagnostic_only,
        "holdout_metrics": result.holdout_metrics,
        "boundary": (
            "Offline bounded-window engineering comparison on frozen people_1 keypoints. "
            "Reprojection and internal depth consistency are NOT physical 3D accuracy. "
            "Stage 2 feet participate in the camera pose estimate, so foot metrics are not "
            "independent. The old C1000 and its repair results are evidence-different runs "
            "and are never referenced as this experiment's outcome."
        ),
    }
    (config_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(config_dir / "per_frame_metrics.jsonl", result.per_frame_rows)
    write_jsonl(config_dir / "rejection_records.jsonl", result.rejection_rows)
    (config_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")


def timing_stats_for(timed: list[RunResult]) -> dict:
    if not timed:
        return {"runs": 0}
    values = {
        key: sorted([getattr(run, key) for run in timed])
        for key in ("init_seconds", "formal_seconds", "end_to_end_seconds")
    }
    def p50_p95(vals: list[float]) -> dict:
        return {"p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
                "min": float(min(vals)), "max": float(max(vals))}
    return {"runs": len(timed), **{key: p50_p95(values[key]) for key in values}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path(r"D:\my_works\walker_pose_system\models"))
    parser.add_argument("--vposer-dir", type=Path, default=Path(r"D:\my_works\walker_pose_system\models\VPoser02_05\V02_05"))
    parser.add_argument("--motion-json", type=Path, default=Path(r"D:\my_works\walker_pose_system\docs\walker-viewer\data\motion.json"))
    parser.add_argument("--prefit-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_temporal_body_model_prefit_v1\fixed_bone_contact_prefit_v2_gated\temporal_body_prefit.jsonl"))
    parser.add_argument("--interaction-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_offline_walker_frame_evidence_v1\coarse_complete_model_visual_v3_camera_pair\interaction_distance_records.jsonl"))
    parser.add_argument("--initial-fit-summary", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_smplx_temporal_fit_v1\single_frame_quality_selected_v2\summary.json"))
    parser.add_argument("--stereo-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\pmpose_strict_stereo\offline_stereo_results.jsonl"))
    parser.add_argument("--calibration-dir", type=Path, default=Path(r"D:\my_works\walker_pose_system\realtime_app\calibration\results"))
    parser.add_argument("--offset-audit-json", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_smplx_shape_offset_gate_v1\final_shape_offset_audit_v3.json"))
    parser.add_argument("--fixed-betas-json", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_smplx_shape_offset_gate_v1\final_shape_offset_audit_v3.json"))
    parser.add_argument("--r3-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260919_alternative_body_routes_final_v8\R3_kalman_fixed_bone_ik\config_IK3_R030\per_frame_metrics.jsonl"))
    parser.add_argument("--image-left-dir", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\input_448pairs\left_ccw90"))
    parser.add_argument("--image-right-dir", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\input_448pairs\right_cw90"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("auto", "dev", "validate", "render", "aggregate"), default="auto")
    parser.add_argument("--force", action="store_true", help="re-run configs even if output exists")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    # A completed reproducible run is expensive (hours on CPU).  The protocol's
    # verification command intentionally omits --force; treat it as an
    # integrity/aggregation replay when every required final artifact exists.
    # Fresh or incomplete directories still execute the requested experiment.
    completed_required = (
        "run_metadata.json",
        "convergence.json",
        "development_comparison.json",
        "validation_comparison.json",
        "frozen_config.json",
        "per_frame_metrics.jsonl",
        "rejection_records.jsonl",
        "timing.json",
    )
    completed_run = False
    metadata_path = experiment_dir / "run_metadata.json"
    if args.mode == "auto" and not args.force and all(
        (experiment_dir / name).exists() for name in completed_required
    ):
        try:
            completed_run = (
                json.loads(metadata_path.read_text(encoding="utf-8")).get("status")
                == "completed"
            )
        except (OSError, ValueError):
            completed_run = False
    if completed_run:
        aggregate(experiment_dir, args)
        print("completed run already present; verified and re-aggregated (use --force to rerun)")
        return 0

    started = time.time()

    if args.mode in ("auto", "dev"):
        run_development(experiment_dir, args)
    if args.mode in ("auto", "validate"):
        run_validation(experiment_dir, args)
    if args.mode in ("auto", "render"):
        run_render(experiment_dir, args)
    if args.mode in ("auto", "aggregate", "dev", "validate", "render"):
        aggregate(experiment_dir, args)

    import platform
    previous_metadata = {}
    if metadata_path.exists():
        try:
            previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_metadata = {}
    audit_dir = experiment_dir / "audit_frames"
    frozen_config_name = None
    frozen_config_path = experiment_dir / "frozen_config.json"
    if frozen_config_path.exists():
        try:
            frozen_config_name = json.loads(
                frozen_config_path.read_text(encoding="utf-8")
            ).get("frozen_config")
        except (OSError, ValueError):
            frozen_config_name = None
    holdout_configs = (ROUTE_C_PREFIT, ROUTE_C_R3_INIT, ROUTE_B_R3_INIT, frozen_config_name)
    stage_completion = {
        "development_complete": (experiment_dir / "development_comparison.json").exists(),
        "holdout_complete": frozen_config_name is not None and all(
            (experiment_dir / "routes" / name / "holdout_metrics.json").exists()
            for name in holdout_configs
        ),
        "validation_complete": (experiment_dir / "validation_comparison.json").exists(),
        "render_complete": audit_dir.is_dir() and len(list(audit_dir.glob("*.png"))) >= 12,
        "aggregate_complete": (
            (experiment_dir / "per_frame_metrics.jsonl").exists()
            and (experiment_dir / "rejection_records.jsonl").exists()
        ),
    }
    metadata = {
        "schema_version": "smplx_r3_integration_run_v1",
        "status": "completed" if all(stage_completion.values()) else "partial",
        "mode": args.mode,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "elapsed_seconds_by_last_mode": {
            **previous_metadata.get("elapsed_seconds_by_last_mode", {}),
            args.mode: time.time() - started,
        },
        **stage_completion,
        "inputs": {
            "model_root": str(args.model_root),
            "vposer_dir": str(args.vposer_dir),
            "motion_json": str(args.motion_json),
            "prefit_jsonl": str(args.prefit_jsonl),
            "interaction_jsonl": str(args.interaction_jsonl),
            "initial_fit_summary": str(args.initial_fit_summary),
            "stereo_jsonl": str(args.stereo_jsonl),
            "calibration_dir": str(args.calibration_dir),
            "offset_audit_json": str(args.offset_audit_json),
            "fixed_betas_json": str(args.fixed_betas_json),
            "r3_jsonl": str(args.r3_jsonl),
            "image_left_dir": str(args.image_left_dir),
            "image_right_dir": str(args.image_right_dir),
        },
        "window_definitions": {
            "dev": [DEV_WINDOW[0], DEV_WINDOW[1]],
            "validation": [list(window) for window in VALIDATION_WINDOWS],
        },
        "route_definitions": {
            ROUTE_C_PREFIT: "prefit init; L/R 2D + prefit soft 3D + temporal + priors + contacts (re-run baseline)",
            ROUTE_C_R3_INIT: "R3 high-quality init; formal loss identical to C-prefit, no R3 term",
            ROUTE_B_R3_INIT: "R3 init; all 3D data terms cancelled (2D + temporal + priors + contacts only)",
            ROUTE_C_R3_SOFT: f"R3 init; 2D + prefit soft 3D + R3 soft 3D (weight grid {list(R3_SOFT_WEIGHT_GRID)})",
        },
        "hyperparameters": {
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "checkpoints": list(CHECKPOINTS),
            "timed_runs": TIMED_RUNS,
            "r3_init_iterations": R3_INIT_ITERATIONS,
            "r3_init_learning_rate": R3_INIT_LEARNING_RATE,
            "weights": WEIGHTS,
            "gm_scale_px": GM_SCALE_PX,
            "gm_scale_m": GM_SCALE_M,
        },
        "boundary": (
            "Offline bounded-window engineering comparison; no external 3D truth. "
            "Reprojection and internal depth consistency are not physical accuracy. "
            "Stage 2 feet participate in camera pose; foot metrics are not independent. "
            "Old C1000 and repair results are not referenced as this experiment's outcome."
        ),
    }
    (experiment_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("done.")
    return 0


def run_development(experiment_dir: Path, args: argparse.Namespace) -> None:
    env = FitEnv(args, DEV_WINDOW[0], DEV_WINDOW[1], device=args.device)
    print(f"[dev] window {DEV_WINDOW[0]}-{DEV_WINDOW[1]}, {env.window_size} frames; "
          f"input accel P95 {env.input_accel_p95:.2f} mm/f2; pelvis input {env.pelvis_input:.2f} mm")
    configs = route_configs()
    results: dict[str, dict] = {}

    for config in configs:
        name = config_subdir(config)
        out_dir = experiment_dir / "routes" / name
        if out_dir.exists() and not args.force:
            print(f"[dev] {name}: output exists, skipping (--force to rerun)")
            continue
        print(f"[dev] {name}: convergence run (warmup) + {TIMED_RUNS} timed runs ...", flush=True)
        convergence = run_fit(env, config, set(), CHECKPOINTS, run_type="convergence", repeat=0, warmup=True)
        timed: list[RunResult] = []
        for repeat in range(1, TIMED_RUNS + 1):
            timed.append(run_fit(env, config, set(), (), run_type="timed", repeat=repeat, warmup=False))
        stats = timing_stats_for(timed)
        results[name] = {"config": config, "convergence": convergence, "timed": timed, "stats": stats}
        write_summary(config, DEV_WINDOW, convergence, env, set(), stats,
                      time_to_baseline=None, baseline_checks=None,
                      dev_passed=None, diagnostic_only=False, experiment_dir=experiment_dir)
        # final summary including baseline analysis is written after the baseline is known.

    # Baseline: C-prefit convergence @400.
    base_name = config_subdir(FitRouteConfig(ROUTE_C_PREFIT, "prefit", True, None))
    if base_name not in results:
        raise RuntimeError("C-prefit convergence result missing; cannot define baseline quality target")
    baseline = results[base_name]["convergence"].final_metrics

    # Baseline analysis for every config.
    for name, entry in results.items():
        config = entry["config"]
        convergence = entry["convergence"]
        time_to_baseline = None
        checks_at = None
        for snapshot in convergence.snapshots:
            checks = baseline_target_checks(snapshot, baseline)
            if all(checks.values()):
                time_to_baseline = snapshot["iteration"]
                checks_at = checks
                break
        if time_to_baseline is None:
            time_to_baseline = "not_reached"
            checks_at = baseline_target_checks(convergence.final_metrics, baseline)
        dev_passed = time_to_baseline != "not_reached"
        final_quality_not_degraded = bool(all(baseline_target_checks(convergence.final_metrics, baseline).values()))
        entry["time_to_baseline"] = time_to_baseline
        entry["final_quality_not_degraded"] = final_quality_not_degraded
        entry["baseline_checks"] = checks_at
        entry["dev_passed"] = dev_passed
        write_summary(config, DEV_WINDOW, convergence, env, set(), entry["stats"],
                      time_to_baseline=time_to_baseline, baseline_checks=checks_at,
                      dev_passed=dev_passed, diagnostic_only=False, experiment_dir=experiment_dir)

    # C-R3-soft weight selection (frozen rule; dev window only).
    soft_candidates = []
    for weight in R3_SOFT_WEIGHT_GRID:
        name = config_subdir(FitRouteConfig(f"{ROUTE_C_R3_SOFT}_w{weight:05.2f}", "r3", True, weight))
        if name not in results:
            continue
        final = results[name]["convergence"].final_metrics
        degraded = (
            final["nan_joint_count"] > baseline["nan_joint_count"]
            or final["negative_depth_count"] > baseline["negative_depth_count"]
            or final["available_joint_count"] < baseline["available_joint_count"]
        )
        strict_ok = (
            final["reproj_left_strict_p95_px"] is not None
            and baseline["reproj_left_strict_p95_px"] is not None
            and final["reproj_left_strict_p95_px"] <= baseline["reproj_left_strict_p95_px"] + 1.0
            and final["reproj_right_strict_p95_px"] is not None
            and baseline["reproj_right_strict_p95_px"] is not None
            and final["reproj_right_strict_p95_px"] <= baseline["reproj_right_strict_p95_px"] + 1.0
        )
        soft_candidates.append({
            "weight": weight, "name": name, "final": final,
            "degraded": degraded, "strict_ok": strict_ok,
        })
    eligible = [c for c in soft_candidates if not c["degraded"] and c["strict_ok"]]
    if eligible:
        best = min(eligible, key=lambda c: (c["final"]["accel_p95_mm_per_frame2"], c["weight"]))
        frozen_weight = best["weight"]
        frozen_name = best["name"]
    else:
        frozen_weight = None
        frozen_name = None
        print("[dev] WARNING: no C-R3-soft weight is eligible; nothing frozen", flush=True)
    freeze = {
        "frozen_route": ROUTE_C_R3_SOFT if frozen_name else None,
        "frozen_config": frozen_name,
        "frozen_weight": frozen_weight,
        "selection_rule": (
            "1 exclude NaN/unavailable degradation; 2 require strict L/R P95 <= C-prefit@400 + 1 px; "
            "3 minimise acceleration P95; tie -> lower R3 weight"
        ),
        "candidates": [
            {"weight": c["weight"], "name": c["name"],
             "degraded": c["degraded"], "strict_ok": c["strict_ok"],
             "accel_p95": c["final"]["accel_p95_mm_per_frame2"],
             "strict_L_p95": c["final"]["reproj_left_strict_p95_px"],
             "strict_R_p95": c["final"]["reproj_right_strict_p95_px"]}
            for c in soft_candidates
        ],
    }
    (experiment_dir / "frozen_config.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[dev] frozen C-R3-soft: {frozen_name} (weight {frozen_weight})", flush=True)

    # Holdout audit on the dev window for the four comparison routes.
    holdout_routes = [
        FitRouteConfig(ROUTE_C_PREFIT, "prefit", True, None),
        FitRouteConfig(ROUTE_C_R3_INIT, "r3", True, None),
        FitRouteConfig(ROUTE_B_R3_INIT, "r3", False, None),
    ]
    if frozen_name:
        holdout_routes.append(FitRouteConfig(frozen_name, "r3", True, frozen_weight))
    for config in holdout_routes:
        name = config_subdir(config)
        out_dir = experiment_dir / "routes" / name
        holdout = run_fit(env, config, set(HOLDOUT_JOINTS), (), run_type="holdout", repeat=0, warmup=False)
        write_jsonl(out_dir / "per_frame_metrics_holdout.jsonl", holdout.per_frame_rows)
        write_jsonl(out_dir / "rejection_records_holdout.jsonl", holdout.rejection_rows)
        (out_dir / "holdout_metrics.json").write_text(
            json.dumps(holdout.holdout_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[dev] holdout {name}: {json.dumps(holdout.holdout_metrics)}", flush=True)

    # Development comparison tables.
    rows = []
    for name, entry in results.items():
        config = entry["config"]
        final = entry["convergence"].final_metrics
        gates = dev_gates(final, env)
        rows.append({
            "route": config.name,
            "config": name,
            "r3soft_weight": config.r3soft_weight,
            "init": config.init,
            "data3d": config.data3d,
            "strict_L_median_px": final["reproj_left_strict_median_px"],
            "strict_L_p95_px": final["reproj_left_strict_p95_px"],
            "strict_R_median_px": final["reproj_right_strict_median_px"],
            "strict_R_p95_px": final["reproj_right_strict_p95_px"],
            "accel_p95_mm_per_frame2": final["accel_p95_mm_per_frame2"],
            "pelvis_change_fraction": final["pelvis_change_fraction"],
            "nan_joint_count": final["nan_joint_count"],
            "negative_depth_count": final["negative_depth_count"],
            "available_joint_count": final["available_joint_count"],
            "time_to_baseline_quality": entry["time_to_baseline"],
            "reached_baseline_quality": entry["time_to_baseline"] != "not_reached",
            "final_quality_not_degraded": entry.get("final_quality_not_degraded", False),
            "dev_gates_passed": entry["dev_passed"],
            "absolute_gates_diagnostics": gates,
            "timing_p50_init_s": entry["stats"].get("init_seconds", {}).get("p50"),
            "timing_p50_formal_s": entry["stats"].get("formal_seconds", {}).get("p50"),
            "timing_p50_e2e_s": entry["stats"].get("end_to_end_seconds", {}).get("p50"),
            "timing_p95_e2e_s": entry["stats"].get("end_to_end_seconds", {}).get("p95"),
        })
    write_csv_json(experiment_dir / "development_comparison", rows)

    # Convergence tables.
    conv_rows = []
    for name, entry in results.items():
        config = entry["config"]
        for snapshot in entry["convergence"].snapshots:
            checks = baseline_target_checks(snapshot, baseline)
            conv_rows.append({
                "route": config.name,
                "config": name,
                "r3soft_weight": config.r3soft_weight,
                "checkpoint": snapshot["iteration"],
                "strict_L_p95_px": snapshot["reproj_left_strict_p95_px"],
                "strict_R_p95_px": snapshot["reproj_right_strict_p95_px"],
                "accel_p95_mm_per_frame2": snapshot["accel_p95_mm_per_frame2"],
                "pelvis_change_fraction": snapshot["pelvis_change_fraction"],
                "nan_joint_count": snapshot["nan_joint_count"],
                "negative_depth_count": snapshot["negative_depth_count"],
                "available_joint_count": snapshot["available_joint_count"],
                "reached_baseline": all(checks.values()),
                "loss_total": snapshot["loss_total"],
                "loss_terms": snapshot["loss_terms"],
            })
    write_csv_json(experiment_dir / "convergence", conv_rows)

    # Timing table.
    timing_rows = []
    for name, entry in results.items():
        for run in [entry["convergence"]] + entry["timed"]:
            timing_rows.append({
                "route": entry["config"].name, "config": name,
                "warmup": run.warmup, "repeat": run.repeat,
                "init_seconds": run.init_seconds, "formal_seconds": run.formal_seconds,
                "end_to_end_seconds": run.end_to_end_seconds,
            })
    write_csv_json(experiment_dir / "timing", timing_rows)


def run_validation(experiment_dir: Path, args: argparse.Namespace) -> None:
    freeze = json.loads((experiment_dir / "frozen_config.json").read_text(encoding="utf-8"))
    frozen_weight = freeze["frozen_weight"]
    frozen_name = freeze["frozen_config"]
    routes = [
        FitRouteConfig(ROUTE_C_PREFIT, "prefit", True, None),
        FitRouteConfig(ROUTE_C_R3_INIT, "r3", True, None),
        FitRouteConfig(ROUTE_B_R3_INIT, "r3", False, None),
    ]
    if frozen_weight is not None and frozen_name:
        routes.append(FitRouteConfig(frozen_name, "r3", True, frozen_weight))

    # Dev pass/fail comes from the development comparison.
    dev_rows = read_csv_json(experiment_dir / "development_comparison.json")
    dev_passed_by_config = {row["config"]: bool(row["dev_gates_passed"]) for row in dev_rows}

    rows = []
    for window in VALIDATION_WINDOWS:
        env = FitEnv(args, window[0], window[1], device=args.device)
        print(f"[validate] window {window[0]}-{window[1]}", flush=True)
        for config in routes:
            name = config_subdir(config)
            diagnostic_only = not dev_passed_by_config.get(name, False)
            print(f"[validate] {name} (diagnostic_only={diagnostic_only})", flush=True)
            result = run_fit(env, config, set(), (), run_type="validation", repeat=0, warmup=False)
            gates = dev_gates(result.final_metrics, env)
            write_summary(config, window, result, env, set(), None,
                          time_to_baseline=None, baseline_checks=None,
                          dev_passed=None, diagnostic_only=diagnostic_only, experiment_dir=experiment_dir,
                          subdir="validation")
            final = result.final_metrics
            rows.append({
                "window": f"{window[0]:04d}_{window[1]:04d}",
                "route": config.name, "config": name,
                "r3soft_weight": config.r3soft_weight,
                "diagnostic_only": diagnostic_only,
                "strict_L_median_px": final["reproj_left_strict_median_px"],
                "strict_L_p95_px": final["reproj_left_strict_p95_px"],
                "strict_R_median_px": final["reproj_right_strict_median_px"],
                "strict_R_p95_px": final["reproj_right_strict_p95_px"],
                "accel_p95_mm_per_frame2": final["accel_p95_mm_per_frame2"],
                "pelvis_change_fraction": final["pelvis_change_fraction"],
                "nan_joint_count": final["nan_joint_count"],
                "negative_depth_count": final["negative_depth_count"],
                "available_joint_count": final["available_joint_count"],
                "gates_passed": bool(all(gates.values())),
                "gates": gates,
            })
    write_csv_json(experiment_dir / "validation_comparison", rows)


def run_render(experiment_dir: Path, args: argparse.Namespace) -> None:
    """Render L/R raw-fisheye audit overlays for frames 60/75/90 (solid lines)."""
    from PIL import Image, ImageDraw
    freeze = json.loads((experiment_dir / "frozen_config.json").read_text(encoding="utf-8"))
    frozen_name = freeze["frozen_config"]
    render_configs = [config_subdir(FitRouteConfig(ROUTE_C_PREFIT, "prefit", True, None))]
    if frozen_name:
        render_configs.append(frozen_name)
    cal = load_stereo_fisheye(args.calibration_dir)
    stereo_rows = read_jsonl(args.stereo_jsonl)
    stereo_by_pair = {int(row["pair_id"]): row for row in stereo_rows}
    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    out_root = experiment_dir / "audit_frames"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_w, raw_h = 1920, 1080

    for config_name in render_configs:
        config_dir = experiment_dir / "routes" / config_name
        per_frame = read_jsonl_rows(config_dir / "per_frame_metrics.jsonl")
        by_frame = {row["frame_index"]: row for row in per_frame}
        for frame_index in AUDIT_FRAMES:
            pair_id = int(motion["frames"][frame_index]["pair"])
            left_path = args.image_left_dir / f"pair_{pair_id:04d}.png"
            right_path = args.image_right_dir / f"pair_{pair_id:04d}.png"
            if not left_path.exists() or not right_path.exists():
                print(f"[render] WARN missing images for pair {pair_id}", file=sys.stderr)
                continue
            raw_left = _unrotate_to_raw(Image.open(left_path).convert("RGB"), "left").resize((raw_w, raw_h))
            raw_right = _unrotate_to_raw(Image.open(right_path).convert("RGB"), "right").resize((raw_w, raw_h))
            row = stereo_by_pair.get(pair_id)
            observed = {view: {} for view in ("left", "right")}
            if row is not None:
                for view, key in (("left", "left"), ("right", "right")):
                    if row[key]["persons"]:
                        kp = np.asarray(row[key]["persons"][0]["keypoints"], dtype=np.float64)
                        for joint in BODY_COCO_INDICES:
                            if np.all(np.isfinite(kp[joint][:2])):
                                observed[view][joint] = kp[joint][:2]
            record = by_frame.get(int(frame_index))
            fitted = {view: {} for view in ("left", "right")}
            if record is not None and "smplx_coco17_joints_ground_mm" in record:
                fitted_mm = np.asarray(record["smplx_coco17_joints_ground_mm"], dtype=np.float64)
                rotation = np.asarray(motion["frames"][frame_index]["rotation_world_from_left"])
                translation = np.asarray(motion["frames"][frame_index]["translation_world_from_left_mm"])
                left_pts = np.asarray([rotation.T @ (fitted_mm[joint] - translation) for joint in BODY_COCO_INDICES])
                right_pts = (cal.R_cam0_to_cam1 @ left_pts.T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
                proj_l = fisheye_project_numpy(left_pts, cal.K0, cal.D0)
                proj_r = fisheye_project_numpy(right_pts, cal.K1, cal.D1)
                for position, joint in enumerate(BODY_COCO_INDICES):
                    fitted["left"][joint] = proj_l[position]
                    fitted["right"][joint] = proj_r[position]
            for view, image in (("left", raw_left), ("right", raw_right)):
                draw = ImageDraw.Draw(image)
                _draw_skeleton(draw, observed[view], (0, 220, 0))
                _draw_skeleton(draw, fitted[view], (255, 40, 40))
                draw.text((20, 20), f"frame {frame_index} pair {pair_id} view {view}  green=observed red=fitted  route={config_name}",
                          fill=(255, 255, 0))
            composite = Image.new("RGB", (raw_w * 2, raw_h), (0, 0, 0))
            composite.paste(raw_left, (0, 0))
            composite.paste(raw_right, (raw_w, 0))
            tag = config_name.replace("-", "_").replace(".", "_")
            composite.save(out_root / f"audit_{tag}_frame_{frame_index:04d}_lr.png")
            raw_left.save(out_root / f"audit_{tag}_frame_{frame_index:04d}_left.png")
            raw_right.save(out_root / f"audit_{tag}_frame_{frame_index:04d}_right.png")
            print(f"[render] saved audit frame {frame_index} for {config_name}")


def _unrotate_to_raw(image, view: str):
    if view == "left":
        return image.rotate(-90, expand=True)
    return image.rotate(90, expand=True)


def _draw_skeleton(draw, points: dict, color: tuple) -> None:
    edges = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
    radius = 7
    for index, point in points.items():
        draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                     fill=color, outline="white", width=2)
    for start, end in edges:
        if start in points and end in points:
            draw.line([tuple(points[start]), tuple(points[end])], fill=color, width=3)


def aggregate(experiment_dir: Path, args: argparse.Namespace) -> None:
    """Merge per-config per-frame/rejection JSONL into experiment-root files."""
    all_frames: list[dict] = []
    all_rejections: list[dict] = []
    for config_dir in (experiment_dir / "routes").glob("*"):
        if not config_dir.is_dir():
            continue
        for name in ("per_frame_metrics.jsonl", "per_frame_metrics_holdout.jsonl"):
            path = config_dir / name
            if path.exists():
                all_frames.extend(read_jsonl_rows(path))
        for name in ("rejection_records.jsonl", "rejection_records_holdout.jsonl"):
            path = config_dir / name
            if path.exists():
                all_rejections.extend(read_jsonl_rows(path))
    validation_root = experiment_dir / "validation"
    if validation_root.is_dir():
        for window_dir in validation_root.glob("*"):
            if not window_dir.is_dir():
                continue
            for config_dir in window_dir.glob("*"):
                for name in ("per_frame_metrics.jsonl", "rejection_records.jsonl"):
                    path = config_dir / name
                    if path.exists():
                        rows = read_jsonl_rows(path)
                        if name.startswith("per_frame"):
                            all_frames.extend(rows)
                        else:
                            all_rejections.extend(rows)
    write_jsonl(experiment_dir / "per_frame_metrics.jsonl", all_frames)
    write_jsonl(experiment_dir / "rejection_records.jsonl", all_rejections)
    print(f"[aggregate] {len(all_frames)} per-frame rows, {len(all_rejections)} rejection rows")


def write_csv_json(base: Path, rows: list[dict]) -> None:
    base.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        keys = list(rows[0].keys())
        lines = [",".join(keys)]
        for row in rows:
            lines.append(",".join(_csv_cell(row.get(key)) for key in keys))
        base.with_suffix(".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    if any(char in text for char in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def read_csv_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    raise SystemExit(main())
