"""Existing six-stage trainability policy; no method selection or promotion."""

import math
from numbers import Integral


def validate_optimization_options(args):
    """Reject invalid budgets/scales before loading assets or creating outputs."""
    ankle_multiplier = getattr(args, "body_ankle_reprojection_multiplier", 1.0)
    if not math.isfinite(ankle_multiplier) or ankle_multiplier <= 0:
        raise ValueError("ankle reprojection multiplier must be finite and positive")
    if ankle_multiplier != 1 and getattr(args, "body_observation_mode", "legacy") != "reprojection_only":
        raise ValueError("ankle reprojection multiplier requires reprojection_only")
    if getattr(args, "body_observation_mode", "legacy") == "reprojection_only":
        if args.body_3d_weight != 0 or args.body_reprojection_weight <= 0:
            raise ValueError("reprojection_only requires body-3d-weight=0 and positive body-reprojection-weight")
        if args.wrist_reference_weight > 0 or args.contact_labels is not None:
            raise ValueError("reprojection_only forbids external 3D wrist/contact observation targets")
    for name, default in (("beta_prior_weight", 1e-3), ("body_3d_weight", 1.0)):
        value = getattr(args, name, default)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if getattr(args, "beta_policy", "joint") not in ("joint", "freeze_after_b"):
        raise ValueError("unknown beta policy")
    if getattr(args, "body_weighting", "quality") not in ("quality", "uniform", "no_confidence", "no_ray", "no_reprojection", "reference"):
        raise ValueError("unknown body weighting")
    for name, default in (("body_reference_reprojection_px", 25.0), ("body_reference_ray_gap_mm", 80.0)):
        value = getattr(args, name, default)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    floor = getattr(args, "body_reference_confidence_floor", 0.6)
    if not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("--body-reference-confidence-floor must be in [0,1]")
    for name in (
        "steps",
        "base_steps",
        "beta_steps",
        "joint_steps",
        "hand_steps",
        "contact_steps",
        "contact_refine_steps",
    ):
        value = getattr(args, name)
        if value is not None and (not isinstance(value, Integral) or value < 0):
            raise ValueError(
                f"--{name.replace('_', '-')} must be a non-negative integer"
            )
    for name in ("lr", "body_reprojection_scale_px", "max_init_body_rms_mm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in (
        "vposer_prior_weight",
        "root_anchor_weight",
        "hand_pca_prior_weight",
        "hand_temporal_weight",
        "mano_pose_weight",
        "hand_2d_weight",
        "body_temporal_weight",
        "body_reprojection_weight",
        "wrist_reference_weight",
        "bone_weight",
        "surface_hand_contact_weight",
        "surface_foot_contact_weight",
        "global_hand_handle_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    policy = getattr(args, "fit_policy", "legacy")
    if policy not in ("reliability", "legacy"):
        raise ValueError("unknown fit policy")
    if policy == "reliability":
        if args.body_3d_weight > 0 and args.body_reprojection_weight > 0:
            raise ValueError("reliability policy requires one body data term: 3D and its source 2D are correlated")
        if args.body_3d_weight == 0 and args.body_reprojection_weight == 0:
            raise ValueError("reliability policy requires a positive body observation weight")
        if getattr(args, "body_weighting", "quality") != "quality":
            raise ValueError("historical body-weighting recipes require --fit-policy legacy")
        if getattr(args, "body_reference_confidence_floor", .6) != .6:
            raise ValueError("confidence-floor is legacy-only; reliability always uses zero floor")


def build_stage_schedule(args, contact_enabled: bool):
    """Return names, trainable parameter names and budgets in their execution order."""
    default_stage = max(1, args.steps // 6)
    stage_steps = [
        x if x is not None else default_stage
        for x in (
            args.base_steps,
            args.beta_steps,
            args.joint_steps,
            args.hand_steps,
            args.contact_steps,
            args.contact_refine_steps,
        )
    ]
    if args.steps < 0 or any(x < 0 for x in stage_steps):
        raise ValueError("optimization budgets must be non-negative")
    if not contact_enabled:
        # Without a physical contact target, D3 must not release the whole
        # body's root/translation to explain model-derived WiLoR pixels.
        stage_steps[5] = 0
    stage_names = [
        "A_body_vposer",
        "B_shared_beta",
        "C_body_vposer_refine",
        "D1_hand_proximal",
        "D2_hand_foot_surface_contact"
        if contact_enabled
        else "D2_hand_refine_no_contact",
        "D3_hand_foot_contact_refine",
    ]
    # Contact refinement must be able to move the body/feet as well as the
    # hands.  Keep the no-contact route unchanged: WiLoR observations are
    # model-derived and must not release the body root in that route.
    contact_body_train = ["root", "transl", "latent"] if contact_enabled else []
    stage_train = [
        ["transl", "latent"],
        ["beta"],
        ["beta", "root", "transl", "latent"],
        ["lhand", "rhand"],
        ["lhand", "rhand"] + contact_body_train,
        ["lhand", "rhand"] + contact_body_train,
    ]
    if getattr(args, "beta_policy", "joint") == "freeze_after_b":
        stage_train[2] = ["root", "transl", "latent"]
    return stage_names, stage_train, stage_steps


def set_trainable_parameters(param_map, train):
    """Freeze all parameters, then release only the named stage parameters."""
    for p in param_map.values():
        p.requires_grad_(False)
    for k in train:
        param_map[k].requires_grad_(True)


def uses_local_hand_objective(*, stage_index, train_names, hand_2d_weight, contact_enabled):
    """Local rotations/PCA need no body mesh when no spatial hand term is active."""
    return (
        stage_index >= 3
        and set(train_names) == {"lhand", "rhand"}
        and hand_2d_weight == 0
        and not (contact_enabled and stage_index >= 4)
    )


def require_finite_tensors(tensors, context: str):
    """Stop before an invalid gradient/update or final result can be used."""
    import torch

    checks = {
        name: torch.isfinite(value).all()
        for name, value in tensors.items()
        if value is not None
    }
    if checks and not bool(torch.stack(list(checks.values())).all()):
        failed = [name for name, finite in checks.items() if not bool(finite)]
        raise RuntimeError(f"nonfinite {context}: {', '.join(failed)}")
