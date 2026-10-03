"""Existing six-stage trainability policy; no method selection or promotion."""

import math
from numbers import Integral


def validate_optimization_options(args):
    """Reject invalid budgets/scales before loading assets or creating outputs."""
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
    return stage_names, stage_train, stage_steps


def set_trainable_parameters(param_map, train):
    """Freeze all parameters, then release only the named stage parameters."""
    for p in param_map.values():
        p.requires_grad_(False)
    for k in train:
        param_map[k].requires_grad_(True)


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
