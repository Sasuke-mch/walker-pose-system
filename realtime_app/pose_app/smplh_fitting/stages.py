"""Existing six-stage trainability policy; no method selection or promotion."""


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
