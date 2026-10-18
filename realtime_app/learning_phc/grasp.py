"""Paired translational grip models with explicit work accounting.

Both wrenches act at the same world point. This preserves total angular
momentum even when the grip anchors are separated and damping is transverse.
There is no world anchor and no orientation weld.
Only the spring/damper model is passive by construction; impulse position
recovery can do numerical work, which is reported separately.
"""
import numpy as np


def skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def point_jacobian(point, com):
    """Map world COM linear/angular velocity to a world point velocity."""
    return np.column_stack([np.eye(3), -skew(np.asarray(point)-com)])


def free_velocity_predictor(current, previous=None, previous_constraint_response=None):
    """One-step lagged free velocity, before applying the next pair impulse.

    u_now = u_old + free_increment - W_old * impulse_old. Thus the
    measured increment with that response removed estimates free motion.
    This is an explicit approximation, including a one-step contact delay;
    it is not a native simultaneous contact/joint solver.
    """
    current = np.asarray(current, dtype=float)
    if previous is None:
        return current.copy()
    return 2*current-np.asarray(previous)+np.asarray(previous_constraint_response)


def grip_impulse(mobility, error, relative_velocity, dt, recovery_time=.02):
    """Coupled bilateral velocity impulse; positive impulse acts on body A.

    Both reactions must act at a common point. The recovery term corrects
    positional drift, with a fixed time constant across different timesteps.
    Its numerical work is explicit; it is not measured elastic energy.
    """
    if dt <= 0 or recovery_time <= 0 or not np.isfinite([dt, recovery_time]).all():
        raise ValueError('positive finite timestep and recovery time required')
    w = np.asarray(mobility, dtype=float)
    u = np.asarray(relative_velocity, dtype=float).ravel()
    correction = -np.expm1(-dt/recovery_time)*np.asarray(error, dtype=float).ravel()/dt
    if w.shape != (len(u), len(u)) or not np.isfinite(w).all() or not np.allclose(w, w.T, atol=1e-8):
        raise ValueError('finite symmetric constraint mobility required')
    # Cholesky rejects missing directions; no arbitrary regularization/fallback.
    factor = np.linalg.cholesky(w)
    impulse = np.linalg.solve(factor.T, np.linalg.solve(factor, u+correction))
    return impulse, dict(stabilization_work_J=float(impulse@correction),
                         velocity_projection_dissipation_J=float(.5*impulse@w@impulse))


def point_grasp(a, local_a, b, local_b, stiffness, damping, active=True):
    """Return a paired world-space force at the anchor midpoint.

    p/R are body-origin pose; com is world COM; v/w are COM linear/world
    angular velocity. Parameters are explicit assumptions, not fitted loads.
    """
    if not np.isfinite([stiffness, damping]).all() or stiffness <= 0 or damping < 0:
        raise ValueError("positive stiffness and nonnegative damping required")
    xa = a["p"] + a["R"] @ np.asarray(local_a, dtype=float)
    xb = b["p"] + b["R"] @ np.asarray(local_b, dtype=float)
    point = (xa + xb) / 2
    va = a["v"] + np.cross(a["w"], point-a["com"])
    vb = b["v"] + np.cross(b["w"], point-b["com"])
    error, relative = xb-xa, vb-va
    force = stiffness*error + damping*relative if active else np.zeros(3)
    e_dot = (b["v"] + np.cross(b["w"], xb-b["com"]) -
             a["v"] - np.cross(a["w"], xa-a["com"]))
    return {"point": point, "force_a": force, "force_b": -force,
            "anchor_a": xa, "anchor_b": xb, "error_m": float(np.linalg.norm(error)),
            "potential_J": float(0.5*stiffness*(error@error)) if active else 0.0,
            "potential_rate_W": float(stiffness*(error@e_dot)) if active else 0.0,
            "dissipation_W": float(damping*(relative@relative)) if active else 0.0,
            "mechanical_power_W": float(force@(va-vb))}
