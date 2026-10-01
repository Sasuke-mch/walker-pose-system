"""Projected Adam proposals with exact nonlinear acceptance, not full SQP.

Constraints must be frame-local; the objective may couple frames. No decoded
joint is overridden. An unsuccessful local probe is not global infeasibility.
"""
import copy
import numpy as np
import torch
from scipy.optimize import minimize


def project_halfspaces(proposal, matrix, bound):
    """Euclidean projection onto A u <= b, using a nonnegative dual QP."""
    proposal = np.asarray(proposal, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    bound = np.asarray(bound, dtype=np.float64)
    if not all(np.isfinite(v).all() for v in (proposal, matrix, bound)):
        raise ValueError("nonfinite projection input")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any((norms < 1e-12) & (bound < -1e-8)):
        return np.zeros_like(proposal), dict(passed=False, reason="constant_infeasible_constraint")
    usable = norms >= 1e-12
    a, b = matrix[usable] / norms[usable, None], bound[usable] / norms[usable]
    if not len(b) or np.max(a @ proposal-b) <= 1e-9:
        return proposal.copy(), dict(passed=True, active=0, iterations=0, max_linear_violation=0.)
    gram = a @ a.T
    rhs = a @ proposal-b
    def objective(lam):
        return .5*lam @ gram @ lam-rhs @ lam, gram @ lam-rhs
    result = minimize(objective, np.zeros(len(b)), jac=True, method="L-BFGS-B",
                      bounds=[(0., None)]*len(b),
                      options=dict(maxiter=1000, ftol=1e-15, gtol=1e-10, maxls=50))
    value = proposal-a.T @ result.x
    violation = float(np.max(a @ value-b))
    return value, dict(passed=violation <= 1e-7, active=int((result.x > 1e-8).sum()),
                       iterations=int(result.nit), max_linear_violation=violation,
                       solver_success=bool(result.success), message=str(result.message))


def frame_local_jacobian(constraints, params, scales):
    """Sum each column across frames; independence yields per-frame rows.

    Callers must never pass temporal/contact-shared constraints to this helper.
    """
    n, m = constraints.shape
    blocks = []
    for column in range(m):
        gradients = torch.autograd.grad(constraints[:, column].sum(), params,
                                        retain_graph=True, allow_unused=True)
        row = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients)]
        blocks.append(torch.cat([(g*s).reshape(n, -1) for g, s in zip(row, scales)], dim=1))
    return torch.stack(blocks, dim=1).detach().cpu().double().numpy()


def probe_local_goal_descent(params, scales, constraints, goals, radius=.02, trials=8):
    """Read-only local wrist probe, never a global feasibility certificate.

    goals must contain frame-local squared distances, one column per wrist.
    A feasible finite witness certifies only this candidate and this state.
    Parameter values/gradients are restored, but tensor version counters change;
    callers must consume/discard prior autograd graphs before this probe.
    """
    old = [p.detach().clone() for p in params]
    try:
        with torch.enable_grad():
            c_live, g_live = constraints(), goals()
            c = c_live.detach().cpu().double().numpy()
            g = g_live.detach().cpu().double().numpy()
            jac = frame_local_jacobian(c_live, params, scales)
            goal_jac = frame_local_jacobian(g_live, params, scales)
        if not np.isfinite(c).all() or c.max() > 0:
            return dict(status='initial_not_exactly_feasible', global_feasibility='unknown')
        gradient = goal_jac.sum(axis=1)
        proposal = -gradient * (radius / np.maximum(np.linalg.norm(gradient, axis=1), 1e-15))[:, None]
        direction = np.zeros_like(proposal)
        frames = []
        for i in range(len(c)):
            direction[i], info = project_halfspaces(proposal[i], jac[i], -c[i])
            direction[i] *= min(1., radius / max(np.linalg.norm(direction[i]), 1e-15))
            active = np.flatnonzero(c[i] >= -.0005)
            frames.append(dict(frame_index=i, active_columns=active.tolist(),
                active_jacobian_rank=int(np.linalg.matrix_rank(jac[i, active])) if len(active) else 0,
                linear_wrist_squared_distance_derivatives=(goal_jac[i] @ direction[i]).tolist(),
                projection=info))
        if not all(f['projection']['passed'] for f in frames):
            return dict(status='linear_projection_not_certified', frames=frames, global_feasibility='unknown')
        widths = [p[0].numel() for p in params]
        attempts = []
        with torch.no_grad():
            for trial in range(trials):
                alpha, cursor = 2.**(-trial), 0
                for p, a, scale, width in zip(params, old, scales, widths):
                    delta = torch.as_tensor(direction[:, cursor:cursor+width], device=p.device, dtype=p.dtype).reshape_as(p)
                    p.copy_(a + alpha * scale * delta)
                    cursor += width
                exact_c = constraints().cpu().double().numpy()
                exact_g = goals().cpu().double().numpy()
                feasible = bool(np.isfinite(exact_c).all() and exact_c.max() <= 0)
                decrease = (exact_g - g).sum(axis=0)
                changed = any(not torch.equal(p, a) for p, a in zip(params, old))
                witness = feasible and changed and bool(np.isfinite(exact_g).all() and decrease.sum() < -1e-12)
                attempts.append(dict(alpha=alpha, exact_feasible=feasible,
                    wrist_squared_distance_change_m2=decrease.tolist(), witness=witness))
                if witness:
                    break
        return dict(status='local_finite_witness' if any(a['witness'] for a in attempts) else 'no_witness_on_this_ray',
                    frames=frames, attempts=attempts, global_feasibility='unknown',
                    scope='original observation/upper-leg constraints; feet/objective not certified by this probe')
    finally:
        with torch.no_grad():
            for p, a in zip(params, old):
                p.copy_(a)


def constrained_adam_step(optimizer, params, evaluate, original_loss, constraints,
                          scales, trust_radius=.2, trials=8, correction_steps=0):
    """Project in normalized coordinates, then check the full coupled objective.

    Rejected/exceptional transactions restore both parameters and Adam state.
    Accepted projected steps retain proposal moments, as existing backtracking
    does; this is a documented heuristic, not a KKT convergence certificate.
    """
    if trust_radius <= 0 or not np.isfinite(trust_radius):
        raise ValueError("positive finite trust radius required")
    n = params[0].shape[0]
    if len(params) != len(scales) or any(p.shape[0] != n for p in params):
        raise ValueError("only frame-local parameter blocks are supported")
    if any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("positive finite parameter scales required")
    if not isinstance(correction_steps, int) or correction_steps < 0:
        raise ValueError("nonnegative integer correction steps required")
    old = [p.detach().clone() for p in params]
    moments = copy.deepcopy(optimizer.state_dict())
    committed, attempts = False, []
    try:
        optimizer.step()
        proposal = [p.detach().clone() for p in params]
        u0 = torch.cat([((b-a)/s).reshape(n, -1) for a,b,s in zip(old,proposal,scales)], dim=1).cpu().double().numpy()
        norms = np.linalg.norm(u0, axis=1)
        u0 *= np.minimum(1., trust_radius/np.maximum(norms, 1e-15))[:, None]
        with torch.no_grad():
            raw_c = constraints().cpu().double().numpy()
            for p,a in zip(params,old):
                p.copy_(a)
        c_live = constraints()
        c = c_live.detach().cpu().double().numpy()
        if not np.isfinite(c).all() or c.max() > 2e-7:
            raise ValueError("projection starts outside exact immutable guard")
        # Existing guard includes its existing 1um numerical tolerance.
        needs_projection = bool((raw_c > 0).any() or (c > -.0005).any())
        report = dict(projected=needs_projection, trust_radius=trust_radius,
                      parameter_scales=list(scales), initial_max_constraint_m=float(c.max()),
                      raw_max_constraint_m=float(raw_c.max()),
                      trust_clipped_frames=int((norms>trust_radius).sum()), frames=[])
        direction = u0.copy()
        if needs_projection:
            jac = frame_local_jacobian(c_live, params, scales)
            for frame in range(n):
                direction[frame], info = project_halfspaces(u0[frame], jac[frame], -c[frame])
                info.update(frame_index=frame,
                            linear_blocking_columns=np.flatnonzero(jac[frame] @ u0[frame]+c[frame] > 0).tolist())
                report['frames'].append(info)
            if not all(r['passed'] for r in report['frames']):
                return dict(accepted=False, alpha=0., attempts=[], projection=report,
                            reason="linear_projection_not_certified")
            # Origin is feasible; shrinking a linear-feasible direction preserves
            # linear feasibility. Nonlinear feasibility is still checked below.
            lengths = np.linalg.norm(direction,axis=1)
            direction *= np.minimum(1.,trust_radius/np.maximum(lengths,1e-15))[:,None]
        widths = [p[0].numel() for p in params]
        deltas = []
        cursor = 0
        for p,s,w in zip(params,scales,widths):
            deltas.append(torch.as_tensor(direction[:,cursor:cursor+w],device=p.device,dtype=p.dtype).reshape_as(p)*s)
            cursor += w
        report['direction_norm'] = float(np.linalg.norm(direction))
        report['proposal_norm'] = float(np.linalg.norm(u0))
        report['objective_directional_derivative'] = sum(float((p.grad*d).sum()) for p,d in zip(params,deltas) if p.grad is not None)
        with torch.no_grad():
            for trial in range(trials):
                alpha = 2.**(-trial)
                for p,a,d in zip(params,old,deltas):
                    p.copy_(a+alpha*d)
                loss, ok, diagnostic = evaluate()
                corrections = []
                # Re-linearize at the rejected trial, rather than repeatedly
                # shrinking a tangent ray outside a curved feasible boundary.
                for correction in range(correction_steps):
                    if ok:
                        break
                    with torch.enable_grad():
                        candidate_c = constraints()
                        values = candidate_c.detach().cpu().double().numpy()
                        if not np.isfinite(values).all() or values.max() <= 0:
                            break  # Other exact guards cannot be repaired here.
                        candidate_jac = frame_local_jacobian(candidate_c, params, scales)
                    updates = []
                    certified = True
                    cursor = 0
                    correction_u = np.zeros_like(direction)
                    for frame in range(n):
                        # 0.1um inward margin is a numerical recovery target,
                        # never an enlargement of the original 1um guard.
                        correction_u[frame], info = project_halfspaces(
                            np.zeros(direction.shape[1]), candidate_jac[frame], -values[frame]-1e-7)
                        certified = certified and info['passed']
                    if not certified:
                        corrections.append(dict(iteration=correction, status='linear_recovery_not_certified'))
                        break
                    for p, s, w in zip(params, scales, widths):
                        updates.append(torch.as_tensor(correction_u[:, cursor:cursor+w], device=p.device,
                                                       dtype=p.dtype).reshape_as(p)*s)
                        cursor += w
                    candidate_u = torch.cat([((p+d-a)/s).reshape(n, -1)
                                             for p,d,a,s in zip(params,updates,old,scales)],dim=1)
                    if bool((torch.linalg.vector_norm(candidate_u,dim=1) > trust_radius+1e-8).any()):
                        corrections.append(dict(iteration=correction, status='recovery_exceeds_radius'))
                        break
                    for p,d in zip(params,updates):
                        p.add_(d)
                    loss, ok, diagnostic = evaluate()
                    corrections.append(dict(iteration=correction, status='evaluated',
                        previous_max_constraint_m=float(values.max()), exact_guard_passed=bool(ok),
                        loss=float(loss), correction_norm=float(np.linalg.norm(correction_u))))
                changed = any(not torch.equal(p,a) for p,a in zip(params,old))
                accepted = bool(changed and ok and torch.isfinite(loss) and float(loss) <= float(original_loss)+1e-6)
                actual_norm = sum(float(((p-a)/s).square().sum()) for p,a,s in zip(params,old,scales))**.5
                attempts.append(dict(alpha=alpha,loss=float(loss),accepted=accepted,
                                     corrections=corrections,
                                     actual_update_norm=actual_norm,
                                     normalized_direction_norm_after_backtrack=float(np.linalg.norm(direction))*alpha, **diagnostic))
                if accepted:
                    committed = True
                    return dict(accepted=True,alpha=alpha,attempts=attempts,projection=report)
        return dict(accepted=False,alpha=0.,attempts=attempts,projection=report,
                    reason="no_nonlinear_feasible_descent_on_projected_ray")
    finally:
        if not committed:
            with torch.no_grad():
                for p,a in zip(params,old):
                    p.copy_(a)
            optimizer.load_state_dict(moments)
