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


def constrained_adam_step(optimizer, params, evaluate, original_loss, constraints,
                          scales, trust_radius=.2, trials=8):
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
                changed = any(not torch.equal(p,a) for p,a in zip(params,old))
                accepted = bool(changed and ok and torch.isfinite(loss) and float(loss) <= float(original_loss)+1e-6)
                actual_norm = sum(float(((p-a)/s).square().sum()) for p,a,s in zip(params,old,scales))**.5
                attempts.append(dict(alpha=alpha,loss=float(loss),accepted=accepted,
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
