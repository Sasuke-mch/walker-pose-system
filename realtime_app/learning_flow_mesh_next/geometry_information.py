"""Frozen local data information in standardized 169/79-dimensional coordinates."""
import time
from dataclasses import dataclass, field, replace

import torch

from .aggregate import decide
from .body import observation_points
from .geometry_observation import check_rig, observation_errors, project
from .input_binding import information_snapshot
from .tangent import skew


@dataclass
class Information:
    matrix: torch.Tensor | None
    jacobian: torch.Tensor | None
    anchor_id: str | None
    anchor_rotation: torch.Tensor | None
    scale: torch.Tensor
    noise_source: str
    calibrated: bool
    reasons: list
    diagnostics: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    binding: dict = field(default_factory=dict)


@dataclass
class EndpointCostVJP:
    """Endpoint-only cost covectors; unavailable slots stay NaN, never zero."""
    gradient: torch.Tensor
    available: torch.Tensor
    costs: torch.Tensor
    reasons: list
    ids: list
    scale: torch.Tensor
    noise_source: str
    calibrated: bool
    binding: dict
    resources: dict


def endpoint_cost_vjp(store, obs, cameras, noise, scale, surface, *, huber=3.0):
    """One cost-sum reverse pass for independent full SMPLH52 candidate rows.

    Only standardized local delta is differentiated. This is a covector, not
    an angular velocity, covariance or replacement for FD neighborhood guards.
    The surface callable must decode rows independently (as real SMPL LBS does).
    """
    from .so3_flow import rotation_exp

    if store.topology.nodes != 52:
        raise ValueError("endpoint VJP requires complete SMPLH52")
    k, m = len(store.raw), store.topology.tangent_dimensions
    if (store.rotations.shape != (k, 52, 3, 3) or store.translation.shape != (k, 3)
            or store.beta.shape != (k, 10) or store.numerical.shape != (k,)
            or len(store.reasons) != k or len(store.ids) != k):
        raise ValueError("complete candidate slot shapes required")
    if not 0 < huber < float("inf"):
        raise ValueError("positive finite Huber threshold required")
    # no_grad and inference_mode differ: make ordinary detached tensors here,
    # including when both callers and their candidate tensors are in inference mode.
    with torch.inference_mode(False), torch.enable_grad():
        obs = replace(obs, **{name: getattr(obs, name).detach().clone() for name in
                              ("xy", "confidence", "present", "quality", "geometry",
                               "views", "origins", "directions")})
        cameras = tuple(replace(camera, **{name: getattr(camera, name).detach().clone()
                                          for name in ("K", "D", "R", "T")}) for camera in cameras)
        noise = replace(noise, precision=noise.precision.detach().clone())
        scale = replace(scale, values=scale.values.detach().clone())
        binding = information_snapshot(obs, cameras, noise)
        scale.validate(store.topology)
        result = EndpointCostVJP(
            store.raw.new_full((k, m), torch.nan),
            torch.zeros(k, device=store.raw.device, dtype=torch.bool),
            store.raw.new_full((k,), torch.nan), [list(row) for row in store.reasons],
            list(store.ids), scale.values, noise.source, noise.calibrated, binding,
            {"decoder_batch_calls": 0, "decoder_state_evaluations": 0, "reverse_calls": 0,
             "method": "endpoint_cost_sum_reverse", "admitted_count": int(obs.admitted().sum()),
             "boundary": "independent decoder rows; endpoint-only validity; no FD probes"})

        def reject(indices, reason):
            for index in indices:
                result.reasons[index].append(reason)

        candidate_binding = store.provenance.get("observation_binding")
        if candidate_binding is None or candidate_binding != binding["observation"]:
            reject(range(k), "candidate_observation_binding_missing" if candidate_binding is None
                   else "candidate_observation_binding_mismatch")
            return result
        for key in ("identity", "sequence", "frame", "timestamp"):
            if key in store.provenance and store.provenance[key] != binding["observation"][key]:
                reject(range(k), f"candidate_observation_provenance_mismatch:{key}")
                return result
        if not obs.admitted().any():
            reject(range(k), "no_current_observation")
            return result
        rotations = store.rotations.detach().clone()
        translation, beta = store.translation.detach().clone(), store.beta.detach().clone()
        finite = (torch.isfinite(rotations).flatten(1).all(-1)
                  & torch.isfinite(translation).all(-1) & torch.isfinite(beta).all(-1)
                  & torch.isfinite(store.raw).all(-1))
        legal = store.numerical.detach().clone() & finite
        rows = legal.nonzero().flatten()
        if len(rows):
            identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
            proper = ((rotations[rows].transpose(-1, -2) @ rotations[rows]-identity)
                      .abs().flatten(1).amax(-1) <= 1e-4)
            proper &= (torch.linalg.det(rotations[rows])-1).abs().amax(-1) <= 1e-4
            legal[rows] &= proper
        reject((~legal).nonzero().flatten().tolist(), "numerically_invalid_endpoint")
        rows = legal.nonzero().flatten()
        if not len(rows):
            return result
        try:
            delta = store.raw.new_zeros(len(rows), m, requires_grad=True)
            physical = delta * scale.values.to(delta)
            perturbed = rotations[rows] @ rotation_exp(physical[:, :156].reshape(-1, 52, 3))
            position = translation[rows] + physical[:, 156:159]
            shape = beta[rows] + physical[:, -10:]
            result.resources["decoder_batch_calls"] += 1
            result.resources["decoder_state_evaluations"] += len(rows)
            points = observation_points(surface, perturbed, position, shape)
            costs, _, uv, valid = observation_errors(points, obs, cameras, noise, huber=huber)
            projected = (valid[:, obs.admitted()].all(-1)
                         & torch.isfinite(uv[:, obs.admitted()]).flatten(1).all(-1))
            reject(rows[~projected].tolist(), "admitted_projection_invalid")
            cost_finite = torch.isfinite(costs)
            reject(rows[projected & ~cost_finite].tolist(), "nonfinite_endpoint_cost")
            usable = projected & cost_finite
            if not usable.any():
                return result
            result.resources["reverse_calls"] += 1
            # Sum candidate means: dividing by K would shrink each covector.
            gradient = torch.autograd.grad(costs[usable].sum(), delta, create_graph=False)[0]
            finite_gradient = torch.isfinite(gradient).all(-1)
            reject(rows[usable & ~finite_gradient].tolist(), "nonfinite_endpoint_gradient")
            good = usable & finite_gradient
            result.gradient[rows[good]] = gradient[good].detach()
            result.costs[rows[usable]] = costs[usable].detach()
            result.available[rows[good]] = True
        except (ValueError, RuntimeError) as error:
            # One failed batch affects its slots; never retry or redraw privately.
            reject(rows.tolist(), f"endpoint_vjp_unavailable:{type(error).__name__}:{error}")
        return result


def schur_shape(matrix, *, rtol=1e-6):
    if rtol <= 0 or not torch.isfinite(matrix).all():
        raise ValueError("finite information and explicit positive pseudoinverse threshold required")
    u, ub, bb = matrix[:-10, :-10], matrix[:-10, -10:], matrix[-10:, -10:]
    schur = bb - ub.T @ torch.linalg.pinv(u, rtol=rtol, hermitian=True) @ ub
    schur = (schur+schur.T)/2
    return schur, {"pinv_rtol": rtol, "nuisance_rank": int(torch.linalg.matrix_rank(u, rtol=rtol, hermitian=True)),
                   "shape_schur_eigenvalues": torch.linalg.eigvalsh(schur).detach().cpu().tolist(),
                   "shape_schur_rank": int(torch.linalg.matrix_rank(schur, rtol=rtol, hermitian=True)),
                   "boundary": "local linear diagnostic; prior regularization excluded"}


def local_parameters(anchor, displacement, scale):
    n = anchor["rotations"].shape[0]
    delta = displacement * scale.to(displacement)
    rotations = anchor["rotations"][None] @ torch.matrix_exp(skew(delta[:, :3*n].reshape(-1, n, 3)))
    position = anchor["translation"][None] + delta[:, 3*n:3*n+3]
    beta = anchor["beta"][None] + delta[:, -10:]
    return rotations, position, beta


def whitened_shape_audit(white, *, rtol=1e-6):
    """Project out nuisance with SVD, without squaring conditioning first.

    Rank is threshold dependent. A residual is a local linear diagnostic, not
    an assertion of globally identifiable shape or independent measurement truth.
    """
    if (white.ndim != 2 or white.shape[1] <= 10 or white.shape[0] == 0
            or not torch.isfinite(white).all() or not 0 < rtol < 1):
        raise ValueError("finite nonempty whitened Jacobian and 0<rtol<1 required")
    nuisance, shape = white[:, :-10], white[:, -10:]
    left, singular, _ = torch.linalg.svd(nuisance, full_matrices=False)
    retained = singular > rtol*singular.max()
    basis = left[:, retained]
    residual = shape - basis @ (basis.T @ shape)
    gram = residual.T @ residual
    signal = float(shape.square().sum())
    return gram, {
        "jacobian_rtol": rtol, "equivalent_gram_rtol": rtol**2,
        "nuisance_rank": int(retained.sum()),
        "nuisance_singular_values": singular.cpu().tolist(),
        "residual_shape_eigenvalues": torch.linalg.eigvalsh(gram).cpu().tolist(),
        "residual_shape_energy_fraction": float(residual.square().sum())/signal if signal > 0 else None,
        "shape_signal_present": signal > 0,
        "boundary": "threshold-dependent local linear diagnostic; no prior or identifiability certificate",
    }


def information_diagnostics(matrix, jacobian, precision):
    """Optional rank/shape study; it does not construct or validate Lambda."""
    _, diagnostic = schur_shape(matrix)
    diagnostic["rank_sensitivity"] = [dict(diagnostic) if r == 1e-6 else schur_shape(matrix, rtol=r)[1]
                                      for r in (1e-4, 1e-5, 1e-6, 1e-8, 1e-10)]
    # Gram matrices square singular values; direct whitened-J diagnostics are
    # retained for research, without treating a numerical rank as identifiability.
    white = torch.linalg.cholesky(precision).T @ jacobian
    Ju, Jb = white[:, :-10], white[:, -10:]
    residual = Jb - Ju @ torch.linalg.pinv(Ju, rtol=1e-6) @ Jb
    diagnostic["whitened_J_nuisance_rank_rtol1e-6"] = int(torch.linalg.matrix_rank(Ju, rtol=1e-6))
    diagnostic["whitened_J_residual_shape_eigenvalues"] = torch.linalg.eigvalsh(residual.T@residual).cpu().tolist()
    diagnostic["whitened_svd_sensitivity"] = [whitened_shape_audit(white, rtol=r)[1]
                                              for r in (1e-4, 1e-5, 1e-6, 1e-8, 1e-10)]
    diagnostic["rank_warning"] = "Gram pseudoinverse thresholds square Jacobian conditioning; apparent nonzero Schur under truncation is not shape identifiability"
    diagnostic["jacobian_rank"] = int(torch.linalg.matrix_rank(jacobian, rtol=1e-6))
    return diagnostic


def local_information(store, obs, cameras, noise, scale, surface, *, step=1e-4, chunk=None, diagnostics=True):
    """Central finite-difference J in train-scaled coordinates, no anchor/J gradients.

    Decoder evaluations and batch calls are counted explicitly. Central differences
    permit the same callable for licensed SMPL and independent analytic checks.
    Research diagnostics are optional and cannot invalidate a finite guarded J.
    """
    if not isinstance(diagnostics, bool):
        raise TypeError("explicit boolean diagnostics option required")
    check_rig(obs, cameras)
    noise.validate_for(obs)
    scale.validate(store.topology)
    if chunk is None:
        # Observation-only LBS has bounded vertex work; batch every original
        # +/- probe. Generic/full-mesh callables retain the old memory bound.
        chunk = 2*store.topology.tangent_dimensions if hasattr(surface, "coco17") else 16
    if step <= 0 or chunk < 1:
        raise ValueError("positive numerical Jacobian step/chunk required")
    start = time.perf_counter()
    if store.raw.is_cuda:
        torch.cuda.synchronize()
        start = time.perf_counter()
    resources = {"decoder_batch_calls": 0, "decoder_state_evaluations": 0,
                 "method": "central_finite_difference", "step_standardized": step,
                 "chunk": chunk,
                 "diagnostics_requested": diagnostics}
    result = Information(None, None, None, None, scale.values.detach().clone(), noise.source,
                         noise.calibrated, [], resources=resources)
    result.provenance = {"identity": obs.identity, "sequence": obs.sequence, "frame": obs.frame,
                         "timestamp": obs.timestamp, "camera_ids": list(obs.camera_ids),
                         "operator": obs.operator, "source": obs.source}
    result.binding = information_snapshot(obs, cameras, noise)
    candidate_binding = store.provenance.get("observation_binding")
    binding_status = "legacy_unverified" if candidate_binding is None else "verified"
    result.diagnostics["candidate_current_observation_binding"] = binding_status
    result.diagnostics["status"] = "not_computed" if diagnostics else "not_requested"
    result.diagnostics["coordinate_scale_source"] = scale.source
    if candidate_binding is not None and candidate_binding != result.binding["observation"]:
        result.reasons.append("candidate_anchor_observation_binding_mismatch")
        return result
    for key in ("identity", "sequence", "frame", "timestamp"):
        if key in store.provenance and store.provenance[key] != result.provenance[key]:
            result.reasons.append(f"candidate_anchor_observation_provenance_mismatch:{key}")
            return result
    admitted = obs.admitted()
    if not admitted.any():
        result.reasons.append("no_current_observation")
        return result
    anchor_choice = decide(replace(store, actions=store.numerical.clone()), "A4", scale)
    if anchor_choice.selected_index is None:
        result.reasons.append("no_numerically_legal_anchor")
        return result
    anchor = store.select(anchor_choice.selected_index) if store.actions[anchor_choice.selected_index] else {
        "rotations": store.rotations[anchor_choice.selected_index],
        "translation": store.translation[anchor_choice.selected_index],
        "beta": store.beta[anchor_choice.selected_index]}
    anchor = {key: value.detach().clone() for key, value in anchor.items() if torch.is_tensor(value)}
    result.anchor_id = anchor_choice.selected_id
    result.anchor_rotation = anchor["rotations"]
    m = store.topology.tangent_dimensions
    def pixels(displacement):
        rotations, position, beta = local_parameters(anchor, displacement, scale.values)
        points = observation_points(surface, rotations, position, beta)
        resources["decoder_batch_calls"] += 1
        resources["decoder_state_evaluations"] += len(displacement)
        views = [project(points, camera) for camera in cameras]
        uv = torch.stack([value[0] for value in views], 1)
        valid = torch.stack([value[1] for value in views], 1)
        if not valid[:, admitted].all() or not torch.isfinite(uv[:, admitted]).all():
            raise ValueError("anchor_or_perturbation_projection_invalid")
        return uv[:, admitted].flatten(1)
    try:
        with torch.no_grad():
            # Base projection validates the actual anchor before any perturbation.
            pixels(store.raw.new_zeros(1, m))
            basis = torch.eye(m, device=store.raw.device, dtype=store.raw.dtype)*step
            perturbations = torch.cat((basis, -basis), 0)
            output = torch.cat([pixels(part) for part in perturbations.split(chunk)], 0)
            jacobian = ((output[:m]-output[m:])/(2*step)).T
            precision = torch.block_diag(*noise.precision.to(jacobian)[admitted].unbind(0))
            matrix = jacobian.T @ precision @ jacobian
            matrix = (matrix+matrix.T)/2
            if not torch.isfinite(matrix).all():
                raise ValueError("nonfinite_information")
            result.matrix, result.jacobian = matrix.detach(), jacobian.detach()
    except (ValueError, RuntimeError) as error:
        result.matrix, result.jacobian = None, None
        result.reasons.append(f"information_unavailable:{type(error).__name__}:{error}")
    if diagnostics and result.matrix is not None:
        try:
            with torch.no_grad():
                result.diagnostics.update(information_diagnostics(result.matrix, result.jacobian, precision))
            result.diagnostics["status"] = "computed"
        except (ValueError, RuntimeError) as error:
            result.diagnostics.update(status="unavailable", reason=f"{type(error).__name__}:{error}")
    if store.raw.is_cuda:
        torch.cuda.synchronize()
    resources["total_ms"] = (time.perf_counter()-start)*1000
    return result
