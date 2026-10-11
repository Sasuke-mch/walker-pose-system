"""Frozen local data information in standardized 169/79-dimensional coordinates."""
import time
from dataclasses import dataclass, field, replace

import torch

from .aggregate import decide
from .geometry_observation import check_rig, project
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


def local_information(store, obs, cameras, noise, scale, surface, *, step=1e-4, chunk=16):
    """Central finite-difference J in train-scaled coordinates, no anchor/J gradients.

    Decoder evaluations and batch calls are counted explicitly. Central differences
    permit the same callable for licensed SMPL and independent analytic checks.
    """
    check_rig(obs, cameras)
    noise.validate_for(obs)
    scale.validate(store.topology)
    if step <= 0 or chunk < 1:
        raise ValueError("positive numerical Jacobian step/chunk required")
    start = time.perf_counter()
    if store.raw.is_cuda:
        torch.cuda.synchronize()
        start = time.perf_counter()
    resources = {"decoder_batch_calls": 0, "decoder_state_evaluations": 0,
                 "method": "central_finite_difference", "step_standardized": step}
    result = Information(None, None, None, None, scale.values.detach().clone(), noise.source,
                         noise.calibrated, [], resources=resources)
    result.provenance = {"identity": obs.identity, "sequence": obs.sequence, "frame": obs.frame,
                         "timestamp": obs.timestamp, "camera_ids": list(obs.camera_ids),
                         "operator": obs.operator, "source": obs.source}
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
    anchor = {key: value.detach() for key, value in anchor.items() if torch.is_tensor(value)}
    result.anchor_id = anchor_choice.selected_id
    result.anchor_rotation = anchor["rotations"]
    m = store.topology.tangent_dimensions
    def pixels(displacement):
        rotations, position, beta = local_parameters(anchor, displacement, scale.values)
        points = surface(rotations, position, beta)["coco17"]
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
            _, diagnostic = schur_shape(matrix)
            diagnostic["rank_sensitivity"] = [schur_shape(matrix, rtol=r)[1] for r in (1e-4, 1e-5, 1e-6, 1e-8, 1e-10)]
            # Gram matrices square singular values. Inspect a whitened-J residual
            # as an additional conditioning audit, rather than treating rank as proof.
            white = torch.linalg.cholesky(precision).T @ jacobian
            Ju, Jb = white[:, :-10], white[:, -10:]
            residual = Jb - Ju @ torch.linalg.pinv(Ju, rtol=1e-6) @ Jb
            diagnostic["whitened_J_nuisance_rank_rtol1e-6"] = int(torch.linalg.matrix_rank(Ju, rtol=1e-6))
            diagnostic["whitened_J_residual_shape_eigenvalues"] = torch.linalg.eigvalsh(residual.T@residual).cpu().tolist()
            diagnostic["whitened_svd_sensitivity"] = [whitened_shape_audit(white, rtol=r)[1]
                                                      for r in (1e-4, 1e-5, 1e-6, 1e-8, 1e-10)]
            diagnostic["rank_warning"] = "Gram pseudoinverse thresholds square Jacobian conditioning; apparent nonzero Schur under truncation is not shape identifiability"
            diagnostic["jacobian_rank"] = int(torch.linalg.matrix_rank(jacobian, rtol=1e-6))
            diagnostic["coordinate_scale_source"] = scale.source
            result.diagnostics = diagnostic
    except (ValueError, RuntimeError) as error:
        result.matrix, result.jacobian = None, None
        result.reasons.append(f"information_unavailable:{type(error).__name__}:{error}")
    if store.raw.is_cuda:
        torch.cuda.synchronize()
    resources["total_ms"] = (time.perf_counter()-start)*1000
    return result
