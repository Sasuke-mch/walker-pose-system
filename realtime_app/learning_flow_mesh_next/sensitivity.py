"""Fixed-cache numerical and proxy-noise audit; no generators or policy tuning.

Reference for step/error diagnostics:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.jacobian.html
This uses existing central differences, not SciPy's higher-order adaptive routine.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .aggregate import decide
from .body import SurfaceDecoder
from .candidate_store import CandidateStore
from .geometry_information import local_information, local_parameters
from .geometry_observation import NoiseModel, project
from .information_risk import RiskPolicy, decide_information, information_metric
from .tangent import TangentScale


def reweight_information(information, observation, noise):
    """Reuse unweighted J at exactly the same observation/anchor, not old Lambda."""
    noise.validate_for(observation)
    observation.validate()
    expected = {"identity": observation.identity, "sequence": observation.sequence,
                "frame": observation.frame, "timestamp": observation.timestamp,
                "camera_ids": list(observation.camera_ids), "operator": observation.operator,
                "source": observation.source}
    if information.provenance != expected:
        raise ValueError("reweighting requires exactly the original observation provenance")
    J = information.jacobian
    if J is None:
        raise ValueError("cannot reweight unavailable Jacobian")
    precision = torch.block_diag(*noise.precision.to(J)[observation.admitted()].unbind(0))
    matrix = J.T @ precision @ J
    return replace(information, matrix=(matrix+matrix.T)/2,
                   noise_source=noise.source, calibrated=noise.calibrated,
                   diagnostics={"scope": "reweighted J; old noise diagnostics deliberately excluded"})


def relative_error(value, reference):
    denominator = float(torch.linalg.vector_norm(reference))
    absolute = float(torch.linalg.vector_norm(value-reference))
    return {"absolute_frobenius": absolute,
            "relative_frobenius": absolute/denominator if denominator > 0 else None}


def decision_report(store, information, scale, obs, cameras, noise, surface, *, lambda0=1, lambda_g=.1):
    decision, _ = decide_information(
        store, information, scale, RiskPolicy(lambda0=lambda0, lambda_g=lambda_g,
                                             source="analytic_fixture_sensitivity_not_validation_locked"),
        observation=obs, cameras=cameras, noise=noise, surface=surface)
    return {"selected_id": decision.selected_id, "reasons": decision.reasons,
            "fallback": decision.diagnostic.get("fallback"), "gap": decision.risk_gap,
            "costs": [float(x) if torch.isfinite(x) else None for x in decision.costs],
            "lambda0": lambda0, "lambda_g": lambda_g}


def run(cache_dir):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .tests.test_aggregate import rig
    from .tests.test_firewall import observation

    torch.set_num_threads(1)
    cache_dir = Path(cache_dir)
    g0, store = CandidateStore.load(cache_dir/'G0.npz'), CandidateStore.load(cache_dir/'G1.npz')
    tensor_keys = ('raw', 'rotations', 'translation', 'beta', 'numerical', 'actions')
    original = [getattr(s, key).clone() for s in (g0, store) for key in tensor_keys]
    metadata = json.dumps([{key: getattr(s, key) for key in ('ids', 'seeds', 'reasons', 'provenance')}
                           for s in (g0, store)], sort_keys=True)
    cfg = json.loads(Path('realtime_app/learning_flow_mesh/config.json').read_text(encoding='utf-8'))
    surface = SurfaceDecoder(BodyDecoder(**cfg['assets']).double(), torch.tensor(
        load_coco17_regressor(Path('models/smpl/J_regressor_coco.npy')), dtype=torch.float64))
    cameras, scale, noise = rig(), TangentScale.fixture(store.topology), NoiseModel.proxy()
    with torch.no_grad():
        points = surface(torch.eye(3, dtype=torch.float64).expand(1, 52, 3, 3),
                         torch.tensor([[0., 0., 3.]], dtype=torch.float64),
                         torch.zeros(1, 10, dtype=torch.float64))['coco17'][0]
    obs = replace(observation(), xy=torch.stack([project(points, c)[0] for c in cameras]).detach())
    base = local_information(g0, obs, cameras, noise, scale, surface, step=1e-4)
    if base.matrix is None:
        return {"available": False, "reasons": base.reasons, "cache_dir": str(cache_dir),
                "network_executed": False, "training_executed": False}
    anchor_index = g0.ids.index(base.anchor_id)
    anchor = {key: getattr(g0, key)[anchor_index].detach() for key in ('rotations', 'translation', 'beta')}
    def pixels(delta):
        R, T, beta = local_parameters(anchor, delta[None], scale.values)
        points = surface(R, T, beta)['coco17']
        return torch.stack([project(points, c)[0] for c in cameras], 1)[:, obs.admitted()].flatten()
    generator = torch.Generator().manual_seed(20261011)
    directions = torch.randn(4, store.topology.tangent_dimensions, generator=generator, dtype=torch.float64)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    derivatives = torch.stack([torch.autograd.functional.jvp(
        pixels, torch.zeros_like(d), d, strict=True)[1].detach() for d in directions], 1)
    steps = []
    for step in (1e-3, 1e-4, 1e-5, 1e-6):
        info = base if step == 1e-4 else local_information(g0, obs, cameras, noise, scale, surface, step=step)
        row = {"step": step, "available": info.matrix is not None, "reasons": info.reasons,
               "anchor_id": info.anchor_id, "resources": info.resources}
        if info.matrix is not None:
            row.update(jacobian_vs_baseline=relative_error(info.jacobian, base.jacobian),
                       directional_vs_autograd=relative_error(info.jacobian @ directions.T, derivatives),
                       shape_svd=info.diagnostics['whitened_svd_sensitivity'],
                       decision=decision_report(store, info, scale, obs, cameras, noise, surface))
        steps.append(row)
    Bbase = information_metric(base, RiskPolicy())
    sigmas = []
    for sigma in (5., 10., 20.):
        proxy = NoiseModel.proxy(sigma)
        info = reweight_information(base, obs, proxy)
        ratio = (10/sigma)**2
        metric = information_metric(info, RiskPolicy())
        compensated = information_metric(info, RiskPolicy(lambda0=ratio))
        sigmas.append({"sigma_px": sigma, "precision_ratio": ratio,
                       "matrix_scaling_error": relative_error(info.matrix, base.matrix*ratio),
                       "fixed_lambda0_metric_change": relative_error(metric, Bbase),
                       "compensated_lambda0_metric_change": relative_error(compensated, Bbase),
                       "A6_metric_only": decision_report(store, info, scale, obs, cameras, proxy, surface, lambda_g=0),
                       "A6_fixed_policy": decision_report(store, info, scale, obs, cameras, proxy, surface),
                       "A6_compensated_metric_only": decision_report(
                           store, info, scale, obs, cameras, proxy, surface, lambda0=ratio, lambda_g=0),
                       "A5_fixed_geometry_control": decide(
                           store, 'A5', scale, observation=obs, cameras=cameras,
                           noise=proxy, surface=surface, lambda_g=.1).selected_id})
    unchanged = all(torch.equal(a, b) for a, b in zip(original,
                    (getattr(s, key) for s in (g0, store) for key in tensor_keys)))
    unchanged = unchanged and metadata == json.dumps([
        {key: getattr(s, key) for key in ('ids', 'seeds', 'reasons', 'provenance')}
        for s in (g0, store)], sort_keys=True)
    return {"available": True, "cache_dir": str(cache_dir), "steps": steps, "sigmas": sigmas,
            "candidate_ids": {"G0": g0.ids, "G1": store.ids}, "candidates_masks_unchanged": unchanged,
            "anchor_id": base.anchor_id, "jvp_directions": 4, "jvp_seed": 20261011,
            "observation_provenance": base.provenance,
            "assets": cfg['assets'], "coco17_regressor": "models/smpl/J_regressor_coco.npy",
            "reference_scope": "four directional AD checks, not an independent full Jacobian or physical truth",
            "G1_reconditioned": False, "network_executed": False, "training_executed": False,
            "neural_weights_loaded": False, "decoder": "existing licensed SMPL-H statistical asset",
            "dtype": str(store.raw.dtype), "sigma_calibrated": False,
            "boundary": "fixed random candidate aggregator sensitivity and derivative cross-check only; no efficacy or calibration"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable audit output; use a new path')
    report = run(args.cache_dir)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({"output": str(args.output), "available": report['available'],
                      "network_executed": False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
