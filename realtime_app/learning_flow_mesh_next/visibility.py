"""Fixed-anchor coupling reconstruction diagnostic, never an efficacy metric."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .body import SurfaceDecoder
from .candidate_store import CandidateStore
from .geometry_information import local_information
from .geometry_observation import NoiseModel, project
from .information_encoding import fixed_probes
from .tangent import TangentScale


def cross_visibility(matrix, nodes):
    m = 3*nodes+13
    if matrix.shape != (m, m) or not torch.isfinite(matrix).all():
        raise ValueError('finite matching information matrix required')
    slices = [slice(i*3, i*3+3) for i in range(nodes+1)] + [slice(m-10, m)]
    full_energy = float(matrix.square().sum())
    cross = matrix.clone()
    for part in slices:
        cross[part, part] = 0
    cross_energy = float(cross.square().sum())
    rows = []
    visible, projected_energy = 0., 0.
    for i, part in enumerate(slices):
        k = 10 if i == nodes+1 else 3
        outside = torch.ones(m, dtype=torch.bool, device=matrix.device)
        outside[part] = False
        C = cross[part][:, outside]
        P = fixed_probes(m, k).to(matrix)[outside]
        U, singular, _ = torch.linalg.svd(P, full_matrices=False)
        Q = U[:, singular > singular.max()*1e-12]
        energy = float(C.square().sum())
        captured = float((C @ Q).square().sum())
        sketch_energy = float((C @ P).square().sum())
        visible += captured
        projected_energy += sketch_energy
        rows.append({'token': i, 'kind': 'shape' if k == 10 else ('translation' if i == nodes else 'pose'),
                     'cross_energy': energy, 'projectable_fraction': captured/energy if energy else None,
                     'span_rank': int(Q.shape[1])})
    return {'full_energy': full_energy, 'cross_energy': cross_energy,
            'cross_energy_fraction_of_full': cross_energy/full_energy if full_energy else None,
            'projectable_cross_energy_fraction': visible/cross_energy if cross_energy else None,
            'unseen_cross_energy_fraction': 1-visible/cross_energy if cross_energy else None,
            'raw_sketch_energy_ratio_not_retention': projected_energy/cross_energy if cross_energy else None,
            'tokens': rows,
            'boundary': 'Euclidean coordinate-dependent reconstruction diagnostic; not retained Fisher information or efficacy'}


def restricted_information(J, admitted, precision):
    """J rows use full original view-major COCO17 xy order; anchor stays frozen."""
    if J.shape[0] != 68 or admitted.shape != (2, 17) or precision.shape != (2, 17, 2, 2):
        raise ValueError('explicit full-J view/joint row ordering required')
    if not admitted.any():
        return None
    selected = J[admitted.flatten().repeat_interleave(2)]
    W = torch.block_diag(*precision.to(J)[admitted].unbind(0))
    matrix = selected.T @ W @ selected
    return (matrix+matrix.T)/2


def run(cache_dir):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .tests.test_aggregate import rig
    from .tests.test_firewall import observation

    torch.set_num_threads(1)
    g0, g1 = (CandidateStore.load(Path(cache_dir)/name) for name in ('G0.npz', 'G1.npz'))
    keys = ('raw', 'rotations', 'translation', 'beta', 'numerical', 'actions')
    snapshots = [getattr(s, key).clone() for s in (g0, g1) for key in keys]
    meta_keys = ('ids', 'seeds', 'reasons', 'provenance')
    metadata = json.dumps([{key: getattr(s, key) for key in meta_keys} for s in (g0, g1)], sort_keys=True)
    cfg = json.loads(Path('realtime_app/learning_flow_mesh/config.json').read_text(encoding='utf-8'))
    surface = SurfaceDecoder(BodyDecoder(**cfg['assets']).double(), torch.tensor(
        load_coco17_regressor(Path('models/smpl/J_regressor_coco.npy')), dtype=torch.float64))
    cameras, noise, scale = rig(), NoiseModel.proxy(), TangentScale.fixture(g0.topology)
    with torch.no_grad():
        points = surface(torch.eye(3, dtype=torch.float64).expand(1, 52, 3, 3),
                         torch.tensor([[0., 0., 3.]], dtype=torch.float64),
                         torch.zeros(1, 10, dtype=torch.float64))['coco17'][0]
    obs = replace(observation(), xy=torch.stack([project(points, c)[0] for c in cameras]))
    info = local_information(g0, obs, cameras, noise, scale, surface, step=1e-4)
    if info.jacobian is None:
        return {'available': False, 'reasons': info.reasons, 'training_executed': False}
    masks = {}
    for name in ('all', 'left_only', 'right_only', 'drop_wrists', 'upper_body_only', 'lower_body_only', 'none'):
        mask = torch.ones(2, 17, dtype=torch.bool)
        if name == 'left_only':
            mask[1] = False
        elif name == 'right_only':
            mask[0] = False
        elif name == 'drop_wrists':
            mask[:, 9:11] = False
        elif name in ('upper_body_only', 'lower_body_only', 'none'):
            mask[:] = False
            if name == 'upper_body_only':
                mask[:, 5:11] = True
            elif name == 'lower_body_only':
                mask[:, 11:17] = True
        masks[name] = mask
    cases = []
    for name, mask in masks.items():
        matrix = restricted_information(info.jacobian, mask, noise.precision)
        cases.append({'axis': 'admitted_mask', 'case': name, 'admitted_count': int(mask.sum()),
                      'admitted': mask.tolist(), 'available': matrix is not None,
                      'reasons': [] if matrix is not None else ['no_current_observation'],
                      'diagnostic': cross_visibility(matrix, 52) if matrix is not None else None})
    for sigma in (5, 10, 20):
        matrix = restricted_information(info.jacobian, masks['all'], NoiseModel.proxy(sigma).precision)
        cases.append({'axis': 'precision', 'case': f'isotropic_sigma{sigma}', 'sigma_px': sigma,
                      'available': True, 'diagnostic': cross_visibility(matrix, 52)})
    precision = torch.diag(torch.tensor([1/25, 1/100], dtype=torch.float64)).expand(2, 17, 2, 2)
    matrix = restricted_information(info.jacobian, masks['all'], precision)
    cases.append({'axis': 'precision', 'case': 'anisotropic_sigma_x5_y10',
                  'available': True, 'diagnostic': cross_visibility(matrix, 52)})
    unchanged = all(torch.equal(a, b) for a, b in zip(snapshots,
                    (getattr(s, key) for s in (g0, g1) for key in keys)))
    unchanged = unchanged and metadata == json.dumps([
        {key: getattr(s, key) for key in meta_keys} for s in (g0, g1)], sort_keys=True)
    base = cases[0]['diagnostic']
    return {'available': True, 'cache_dir': str(cache_dir), 'anchor_id': info.anchor_id,
            'fixed_J_shape': list(info.jacobian.shape), 'provenance': info.provenance,
            'cases': cases, 'candidate_tensors_masks_unchanged': unchanged,
            'candidate_ids': {'G0': g0.ids, 'G1': g1.ids}, 'resources': info.resources,
            'suspend_sketch_under_preregistered_screen': base['unseen_cross_energy_fraction'] > .8,
            'network_executed': False, 'training_executed': False, 'neural_weights_loaded': False,
            'boundary': 'one analytic S3 observation and existing licensed decoder; no real accuracy or Fisher-information retention claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable output; specify new path')
    report = run(args.cache_dir)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'available': report['available'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
