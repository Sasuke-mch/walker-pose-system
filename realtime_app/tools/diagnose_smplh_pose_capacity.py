"""Bounded image-space capacity probes, not a replacement motion estimate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
from pose_app.smplh_motion_tracker import (
    SMPLHMotionConfig,
    SMPLHMotionTracker,
    SMPLHObservationMap,
)
from run_stereo_pva_baseline import _load_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    p = json.loads((args.source_run / 'run_parameters.json').read_text())
    inp = p['inputs']
    model = SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], device='cuda')
    torch = model.torch
    torch.set_num_threads(2)
    model.model.double()
    model.vposer.double()
    model.betas = model.betas.double()
    model.regressor = model.regressor.double()
    model.hand_rotations = [h.double() for h in model.hand_rotations]
    fixed = [i for i in range(21) if i not in model.fixed_body_indices]
    cal = load_stereo_fisheye(inp['calibration_dir'])
    gate = SMPLHMotionTracker(model, cal, SMPLHMotionConfig(**p['configuration']))
    left = _load_predictions(Path(inp['left_predictions']), 'left')
    right = _load_predictions(Path(inp['right_predictions']), 'right')
    obs = np.stack((left[:, 5:17], right[:, 5:17]), axis=1)
    obs[0] = np.asarray(json.loads(Path(p['first_frame_observations']).read_text())['keypoints'])[:, 5:17]
    a = np.load(args.source_run / 'result.npz')
    protocol = {'source_run': str(args.source_run.resolve()), 'frames': [0, 35, 61],
        'purpose': 'Separate decoder/pose capacity from frozen shape incompatibility; static probes do not establish correct anatomy or motion.',
        'variants': ['latent_pixels_only', 'rotations_pixels_only', 'rotations_and_shape_pixels_only'],
        'data': 'same raw fish-eye pixels, same original stereo gates, equal Huber25/sigma8; no 3D/contact target',
        'regularization': 'numerical anchor delta/100 (intentionally negligible); shape bounded[-3,3], no shape prior',
        'stop': 'One least_squares TRF call per frame/variant, max60 evaluations; no retries/sweeps. Do not adopt these unregularized static poses as tracking.'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows, archives = [], {}

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float64, device='cuda')

    for frame in protocol['frames']:
        valid, _ = gate._supported_pixels(obs[frame])
        selected = np.repeat(valid.ravel(), 2)
        target = obs[frame, ..., :2].reshape(-1)[selected]
        base_t, base_R = a['transl'][frame], a['root_rotation'][frame]
        base_body, base_z = a['body_rotations'][frame], a['vposer_latent'][frame]
        for variant in protocol['variants']:
            latent = variant.startswith('latent')
            shape_free = variant.startswith('rotations_and')
            n = 38 if latent else 6 + 3*len(fixed) + (10 if shape_free else 0)
            scales = np.r_[np.repeat(.1, 3), np.repeat(.5, n-3)]
            if shape_free:
                scales[-10:] = 1.
            cache = {}

            def forward(xx, scales=scales, base_R=base_R, latent=latent, base_z=base_z,
                        base_body=base_body, shape_free=shape_free, base_t=base_t,
                        selected=selected, target=target):
                delta = xx*scales
                count = len(xx)
                roots = base_R @ Rotation.from_rotvec(delta[:, 3:6]).as_matrix()
                with torch.inference_mode():
                    if latent:
                        body = model.vposer.decode(tensor(base_z+delta[:, 6:]))['pose_body_matrot'].reshape(count, 21, 3, 3).clone()
                        body[:, model.fixed_body_indices] = tensor(base_body[list(model.fixed_body_indices)])
                    else:
                        body_np = np.broadcast_to(base_body, (count, 21, 3, 3)).copy()
                        body_np[:, fixed] = base_body[fixed] @ Rotation.from_rotvec(delta[:, 6:6+3*len(fixed)].reshape(-1, 3)).as_matrix().reshape(count, len(fixed), 3, 3)
                        body = tensor(body_np)
                    betas = tensor(a['betas'] + delta[:, -10:]) if shape_free else model.betas.expand(count, -1)
                    result = model.model(betas=betas, transl=tensor(base_t+delta[:, :3]), global_orient=tensor(roots), body_pose=body,
                        left_hand_pose=model.hand_rotations[0].expand(count, -1, -1, -1), right_hand_pose=model.hand_rotations[1].expand(count, -1, -1, -1))
                    coco = torch.einsum('jv,bvc->bjc', model.regressor, result.vertices).cpu().numpy()
                points = coco[:, 5:17]
                other = points @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm/1000
                proj = np.stack((fisheye_project_numpy(points, cal.K0, cal.D0), fisheye_project_numpy(other, cal.K1, cal.D1)), axis=1)
                raw = (proj.reshape(count, -1)[:, selected]-target).reshape(count, -1, 2)
                norms = np.linalg.norm(raw, axis=-1)
                costs = np.where(norms <= 25, .5*norms**2, 25*(norms-12.5))
                residual = raw * (np.sqrt(2*costs)/np.maximum(norms, 1e-12))[..., None]/8
                return np.c_[residual.reshape(count, -1), delta/100], coco, proj, body.cpu().numpy()

            def residual(x, cache=cache, forward=forward):
                if not cache or not np.array_equal(x, cache['x']):
                    val, coco, proj, body = forward(x[None])
                    cache.clear()
                    cache.update(x=x.copy(), residual=val[0], coco=coco[0], proj=proj[0], body=body[0])
                return cache['residual']

            def jacobian(x, n=n, forward=forward):
                eps = 1e-3
                offsets = np.r_[x[None]+np.eye(n)*eps, x[None]-np.eye(n)*eps]
                values = forward(offsets)[0]
                return ((values[:n]-values[n:])/(2*eps)).T

            before = residual(np.zeros(n)).copy()
            initial_error = np.linalg.norm(cache['proj']-obs[frame, ..., :2], axis=-1)
            lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
            if latent:
                lower[6:], upper[6:] = (-6-base_z)/scales[6:], (6-base_z)/scales[6:]
            if shape_free:
                lower[-10:], upper[-10:] = -3-a['betas'][0], 3-a['betas'][0]
            fit = least_squares(residual, np.zeros(n), jac=jacobian, bounds=(lower, upper),
                max_nfev=60, ftol=1e-6, xtol=1e-6, gtol=.05)
            residual(fit.x)
            error = np.linalg.norm(cache['proj']-obs[frame, ..., :2], axis=-1)
            co = cache['coco']
            row = {'frame': frame, 'variant': variant, 'nfev': fit.nfev, 'success': bool(fit.success), 'termination': fit.message,
                'optimality': fit.optimality, 'cost_before': .5*float(before@before), 'cost_after': float(fit.cost),
                'hip_error_before_median_px': float(np.median(initial_error[:, 6:8])),
                'hip_error_after_median_px': float(np.median(error[:, 6:8])),
                'all_error_median_px': float(np.median(error[valid])), 'all_error_p95_px': float(np.percentile(error[valid], 95)),
                'hip_width_mm': float(np.linalg.norm(co[11]-co[12])*1000),
                'hip_midpoint_camera_m': co[11:13].mean(0).tolist(),
                'limb_lengths_mm': [float(np.linalg.norm(co[i]-co[j])*1000) for i,j in ((11,13),(13,15),(12,14),(14,16))],
                'betas': (a['betas'][0]+fit.x[-10:]).tolist() if shape_free else a['betas'][0].tolist(),
                'errors_px': error.tolist(), 'used': valid.tolist()}
            rows.append(row)
            archives[f'{frame}_{variant}_coco'] = co
            archives[f'{frame}_{variant}_body'] = cache['body']
            archives[f'{frame}_{variant}_root'] = base_R @ Rotation.from_rotvec(fit.x[3:6]*scales[3:6]).as_matrix()
            archives[f'{frame}_{variant}_transl'] = base_t+fit.x[:3]*scales[:3]
            archives[f'{frame}_{variant}_delta'] = fit.x*scales
            np.savez_compressed(args.output / 'poses.npz', **archives)
            (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(frame, variant, 'cost', row['cost_before'], row['cost_after'], 'hip', row['hip_error_after_median_px'], 'width', row['hip_width_mm'], 'nfev', fit.nfev, flush=True)


if __name__ == '__main__':
    main()
