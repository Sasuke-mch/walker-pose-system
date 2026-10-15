"""Rejected MuJoCo transfer diagnostic; use learning_phc.native for PHC execution.

The frozen policy failed the standing gate on this backend. This entry point
retains that experiment and must not be treated as the accepted runtime.
"""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .physics import PHCBody
from .policy import FrozenPHC, observation_v7

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def stats(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return dict(count=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), max=float(v.max())) if len(v) else dict(count=0)


def standing(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    body = PHCBody(args.xml)
    policy = FrozenPHC(args.primitive, args.composer)
    motion = next(iter(joblib.load(args.motion).values()))
    reference_rotation = Rotation.from_quat(motion['pose_quat_global'][0]).as_matrix()
    root = np.asarray(motion['root_trans_offset'][0])
    initial = body.from_global_rotations(root, reference_rotation)
    body.reset(initial)
    position, actual_rotation, _, _ = body.state()
    initial_rotation_error = np.max(np.abs(actual_rotation - reference_rotation))
    if initial_rotation_error > 1e-5:
        raise ValueError(f'global/local rotation mapping mismatch: {initial_rotation_error}')
    reference = position.copy()
    zero = np.zeros((24, 3))
    positions, errors, actions, weights, timings, torques = [position], [], [], [], [], []
    finite = True
    count = int(round(args.seconds * 30))
    for i in range(count):
        start = time.perf_counter()
        p, r, v, w = body.state()
        observation = observation_v7(p, r, v, w, reference, zero)
        action, mix = policy.act(observation)
        finite = body.advance(action, 1 / 30)
        p = body.state()[0]
        positions.append(p)
        errors.append(np.linalg.norm(p - reference, axis=-1))
        actions.append(action)
        weights.append(mix)
        timings.append((time.perf_counter() - start) * 1000)
        torques.append(body.max_torque)
        if (i + 1) % 90 == 0:
            print(f'step={i+1}/{count} pelvis={p[0,2]:.4f} error={np.mean(errors[-1]):.4f}m', flush=True)
        if not finite:
            break
    positions, errors = np.asarray(positions), np.asarray(errors)
    passed = finite and len(errors) == count and positions[:, 0, 2].min() >= .5 and errors.mean() <= .1
    (output / 'humanoid.xml').write_text(body.xml, encoding='utf-8')
    np.savez_compressed(output / 'rollout.npz', time_s=np.arange(len(positions))/30,
                        position_m=positions, reference_position_m=reference,
                        error_m=errors, action=actions, composition_weight=weights,
                        step_ms=timings, max_torque_nm=torques)
    summary = dict(status='passed_backend_standing_gate' if passed else 'failed_backend_standing_gate',
                   backend='MuJoCo transfer; not author PhysX reproduction',
                   finite=finite, steps=len(errors), requested_steps=count,
                   initial_global_rotation_max_matrix_error=float(initial_rotation_error),
                   pelvis_min_height_m=float(positions[:, 0, 2].min()),
                   joint_tracking_error_m=stats(errors), step_ms=stats(timings),
                   torque_max_nm=float(max(torques)),
                   external_force=False, weights_frozen=True,
                   source_xml=str(args.xml.resolve()), primitive=str(args.primitive.resolve()),
                   composer=str(args.composer.resolve()), motion=str(args.motion.resolve()),
                   versions=dict(torch=torch.__version__),
                   accuracy='backend engineering test with static reference; no real 3D truth')
    save_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['standing'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--xml', type=Path, default=ROOT / 'third_party/PHC/phc/data/assets/mjcf/smpl_0_humanoid.xml')
    parser.add_argument('--primitive', type=Path, default=ROOT / 'models/phc/phc_kp_2.pth')
    parser.add_argument('--composer', type=Path, default=ROOT / 'models/phc/phc_comp_kp_2.pth')
    parser.add_argument('--motion', type=Path, default=ROOT / 'models/phc/standing.pkl')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    standing(args)


if __name__ == '__main__':
    main()
