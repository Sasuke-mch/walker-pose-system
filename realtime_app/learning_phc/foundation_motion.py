"""Bounded, causal reference-PD diagnostic in the free human/walker scene.

This is actuator target input, not a learned policy or a pose-state replay.
"""
import argparse
from bisect import bisect_right
import json
from math import isfinite
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET


class ReferenceClock:
    """Latest arrived reference, with explicit missing/segment boundaries."""

    def __init__(self, timestamps, available, segments):
        self.timestamps = tuple(float(x) for x in timestamps)
        self.available, self.segments = tuple(available), tuple(segments)
        if (not self.timestamps or len(self.timestamps) != len(self.available) or
                len(self.timestamps) != len(self.segments) or
                not all(isfinite(t) for t in self.timestamps) or
                any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:]))):
            raise ValueError('finite increasing reference times with matching flags required')

    def at(self, time_s):
        if not isfinite(time_s):
            raise ValueError('finite control time required')
        index = bisect_right(self.timestamps, time_s)-1
        if index < 0:
            raise ValueError('no reference has arrived')
        if not self.available[index]:
            raise ValueError('current reference is unavailable: %d' % index)
        if self.segments[index] != self.segments[0]:
            raise ValueError('reference segment changed; explicit reinitialization required: %d' % index)
        return index, time_s-self.timestamps[index]


def run(foundation, output):
    from isaacgym import gymapi  # noqa: F401 -- Gym bindings must load before numerical libraries.
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .contract import NAMES
    from .humanoid_walker import HumanoidWalkerScene
    from .walker_scene import H, clean

    output.mkdir(parents=True, exist_ok=False)
    base = foundation.parent
    source = base/'corrections_v1/reference_matched.npz'
    z = np.load(source, allow_pickle=False)
    clock = ReferenceClock(z['timestamp_s'], z['available'], z['segment_id'])
    xml = foundation/'shape_shoe25_diagonal_asset_v1/humanoid.xml'
    tree = ET.parse(xml)
    lookup = {child.get('name'): parent.get('name') for parent in tree.findall('.//body')
              for child in parent.findall('body')}
    parents = [-1]+[NAMES.index(lookup[n]) for n in NAMES[1:]]
    calibration = np.load(base/'corrections_v1/reference_exact_heel.npz', allow_pickle=False)
    stereo = np.eye(4)
    stereo[:3, :3], stereo[:3, 3] = calibration['R01'], calibration['T01_m'].ravel()
    report = dict(status='failed', scope='first 2 seconds reference-PD; not RL or video reconstruction',
        reference=str(source.resolve()), requested_duration_s=2., planned_control_steps=480,
        controller_dt_s=H, physics_substeps=2, policy_calls=0, training_steps=0,
        training_ready=False, future_reference_used=False,
        grip_assumption='both initial local grips remain attached; actual release labels unknown',
        limits=dict(gap_m=.02, force_N=1000., pelvis_drop_m=.3))
    human, walker, left, right, gaps, forces, times, indices, ages, targets = [], [], [], [], [], [], [], [], [], []
    try:
        with HumanoidWalkerScene(foundation/'synthetic_fixture_asset_v1', xml,
                z['reference_position_m'][0], z['reference_rotation'][0],
                z['camera_rotation'][0], z['camera_translation_m'][0], 400., 4.,
                stereo_transform=stereo, project_initial_contact=True,
                grip_mode='constraint_predict', physics_substeps=2) as s:
            report['initialization'] = s.initialization_contract
            report['timing'] = s.physics_timing
            initial_z = s.human_state()[0][0, 2]
            for tick in range(481):
                time = tick*H
                index, age = clock.at(time)
                p, r, v, w = s.human_state()
                state = s.read(False)
                human.append(np.concatenate([p[..., None], r], axis=2))
                walker.append(np.r_[state['p'], state['q'], state['v'], state['w']])
                left.append(s.camera_mount()); right.append(s.camera_mount('right'))
                gap = np.array([np.linalg.norm(p[g['index']]+r[g['index']]@g['local_wrist']-
                    state['p']-state['R']@g['local_handle']) for g in s.grips])
                gaps.append(gap); times.append(time); indices.append(index); ages.append(age)
                if not np.isfinite(np.r_[p.ravel(), r.ravel(), v.ravel(), w.ravel(), walker[-1]]).all():
                    raise ValueError('nonfinite simulated state')
                if max(gap) > .02 or initial_z-p[0, 2] > .3:
                    raise ValueError('dynamic state gate exceeded at t=%.6f, gap=%s, pelvis_drop=%g' %
                                     (time, gap.tolist(), initial_z-p[0, 2]))
                if tick == 480:
                    break
                rotation = z['reference_rotation'][index]
                local = np.stack([rotation[parents[j]].T@rotation[j] for j in range(1, 24)])
                target = Rotation.from_matrix(local).as_rotvec().ravel().astype(np.float32)
                s.joint_target = target
                s.gym.set_actor_dof_position_targets(s.env, s.human, target)
                connections = s.advance_coupled(active=True)
                force = np.array([g['force_a'] for g in connections])
                forces.append(force); targets.append(target.copy())
                if np.linalg.norm(force, axis=1).max() > 1000.:
                    # Include the already simulated triggering state as well.
                    p, r, _, _ = s.human_state(); state = s.read(False)
                    human.append(np.concatenate([p[..., None], r], axis=2))
                    walker.append(np.r_[state['p'], state['q'], state['v'], state['w']])
                    left.append(s.camera_mount()); right.append(s.camera_mount('right'))
                    gaps.append([np.linalg.norm(p[g['index']]+r[g['index']]@g['local_wrist']-
                        state['p']-state['R']@g['local_handle']) for g in s.grips])
                    times.append((tick+1)*H); indices.append(index); ages.append(age+H)
                    raise ValueError('dynamic force gate exceeded on interval %.6f..%.6f' % (time, time+H))
            report['status'] = 'bounded_reference_pd_completed'
    except Exception as error:
        report.update(reason=str(error), failure_traceback=traceback.format_exc())
    finally:
        report['simulate_calls'] = s.explicit_simulate_calls if 's' in locals() else 0
        report['completed_duration_s'] = times[-1] if times else 0.
        report['not_run_control_steps'] = 480-report['simulate_calls']
        report['target_frames_used'] = sorted(set(indices))
        report['max_grip_gap_m'] = float(np.max(gaps)) if gaps else None
        report['force_peak_N'] = float(np.linalg.norm(forces, axis=2).max()) if forces else None
        report['max_target_age_s'] = max(ages) if ages else None
        np.savez_compressed(output/'trajectory.npz', human=np.asarray(human), walker=np.asarray(walker),
            cameras=np.asarray(left), right_cameras=np.asarray(right), grip_gap_m=np.asarray(gaps),
            grip_force_N=np.asarray(forces), timestamp_s=np.asarray(times), target_index=np.asarray(indices),
            target_age_s=np.asarray(ages), joint_targets=np.asarray(targets))
        (output/'results.json').write_text(json.dumps(clean(report), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(clean(report), ensure_ascii=False))
    return 0 if report['status'] == 'bounded_reference_pd_completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('foundation', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.foundation, args.output))
