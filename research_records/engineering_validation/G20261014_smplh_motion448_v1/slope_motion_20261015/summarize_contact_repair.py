"""Compare the diagnosed slope and the bounded existing-contact repair attempt."""
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / 'realtime_app/tools'))
from build_smplh_motion_comparison import PAGE  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return {'count': int(a.size), 'median': float(np.median(a)) if a.size else None,
            'p95': float(np.percentile(a, 95)) if a.size else None,
            'max': float(a.max()) if a.size else None}


def number(value):
    return float(value) if np.isfinite(value) else None


def main():
    paths = {'camera_map075': OUT.parent / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075',
             'world_contact80_before': OUT / 'run_contact80',
             'world_contact80_fixed': OUT / 'run_contact80_feasible'}
    scene = np.load(OUT / 'scene/scene_transforms.npz', allow_pickle=False)
    R, t = scene['rotation_ground_from_left'], scene['translation_ground_from_left_mm']
    stages = [json.loads(s)['stage']['operational'] for s in
              (OUT / 'scene/stage1_stage2.jsonl').read_text().splitlines()]
    sets = read(ROOT / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')['sets']
    soles = [np.unique(sum(sets[s + '_sole_surface_candidate'].values(), [])) for s in ('left', 'right')]
    identity = None
    inputs = None
    results, curves, errors_by_run, times = {}, {}, {}, None
    for name, path in paths.items():
        p = read(path / 'run_parameters.json')
        with np.load(path / 'result.npz', allow_pickle=False) as r:
            current = [r[k].copy() for k in ('pair_id', 'timestamp_s', 'betas')]
            if identity is not None and any(not np.array_equal(a, b) for a, b in zip(identity, current)):
                raise ValueError('frame/time/beta changed')
            if inputs is not None and p['inputs'] != inputs:
                raise ValueError('observation/model/initialization inputs changed')
            identity, inputs, times = current, p['inputs'], current[1]
            if len(times) != 448:
                raise ValueError('all448 frames required,including failures')
            vg = np.einsum('nij,nvj->nvi', R, r['vertices']) * 1000 + t[:, None]
            cg = np.einsum('nij,nvj->nvi', R, r['predicted_coco']) * 1000 + t[:, None]
            pelvis = np.einsum('nij,nj->ni', R, r['smplh_joints'][:, 0]) * 1000 + t
            if 'world_vertices' in r and not np.allclose(vg / 1000, r['world_vertices'], atol=2e-6, rtol=0, equal_nan=True):
                raise ValueError('camera/world mesh transform mismatch')
            sole = np.stack([vg[:, ix, 2].min(axis=1) for ix in soles], axis=1)
            available = r['pose_available'].copy()
            derivatives = int(r['derivative_valid'].sum())
        rows = [json.loads(s) for s in (path / 'trajectory.jsonl').read_text().splitlines()]
        errors = np.array([row['reprojection_px'] for row in rows], float)
        errors_by_run[name] = errors
        h = []
        i = 0
        while i < 448:
            end = i + 1
            while end < 448 and stages[end] == stages[i]:
                end += 1
            if stages[i] == 'stage1_walker_static_human_moving':
                h.append({'range': [i, end - 1], 'frames': end - i,
                          'pelvis_delta_z_mm': number(pelvis[end - 1, 2] - pelvis[i, 2]),
                          'ankle_mean_delta_z_mm': number(cg[end - 1, 15:17, 2].mean() - cg[i, 15:17, 2].mean()),
                          'sole_endpoint_delta_mm': [number(v) for v in sole[end - 1] - sole[i]]})
            i = end
        result = {'frames': len(rows), 'available': int(available.sum()),
                  'unavailable_frames': np.flatnonzero(~available).tolist(),
                  'raw_error_px': stats(errors), 'whole_raw_view_slots': 10752,
                  'hip_error_px': stats(errors[:, :, 6:8]),
                  'ankle_error_px': stats(errors[:, :, 10:12]),
                  'sole_abs_height_mm': stats(abs(sole)), 'stage1_episodes': h,
                  'published_derivative_joint_samples': derivatives,
                  'ground_configuration': p.get('ground_configuration')}
        if name.startswith('world_contact80'):
            contacts = [row['ground_constraint'] for row in rows]
            result['constraint_satisfied'] = sum(c['constraint_satisfied'] is True for c in contacts)
            result['constraint_failed_frames'] = [i for i, c in enumerate(contacts) if c['constraint_satisfied'] is not True]
            result['stage2_frames'] = sum(c['stage2'] for c in contacts)
            result['stage2_satisfied'] = sum(c['stage2'] and c['constraint_satisfied'] is True for c in contacts)
            result['stage2_failed_frames'] = [i for i, c in enumerate(contacts) if c['stage2'] and c['constraint_satisfied'] is not True]
            result['stage2_entire_sole_displacement_mm'] = stats(
                [v * 1000 for c in contacts if c['stage2'] for v in (c.get('sole_max_displacement_m') or [np.nan, np.nan])])
            result['first_feasible_budget_kept_frames'] = [i for i, c in enumerate(contacts)
                if c.get('budget_acceptance_reference') == 'first_feasible_current_frame_estimate']
            result['optimizer_unconverged_frames'] = [i for i, c in enumerate(contacts) if c['status'] == 'feasible_unconverged']
            result['solver_status_counts'] = {status: sum(c['status'] == status for c in contacts)
                for status in sorted({c['status'] for c in contacts})}
        results[name] = result
        curves[name] = (cg[:, 15:17, 2].mean(axis=1), sole)
    base, before, candidate = results.values()
    pixel_ok = all(candidate['raw_error_px'][key] <= 1.05 * base['raw_error_px'][key] for key in ('median', 'p95'))
    common = np.isfinite(errors_by_run['world_contact80_before']) & np.isfinite(errors_by_run['world_contact80_fixed'])
    passed = candidate['available'] == 448 and candidate['constraint_satisfied'] == 448 and candidate['stage2_satisfied'] == candidate['stage2_frames'] and pixel_ok
    report = {'results': results, 'same_input_time_beta': True,
              'engineering_criteria': {'complete_pose': candidate['available'] == 448,
                  'all_declared_contact': candidate['constraint_satisfied'] == 448,
                  'all_stage2': candidate['stage2_satisfied'] == candidate['stage2_frames'],
                  'median_and_p95_within5percent': pixel_ok, 'passed': passed},
              'comparison_limit': 'Camera versus world differs in state coordinates and solver;before versus fixed contact80 changes only the feasibility acceptance gate',
              'gate_fix_before_after': {
                  'result_keys': ['world_contact80_before', 'world_contact80_fixed'],
                  'available': [before['available'], candidate['available']],
                  'all_contact_satisfied': [before['constraint_satisfied'], candidate['constraint_satisfied']],
                  'stage2_satisfied': [before['stage2_satisfied'], candidate['stage2_satisfied']],
                  'same_finite_view_slots': int(common.sum()),
                  'same_slots_before_raw_px': stats(errors_by_run['world_contact80_before'][common]),
                  'same_slots_after_raw_px': stats(errors_by_run['world_contact80_fixed'][common])},
              'stop': 'retain candidate;do not promote or sweep more parameters' if not passed else 'engineering criteria only;physical validation absent'}
    (OUT / 'repair_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    variants = []
    for name, label, page in [('world_contact80_fixed', '修后接触MAP · 80次预算 · 整体VPoser×4', '../run_contact80_feasible/view/body_browser.html'),
                              ('camera_map075', '原相机系MAP · 整体VPoser×4', '../control_view/body_browser.html'),
                              ('world_contact80_before', '修前接触MAP · 80次预算 · 整体VPoser×4', '../run_contact80/view/body_browser.html')]:
        row = results[name]
        variants.append({'name': name, 'label': label, 'page': page,
                         'summary': f"{label}；可解码{row['available']}/448；原始二维中位{row['raw_error_px']['median']:.2f} / P95 {row['raw_error_px']['p95']:.2f}px。"})
    note = f"本轮工程停止门：{'通过' if passed else '未通过，保留候选，不提升主线'}。仅首帧Sapiens2，后447PMPose；冻结主线beta。接触修前/修后仅改变可行解接受门；与原相机路线比较还改变坐标系及求解器。"
    page = PAGE.replace('三组448帧对照', '接触修补448帧对照')
    config = {'variants': variants, 'times': times.tolist(), 'initializationNote': note}
    destination = OUT / 'visualization/index.html'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(page.replace('__CONFIG__', json.dumps(config, ensure_ascii=False)), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    ids = np.arange(35, 62)
    for name, (ankles, sole) in curves.items():
        style = {'camera_map075': '-', 'world_contact80_before': ':', 'world_contact80_fixed': '--'}[name]
        ax[0].plot(ids, ankles[ids], style, color='black', label=name)
        for j in range(2):
            ax[1].plot(ids, sole[ids, j], style, color=str(.35 * j), label=f'{name} sole{j}')
    ax[0].set_ylabel('Ankle mean Z (mm)')
    ax[1].set_ylabel('Sole minimum Z (mm)')
    ax[1].set_xlabel('Original frame index (Stage1; fixed R/t)')
    ax[1].axhline(0, color='black', linewidth=.6)
    for a in ax:
        a.legend(fontsize=8)
    fig.savefig(OUT / 'repair_stage1_35_61.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report['engineering_criteria']))


if __name__ == '__main__':
    main()
