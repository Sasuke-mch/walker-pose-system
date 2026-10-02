"""Cold full448 detector fit: reuse solver/settings, never fitted arrays."""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import PROJECT_ROOT as _tool_project_root
_tool_prepare_imports()
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = _tool_project_root
from pose_app.independent_wrist_reference import load_reference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--detector', choices=('sapiens2', 'pmpose'), default='sapiens2')
    ap.add_argument('--joint-steps', type=int, default=30,
                    help='body/shape joint budget; default matches original cold settings')
    a = ap.parse_args()
    if a.joint_steps <= 0:
        raise ValueError('joint steps must be positive')
    run, output = a.run.resolve(), a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    scene = run / a.detector / 'scene'
    sources = read(scene/'scene_sources.json')
    body_paths = {side: Path(sources[f'raw_{side}_pmpose']).resolve() for side in ('left','right')}
    for side in ('left', 'right'):
        expected = (run/f'{side}_sapiens_coco17_model_input.json' if a.detector == 'sapiens2' else
                    ROOT/f'research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/{side}/pmpose/raw_predictions.json')
        if body_paths[side] != expected.resolve():
            raise ValueError('scene/body input mismatch')
    if any(sources[k] for k in ('old_stage_jsonl_read', 'old_dynamic_ground_read', 'old_walker_pose_read')):
        raise ValueError('scene contains old dynamic inputs')
    from pose_app.fisheye_camera import load_stereo_fisheye
    from pose_app import body_observations as raw
    for side in ('left', 'right'):
        if sorted(raw.raw_side(body_paths[side], side)) != list(range(448)):
            raise ValueError('expected complete detector full448 input')
    wrist_path = ROOT/'research_records/engineering_validation/G20261001_annotated_shared_grip_v1/wrist10_audit_v2_corrected/wrist_targets.json'
    cal_dir = ROOT/'realtime_app/calibration/results'
    wrist, wrist_audit = load_reference(wrist_path, cal_dir, True)
    surface_sets = ROOT/'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json'
    if not surface_sets.exists():
        raise FileNotFoundError(surface_sets)
    cal = load_stereo_fisheye(cal_dir)
    if np.any((wrist @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm/1000)[:,2] <= 0):
        raise ValueError('manual wrist reference behind camera1')
    # Settings only. This JSON is a command list, not a fitted result.
    settings = ROOT/'research_records/engineering_validation/G20261001_fixed_mano_pca448_v1/fit_shared_args.json'
    command = [sys.executable, '-u', *read(settings)]
    for flag, value in {'--left-raw':body_paths['left'],
                        '--right-raw':body_paths['right'],
                        '--output-dir':output/'fit'}.items():
        command[command.index(flag)+1] = str(value)
    command += ['--wrist-reference', str(wrist_path), '--wrist-reference-weight', '1',
                '--allow-diagnostic-wrist-reference']
    command[command.index('--joint-steps')+1] = str(a.joint_steps)
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(frames=448, settings_source=str(settings), command=command,
        body_source=f'{a.detector} only, existing full448 detections', wrist_reference=wrist_audit,
        initialization=dict(beta='zeros; optimized B/C', root='current detector shoulder/hip basis',
            translation='current detector hips minus rotated template hips', body='zero VPoser latent, optimized A/C',
            hands='native WiLoR local rotations re-associated to current detector and re-encoded; optimized shared PCA D1/D2'),
        old_fitted_parameters_consumed=False, old_foot_labels_consumed=False,
        constructed_grasp_consumed=False, wrist_weight=1., wrist_loss='mean squared Euclidean metres',
        stages=f'30 A + 30 B + {a.joint_steps} C + 100 D1 + 100 D2; D3 off as in settings source',
        scope='first cold body/shape/hand stage; no surface contact optimization or PMPose pose guards',
        comparison='Not a one-factor comparison with the previous frozen-beta refinement; initialization and wrist reference differ.',
        stopping='original initialization gate; stop on missing frames, nonfinite loss/output or failed input audit; retain all geometry failures',
        evidence='engineering candidate; manual wrist references remain rejected by their original geometry gate',
        accepted_for_main_fit=False))

    def execute(name, cmd):
        write(output/f'{name}_command.json', cmd)
        print('starting', name, flush=True)
        with (output/f'{name}.log').open('w', encoding='utf-8') as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        print('completed', name, flush=True)

    try:
        execute('fit', command)
        fit = output/'fit'
        with np.load(fit/'result.npz', allow_pickle=False) as archive:
            z = {k: archive[k] for k in archive.files}
        init = np.load(fit/'cold_initialization.npz', allow_pickle=False)
        if not np.array_equal(z['pair_id'], np.arange(448)) or z['vertices'].shape != (448,6890,3):
            raise ValueError('incomplete cold fit')
        for key in ('vertices','betas','global_orient','transl','vposer_latent','left_hand_pca','right_hand_pca'):
            if not np.isfinite(z[key]).all(): raise ValueError('nonfinite '+key)
        if np.any(init['betas']) or np.any(init['vposer_latent']):
            raise ValueError('not zero beta/latent initialization')
        if np.array_equal(z['betas'],init['betas']): raise ValueError('beta did not update')
        s = np.load(scene/'scene_transforms.npz', allow_pickle=False)
        z['vertices_ground_m'] = np.einsum('nij,nvj->nvi',s['rotation_ground_from_left'],z['vertices']) + s['translation_ground_from_left_mm'][:,None]/1000
        np.savez_compressed(fit/'result.npz', **z)
        write(fit/'run_metadata.json', dict(cold_start=True, detector=a.detector, allow_diagnostic_wrist_reference=True,
            inputs=dict(left_raw=str(body_paths['left']),
                        right_raw=str(body_paths['right']),
                        wrist_reference=str(wrist_path), calibration_dir=str(cal_dir)),
            old_fitted_parameters_consumed=False, accepted_for_main_fit=False))
        execute('hand_geometry_audit', [sys.executable, str(ROOT/'realtime_app/tools/audit_constructed_grasp_body.py'),
            '--result', str(fit/'result.npz'), '--walker-model', str(scene/'static_walker_model.json'),
            '--output', str(fit/'grasp_geometry_audit.json')])
        # Same model-topology vertex sets; no old body coordinates are read.
        execute('viewer_build', [sys.executable, str(ROOT/'realtime_app/tools/build_grasp_body_canvas_viewer.py'),
            '--result', str(fit/'result.npz'), '--scene', str(scene), '--surface-sets', str(surface_sets),
            '--geometry-audit', str(fit/'grasp_geometry_audit.json'), '--output', str(output/'body_viewer.html')])
        error = np.linalg.norm(z['predicted_coco']-z['raw_triangulated_points']/1000,axis=-1)
        wrist_error = np.linalg.norm(z['smplh_joints'][:,[20,21]]-wrist,axis=-1)*1000
        summary = read(fit/'fit_summary.json')
        sets = read(surface_sets)['sets']
        feet = {side:float(z['vertices_ground_m'][:,np.unique(sum(sets[side+'_sole_surface_candidate'].values(),[])),2].min()*1000) for side in ('left','right')}
        write(output/'assessment.json', dict(frames=448, beta_initial=init['betas'].tolist(), beta_final=z['betas'].tolist(),
            body_3d_rms_mm=float(np.sqrt(np.mean(error[z['body_accepted']]**2))*1000),
            wrist_reference_error_mm={side:dict(median=float(np.median(wrist_error[:,i])),p95=float(np.percentile(wrist_error[:,i],95)),max=float(wrist_error[:,i].max())) for i,side in enumerate(('left','right'))},
            sole_minimum_z_mm=feet, hand_geometry=read(fit/'grasp_geometry_audit.json')['summary'],
            stage_schedule=summary['stage_schedule'], accepted_for_main_fit=False,
            conclusion='Full cold fit complete; fixed optimization budget is not convergence certification. No physical contact or accuracy claim.'))
    except Exception as e:
        write(output/'FAILURE.json', dict(error=repr(e), outputs_preserved=True, accepted_for_main_fit=False))
        raise


if __name__ == '__main__':
    main()
