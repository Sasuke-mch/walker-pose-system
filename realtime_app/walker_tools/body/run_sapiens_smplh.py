"""Run the existing full448 SMPL-H refinement with Sapiens body observations.

Reuse the original WiLoR-derived source/reference and constructed wrist/hand
targets. All fitting and rendering are delegated to existing project entries.
"""

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


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, reference, output = (x.resolve() for x in (args.run,args.reference_run,args.output))
    if output.exists():
        raise FileExistsError(output)
    if read(run/'protocol.json')['pairs'] != 448:
        raise ValueError('expected the audited full448 observation protocol')
    metadata = read(reference/'fit/run_metadata.json')
    original = read(reference/'fit_command.json')
    scene = run/'sapiens2/scene'
    sources = read(scene/'scene_sources.json')
    for side in ('left','right'):
        if Path(sources[f'raw_{side}_pmpose']).resolve() != run/f'{side}_sapiens_coco17_model_input.json':
            raise ValueError('scene is not from the current Sapiens observations')
    if any(sources[k] for k in ('old_stage_jsonl_read','old_dynamic_ground_read','old_walker_pose_read')):
        raise ValueError('old dynamic input in Sapiens scene')
    source = np.load(metadata['inputs']['source_result'],allow_pickle=False)
    transforms = np.load(scene/'scene_transforms.npz',allow_pickle=False)
    if source['vertices'].shape!=(448,6890,3) or transforms['rotation_ground_from_left'].shape!=(448,3,3):
        raise ValueError('not the complete full448 source and scene')
    grasp = read(Path(metadata['inputs']['grasp']))
    if not np.allclose(source['betas'],grasp['betas'],atol=1e-6):
        raise ValueError('frozen shape and wrist/hand targets differ')
    sets = read(Path(metadata['inputs']['contact_vertex_sets']))['sets']
    vg = np.einsum('nij,nvj->nvi',transforms['rotation_ground_from_left'],source['vertices'])+transforms['translation_ground_from_left_mm'][:,None]/1000
    height = np.stack([vg[:,np.unique(sum(sets[s+'_sole_surface_candidate'].values(),[])),2].min(1) for s in ('left','right')],1)
    output.mkdir(parents=True)
    labels = output/'current_foot_candidates.npz'
    # Same equation and immutable source as the PMPose run_full448.py.
    np.savez_compressed(labels,foot_contact_weight=(1/(1+np.exp((np.abs(height)-.03)/.015))).astype(np.float32),
                        baseline_surface_min_z=height,pair_id=np.arange(448),
                        provenance='same immutable source transformed by current Sapiens scene; engineering candidate')
    replacements = {'--left-raw':str(run/'left_sapiens_coco17_model_input.json'),
                    '--right-raw':str(run/'right_sapiens_coco17_model_input.json'),
                    '--scene-transforms':str(scene/'scene_transforms.npz'),
                    '--walker-model':str(scene/'static_walker_model.json'),
                    '--contact-labels':str(labels),'--output-dir':str(output/'fit')}
    command = list(original)
    command[0] = sys.executable
    for flag,value in replacements.items():
        command[command.index(flag)+1] = value
    changed = {original[i-1]:{'before':a,'after':b} for i,(a,b) in enumerate(zip(original,command)) if i>0 and a!=b}
    if set(changed)-set(replacements):
        raise ValueError('unexpected change to PMPose command')
    write(output/'protocol.json',dict(input_run=str(run),reference_run=str(reference),frames=448,
        original_command=original,actual_command=command,changed_arguments=changed,
        frozen_inputs={k:metadata['inputs'][k] for k in ('source_result','pose_reference_result','grasp','contact_vertex_sets','vposer_dir')},
        wilor_provenance=str(ROOT/'research_records/engineering_validation/G20261001_fixed_mano_pca448_v1/fit_shared_args.json'),
        interpretation='Conditional detector substitution from the same PMPose/WiLoR-derived initialization; not independently initialized detector ranking.',
        hand_reuse='Existing constructed wrist transforms and PCA targets plus original WiLoR-derived shared source; no WiLoR inference or target redesign.',
        scene_policy='Current Sapiens replay; foot candidates recomputed by original equation; end-to-end replacement, not fixed-scene ablation.',
        analysis_correction='Strict stereo consistency does not rank 2D accuracy or establish SMPL-H suitability; earlier categorical rejection withdrawn.',
        scope='User requested full448 SMPL-H visualization despite prior observation-gate failures; retain all quality failures.',
        accepted_for_main_fit=False))

    def execute(name, cmd):
        write(output/f'{name}_command.json',cmd)
        print('starting',name,flush=True)
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('completed',name,flush=True)

    execute('fit',command)
    execute('hand_geometry_audit',[sys.executable,str(ROOT/'realtime_app/tools/audit_constructed_grasp_body.py'),
        '--result',str(output/'fit/result.npz'),'--walker-model',str(scene/'static_walker_model.json'),
        '--output',str(output/'fit/grasp_geometry_audit.json')])
    execute('viewer_build',[sys.executable,str(ROOT/'realtime_app/tools/build_grasp_body_canvas_viewer.py'),
        '--result',str(output/'fit/result.npz'),'--scene',str(scene),
        '--surface-sets',metadata['inputs']['contact_vertex_sets'],
        '--geometry-audit',str(output/'fit/grasp_geometry_audit.json'),'--output',str(output/'body_viewer.html')])
    html = output/'body_viewer.html'
    page = html.read_text(encoding='utf-8').replace('SMPL-H 固定手部约束拟合','Sapiens2 · SMPL-H 固定手部约束拟合')
    # Display-only: keep the finite ground plane visible against the light sky.
    # The original 12m plane covered the entire viewport at the default view.
    page = page.replace('project([-6,-6,0]),project([6,-6,0]),project([6,6,0]),project([-6,6,0])',
                        'project([-2.5,-2.5,0]),project([2.5,-2.5,0]),project([2.5,2.5,0]),project([-2.5,2.5,0])')
    page = page.replace("for(let k=-6;k<=6;k++){seg([k,-6,0],[k,6,0],'#b9c8c0',1);seg([-6,k,0],[6,k,0],'#b9c8c0',1)}",
                        "for(let k=-2.5;k<=2.5;k+=.5){seg([k,-2.5,0],[k,2.5,0],'#b9c8c0',1);seg([-2.5,k,0],[2.5,k,0],'#b9c8c0',1)}")
    html.write_text(page,encoding='utf-8')
    z = np.load(output/'fit/result.npz')
    summary = read(output/'fit/fit_summary.json')
    audit = read(output/'fit/grasp_geometry_audit.json')
    guard = np.load(output/'fit/observation_guard_reference.npz')
    errors = np.linalg.norm(z['predicted_coco']-z['raw_triangulated_points']/1000,axis=-1)
    excess = float((errors-guard['baseline_error_m'])[guard['valid']].max()*1000)
    assert np.array_equal(z['pair_id'],np.arange(448))
    assert z['vertices'].shape==(448,6890,3) and z['faces'].shape==(13776,3)
    assert z['faces'].min()>=0 and z['faces'].max()<6890
    assert all(np.isfinite(z[k]).all() for k in ('vertices','predicted_coco','vposer_latent','body_rotation_matrices'))
    initial = summary['checkpoints']['initial'];final = list(summary['checkpoints'].values())[-1]
    gates = dict(observations=excess<=10.002,wrists=all(v['p95']<=10 for v in final['wrist_error_mm'].values()),
                 feet=all(v>=-3 for v in final['sole_minimum_z_mm'].values()),
                 leg_temporal=final['terms']['leg_ground_acceleration']<=initial['terms']['leg_ground_acceleration'],
                 hands=all(r['passed_geometry_proxy'] for r in audit['records']))
    write(output/'assessment.json',dict(frames=448,finite=True,vertices_shape=list(z['vertices'].shape),faces_shape=list(z['faces'].shape),
        maximum_observation_regression_mm=excess,checkpoints=summary['checkpoints'],hand_geometry=audit['summary'],
        gates=gates,passed=all(gates.values()),accepted_for_main_fit=False,
        boundary='Rendering completion is not physical or external 2D/3D accuracy validation.'))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':
    main()
