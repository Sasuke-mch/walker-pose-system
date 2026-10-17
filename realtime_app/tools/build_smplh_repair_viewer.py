"""Display saved hip/feedback repair results with a fresh same-image scene."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_BOOTSTRAP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_BOOTSTRAP), str(_BOOTSTRAP / 'tools')]
from walker_tools.visualization._scene_export import (  # noqa: E402
    APP,
    REPO,
    finite,
    packed,
    prepare_scene,
    save,
)
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
from pose_app.smplh_ground_constraints import load_sole_vertex_indices
from pose_app.smplh_motion_tracker import SMPLHObservationMap
from pose_app.body_observations import load_ordered_predictions as _load_predictions
from pose_app.replay_pair_timing import load_capture_pair_times as _read_capture_times
from run_stereo_pva_baseline import JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    output = args.root/'view'
    if not args.export_only:
        prepare_scene(args.root.parent/'skeleton_tracking_20261016',output)
    if not args.scene_only:
        export(args.root,output)


def export(root,output):
    root,output = root.resolve(),output.resolve()
    proposal = root/'hip_operator_feedback_full448'
    proposal_protocol = json.loads((proposal/'protocol.json').read_text(encoding='utf-8'))
    source = Path(proposal_protocol['source_run'])
    p = json.loads((source/'run_parameters.json').read_text(encoding='utf-8'))
    inp = p['inputs']
    scene_dir = output/'scene'
    scene_sources = json.loads((scene_dir/'scene_sources.json').read_text(encoding='utf-8'))
    if any(scene_sources.get(k) for k in ('old_stage_jsonl_read','old_dynamic_ground_read','old_dynamic_walker_pose_read')):
        raise ValueError('old dynamic scene inputs are forbidden')
    if any(Path(scene_sources[f'raw_{side}_pmpose']).resolve()!=Path(inp[f'{side}_predictions']).resolve() for side in ('left','right')):
        raise ValueError('scene/body observations differ')
    if Path(scene_sources['input_pair_dir']).resolve()!=Path(inp['input_pairs']).resolve():
        raise ValueError('scene/body image rows differ')
    with np.load(scene_dir/'scene_transforms.npz') as a:
        scene = {k:a[k].copy() for k in ('rotation_ground_from_left','translation_ground_from_left_mm',
            'pair_timestamp_sec','detector_only_triangulated_points_mm','raw_triangulated_points_mm','accepted')}
    R,t = scene['rotation_ground_from_left'],scene['translation_ground_from_left_mm']/1000
    def xf(points):
        return np.einsum('tij,tkj->tki',R,points)+t[:,None]
    times = _read_capture_times(Path(inp['input_pairs']),448)[0]
    if not np.allclose(times,scene['pair_timestamp_sec'],atol=1e-7,rtol=0):
        raise ValueError('scene/body capture times differ')
    observed = np.stack([_load_predictions(Path(inp[f'{s}_predictions']),s) for s in ('left','right')],axis=1)
    observed[0] = np.asarray(json.loads(Path(p['first_frame_observations']).read_text(encoding='utf-8'))['keypoints'])
    cal = load_stereo_fisheye(inp['calibration_dir'])
    original = np.load(source/'result.npz')
    candidate = np.load(proposal/'result.npz')
    if not np.array_equal(original['betas'],candidate['betas']):
        raise ValueError('before/after frozen shapes differ')
    if not np.allclose(candidate['timestamp_s'],times,atol=1e-7,rtol=0):
        raise ValueError('candidate frame times differ')
    model = SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model','regressor','vposer_dir','mano_left','mano_right')],device='cuda')
    torch = model.torch
    torch.set_num_threads(2)
    faces = original['faces']
    if faces.shape!=(13776,3) or not np.array_equal(faces,model.faces):
        raise ValueError('real mesh faces do not match saved-pose model')
    # Reconstruct geometry from saved rotations/root/shape only. No optimizer,
    # image inference, VPoser decoding or motion-state update is rerun.
    meshes,natives = [],[]
    for start in range(0,448,32):
        stop = min(start+32,448)
        def tensor(a):
            return torch.as_tensor(a,dtype=torch.float32,device=model.device)
        with torch.inference_mode():
            result = model.model(betas=model.betas.expand(stop-start,-1),transl=tensor(candidate['transl'][start:stop]),
                global_orient=tensor(candidate['root_rotation'][start:stop]),body_pose=tensor(candidate['body_rotations'][start:stop]),
                left_hand_pose=model.hand_rotations[0].expand(stop-start,-1,-1,-1),
                right_hand_pose=model.hand_rotations[1].expand(stop-start,-1,-1,-1))
        meshes.append(result.vertices.cpu().numpy())
        natives.append(result.joints[:,:52].cpu().numpy())
    mesh = np.concatenate(meshes)
    native = np.concatenate(natives)
    if mesh.shape!=(448,6890,3):
        raise ValueError('full448 mesh reconstruction is required')
    base_reg = model.regressor.cpu().numpy()
    rebuilt = np.einsum('jv,tvc->tjc',candidate['regressor'],mesh)
    rebuilt[:11] = np.einsum('jv,tvc->tjc',base_reg,mesh[:11])
    mapping_error = float(np.max(abs(rebuilt-candidate['predicted_coco'])))
    native_error = float(np.max(abs(native-candidate['smplh_joints'])))
    if mapping_error>2e-6 or native_error>2e-6:
        raise ValueError(f'saved pose/geometry disagree: COCO={mapping_error},native={native_error}')
    sole_file = REPO/'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json'
    sole = load_sole_vertex_indices(sole_file)
    sole_ids = np.concatenate(sole)
    def project(points):
        other = points@cal.R_cam0_to_cam1.T+cal.T_cam0_to_cam1_mm/1000
        return np.stack([fisheye_project_numpy(points,cal.K0,cal.D0),fisheye_project_numpy(other,cal.K1,cal.D1)],axis=1)
    models = []
    for path,a,vertices,label,color,name in ((proposal,candidate,mesh,'修正后 · 信息反馈 + 髋映射','#00786d','repaired'),
            (source,original,original['vertices'],'修正前 · 原MAP','#bd6527','original')):
        rows = [json.loads(line) for line in (path/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
        coco = a['predicted_coco']
        pixels = project(coco)
        opt = [r['optimizer'] if name=='repaired' else r['iterated_update'] for r in rows]
        status = [[o['status']]*12 for o in opt]
        reason = [[o.get('termination','首帧')]*12 for o in opt]
        joints = xf(a['smplh_joints'][:,:3])
        sole_world = xf(vertices[:,sole_ids])
        heights = np.stack([sole_world[:,sum(len(x) for x in sole[:s]):sum(len(x) for x in sole[:s+1]),2].min(axis=1)*1000 for s in range(2)],axis=1)
        points = xf(coco)
        points[~a['pose_available']] = np.nan
        entry = {'name':name,'label':label,'color':color,'type':'smplh','points':packed(points),'pixels':finite(pixels),
            'errors':finite(np.linalg.norm(pixels[:,:,5:17]-observed[:,:,5:17,:2],axis=-1)),
            'status':status,'reason':reason,'segments':np.repeat(a['segment_id'][:,None],12,axis=1).tolist(),
            'native':packed(joints),'sole':packed(sole_world),'soleHeight':finite(heights),
            'informationUpdates':[o.get('posterior_updates',0) for o in opt],
            'landmarkDefinition':['原COCO髋映射' if name=='original' or i<=10 else '校准后虚拟观察髋；不是解剖中心' for i in range(448)]}
        if name=='repaired':
            entry['mesh'] = packed(xf(vertices))
        models.append(entry)
    walker = json.loads((scene_dir/'static_walker_model.json').read_text(encoding='utf-8'))
    names = sorted({n for edge in walker['edges'] for n in edge})
    wr = np.asarray(walker['rotation_left_camera_from_walker'])
    wt = np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    nodes = np.asarray([walker['nodes_walker_mm'][n] for n in names])/1000@wr.T+wt
    walker_world = xf(np.broadcast_to(nodes,(448,len(names),3)))
    right_center = -cal.R_cam0_to_cam1.T@(cal.T_cam0_to_cam1_mm/1000)
    cameras = xf(np.broadcast_to(np.stack([np.zeros(3),right_center]),(448,2,3)))
    stages = [json.loads(line)['stage']['operational'] for line in (scene_dir/'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines()]
    poses = [json.loads(line)['pose_audit'] for line in (scene_dir/'dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines()]
    # Fresh strict points drive their trails. All finite raw points drive black
    # diagnostic bones even when rejected by the strict geometric gate.
    raw = xf(scene['detector_only_triangulated_points_mm']/1000)
    strict = xf(scene['raw_triangulated_points_mm']/1000)
    strict_rows = [json.loads(line) for line in (output/'strict.jsonl').read_text(encoding='utf-8').splitlines()]
    raw_reasons = []
    for row in strict_rows:
        reasons = ['association_rejected']*17
        if row['persons_3d']:
            for point in row['persons_3d'][0]['keypoints_3d']:
                reasons[point['index']] = point.get('reason') or 'accepted'
        raw_reasons.append(reasons)
    center = np.median(xf(candidate['predicted_coco'])[:,11:13,:2].reshape(-1,2),axis=0)
    payload = {'frames':448,'time':times.tolist(),'models':models,'names':JOINT_NAMES,'referenceIndex':1,'repairView':True,
        'mesh':packed(xf(original['vertices'])),'faces':faces.tolist(),'walker':packed(walker_world),'walkerNames':names,
        'walkerEdges':[[names.index(a),names.index(b)] for a,b in walker['edges']], 'cameras':packed(cameras),
        'raw':packed(raw),'strict':packed(strict),'accepted':scene['accepted'].tolist(),'observed':finite(observed),
        'stage':stages,'pose':[r['status'] for r in poses],'poseReason':[r.get('reason') for r in poses],
        'center':center.tolist(),'imageRoot':os.path.relpath(inp['input_pairs'],output).replace('\\','/'),
        'soleCount':len(sole_ids),'soleLengths':[len(x) for x in sole],
        'rawReasons':raw_reasons}
    compressed = gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
    (output/'data.json.gz').write_bytes(compressed)
    templates = APP/'walker_tools/visualization'
    (output/'index.html').write_text((templates/'smplh_repair_comparison.html').read_text(encoding='utf-8'),encoding='utf-8')
    (output/'viewer.js').write_text((templates/'smplh_minimal_view.js').read_text(encoding='utf-8'),encoding='utf-8')
    # Only a post-export numerical comparison reads the previous reference.
    old = np.load(root.parent/'slope_motion_20261015/scene/scene_transforms.npz')
    metadata = {'frames':448,'source_original':str(source),'source_candidate':str(proposal),'new_scene':str(scene_dir),
        'old_dynamic_inputs_consumed':False,'tracking_rerun':False,'fitting_rerun':False,
        'candidate_geometry':'Regenerated only from saved body/root rotations,translation,unchanged beta and frozen hand poses; no optimization.',
        'geometry_max_difference_m':{'coco':mapping_error,'native':native_error},'real_topology':{'vertices':6890,'faces':13776,'index_entries':int(faces.size)},
        'fresh_vs_previous_reference_only':{'max_R_entry_difference':float(abs(R-old['rotation_ground_from_left']).max()),
            'max_translation_mm_difference':float(abs(t*1000-old['translation_ground_from_left_mm']).max())},
        'coordinate_map':'X,-Y,Z for all scene objects','smoothing':False,'interpolation':False,
        'hip_definition':'Display virtual observation hips separately from native pelvis0 and native hip centers1/2; adaptation begins frame11.',
        'sole_points_per_foot':[len(x) for x in sole],'payload_bytes':len(compressed),'page':str(output/'index.html'),
        'raw_joint_counts':[{'index':j,'finite':int(np.isfinite(raw[:,j]).all(axis=1).sum()),
            'accepted':int(scene['accepted'][:,j].sum()),'reason_counts':dict(Counter(row[j] for row in raw_reasons))} for j in range(17)],
        'command':[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--export-only'],'browser_status':'pending'}
    save(output/'metadata.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
