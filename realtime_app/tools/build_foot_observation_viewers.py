"""Two minimal saved-result viewers sharing one fresh same-image scene replay."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from walker_tools.visualization._scene_export import finite, packed, prepare_scene, save
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_ground_constraints import load_sole_vertex_indices
from pose_app.smplh_motion_tracker import SMPLHObservationMap
from probe_smplh_foot_observations import OUT, REPAIR, ROOT
from pose_app.replay_pair_timing import load_capture_pair_times as _read_capture_times


def export(run, title, rejection, shared, *, target=None, result_override=None, rows_override=None, frame_ids=None, write_page=True):
    target = target or OUT / run / 'view'
    target.mkdir(exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError(f'Refuse to overwrite populated view: {target}')
    source_protocol = json.loads((REPAIR / 'hip_operator_feedback/protocol.json').read_text(encoding='utf-8'))
    p = json.loads((Path(source_protocol['source_run']) / 'run_parameters.json').read_text(encoding='utf-8'))
    inp = p['inputs']
    result = np.load(OUT / run / 'result.npz') if result_override is None else result_override
    count = len(result['timestamp_s'])
    ids = np.arange(count) if frame_ids is None else np.asarray(frame_ids, dtype=int)
    if ids.shape != (count,) or np.any(np.diff(ids) <= 0):
        raise ValueError('Saved frame indices must be ordered and unique')
    scene_dir = shared / 'scene'
    sources = json.loads((scene_dir / 'scene_sources.json').read_text(encoding='utf-8'))
    if any(sources.get(k) for k in ['old_stage_jsonl_read', 'old_dynamic_ground_read', 'old_dynamic_walker_pose_read']):
        raise ValueError('Historical dynamic scene inputs are forbidden')
    if Path(sources['input_pair_dir']).resolve() != Path(inp['input_pairs']).resolve():
        raise ValueError('Body/scene source images differ')
    for side in ['left', 'right']:
        if Path(sources[f'raw_{side}_pmpose']).resolve() != Path(inp[f'{side}_predictions']).resolve():
            raise ValueError('Body/scene PMPose rows differ')
    scene = np.load(scene_dir / 'scene_transforms.npz')
    R = scene['rotation_ground_from_left'][ids]
    t = scene['translation_ground_from_left_mm'][ids] / 1000
    times = _read_capture_times(Path(inp['input_pairs']), 448)[0][ids]
    if not np.allclose(times, result['timestamp_s'], atol=1e-7, rtol=0) or not np.allclose(times, scene['pair_timestamp_sec'][ids], atol=1e-7, rtol=0):
        raise ValueError('Scene/body timestamps differ')
    def xf(points):
        return np.einsum('tij,tkj->tki', R, points) + t[:, None]
    model = SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ['smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right']], device='cuda')
    torch = model.torch
    torch.set_num_threads(2)
    if not np.array_equal(result['betas'], model.betas.cpu().numpy()):
        raise ValueError('Saved shape differs from source shape')
    meshes, natives = [], []
    for start in range(0, count, 32):
        stop = min(start + 32, count)
        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device=model.device)
        with torch.inference_mode():
            body = model.model(betas=model.betas.expand(stop - start, -1),
                transl=tensor(result['transl'][start:stop]), global_orient=tensor(result['root_rotation'][start:stop]),
                body_pose=tensor(result['body_rotations'][start:stop]),
                left_hand_pose=model.hand_rotations[0].expand(stop - start, -1, -1, -1),
                right_hand_pose=model.hand_rotations[1].expand(stop - start, -1, -1, -1))
        meshes.append(body.vertices.cpu().numpy())
        natives.append(body.joints[:, :52].cpu().numpy())
    mesh, native = np.concatenate(meshes), np.concatenate(natives)
    if mesh.shape != (count, 6890, 3) or model.faces.shape != (13776, 3):
        raise ValueError('Real SMPL-H topology required')
    rebuilt = np.einsum('jv,tvc->tjc', result['regressor'], mesh)
    early = ids <= 10
    rebuilt[early] = np.einsum('jv,tvc->tjc', model.regressor.cpu().numpy(), mesh[early])
    errors = {'coco': float(np.max(abs(rebuilt - result['predicted_coco'])))}
    if 'smplh_joints' in result:
        errors['native'] = float(np.max(abs(native - result['smplh_joints'])))
    if max(errors.values()) > 2e-6:
        raise ValueError(f'Saved pose/mesh disagree: {errors}')
    sole = load_sole_vertex_indices(APP.parent / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')
    sole_world = xf(mesh[:, np.concatenate(sole)])
    heights = np.stack([sole_world[:, :len(sole[0]), 2].min(axis=1), sole_world[:, len(sole[0]):, 2].min(axis=1)], axis=1) * 1000
    rows = rows_override if rows_override is not None else [json.loads(line) for line in (OUT / run / 'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
    # Existing comparison pixels are camera pixels; geometry below uses only this
    # task's newly replayed Stage/ground. No old ground or webpage mesh is read.
    from probe_smplh_foot_observations import project
    from pose_app.body_observations import load_ordered_predictions as _load_predictions
    cal = load_stereo_fisheye(inp['calibration_dir'])
    observed = np.stack([_load_predictions(Path(inp[f'{s}_predictions']), s) for s in ['left', 'right']], axis=1)
    observed[0] = np.asarray(json.loads(Path(p['first_frame_observations']).read_text(encoding='utf-8'))['keypoints'])
    observed = observed[ids]
    pixels = project(result['predicted_coco'], cal)
    entry = {'name': 'repaired', 'label': title, 'mesh': packed(xf(mesh)), 'points': packed(xf(result['predicted_coco'])),
        'sole': packed(sole_world), 'soleHeight': finite(heights),
        'errors': finite(np.linalg.norm(pixels[:, :, 5:17] - observed[:, :, 5:17, :2], axis=-1)),
        'status': [[r['optimizer']['status']] * 12 for r in rows],
        'reason': [[r['optimizer'].get('termination', '首帧')] * 12 for r in rows],
        'pixels': finite(pixels), 'rejection': rejection,
        'landmarkDefinition': ['原COCO髋映射' if i <= 10 else '冻结虚拟观察髋；非解剖中心' for i in ids]}
    walker = json.loads((scene_dir / 'static_walker_model.json').read_text(encoding='utf-8'))
    names = sorted({n for edge in walker['edges'] for n in edge})
    wr = np.asarray(walker['rotation_left_camera_from_walker'])
    wt = np.asarray(walker['translation_left_camera_from_walker_mm']) / 1000
    nodes = np.asarray([walker['nodes_walker_mm'][n] for n in names]) / 1000 @ wr.T + wt
    right_center = -cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm / 1000)
    stages = np.asarray([json.loads(line)['stage']['operational'] for line in (scene_dir / 'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines()])[ids].tolist()
    poses = [json.loads(line)['pose_audit'] for line in (scene_dir / 'dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines()]
    poses = [poses[i] for i in ids]
    strict_rows = [json.loads(line) for line in (shared / 'strict.jsonl').read_text(encoding='utf-8').splitlines()]
    strict_rows = [strict_rows[i] for i in ids]
    raw_reasons = []
    for row in strict_rows:
        reasons = ['association_rejected'] * 17
        if row['persons_3d']:
            for point in row['persons_3d'][0]['keypoints_3d']:
                reasons[point['index']] = point.get('reason') or 'accepted'
        raw_reasons.append(reasons)
    # Both candidates share frame0; use it for identical fixed framing.
    center = np.median(xf(result['predicted_coco'])[:1, 11:13, :2].reshape(-1, 2), axis=0)
    payload = {'frames': count, 'frameIds': ids.tolist(), 'time': times.tolist(), 'models': [entry], 'title': title, 'rejection': rejection,
        'faces': model.faces.tolist(), 'walker': packed(xf(np.broadcast_to(nodes, (count, len(names), 3)))),
        'walkerNames': names, 'walkerEdges': [[names.index(a), names.index(b)] for a, b in walker['edges']],
        'cameras': packed(xf(np.broadcast_to(np.stack([np.zeros(3), right_center]), (count, 2, 3)))),
        'raw': packed(xf(scene['detector_only_triangulated_points_mm'][ids] / 1000)),
        'strict': packed(xf(scene['raw_triangulated_points_mm'][ids] / 1000)),
        'accepted': scene['accepted'][ids].tolist(), 'stage': stages, 'pose': [p['status'] for p in poses],
        'poseReason': [p.get('reason') for p in poses], 'center': center.tolist(), 'soleCount': sum(map(len, sole)),
        'rawReasons': raw_reasons}
    compressed = gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()) if write_page else b''
    if write_page:
        (target / 'data.json.gz').write_bytes(compressed)
        templates = APP / 'walker_tools/visualization'
        html = (templates / 'smplh_repair_comparison.html').read_text(encoding='utf-8')
        html = html.replace('修正后候选', '实验候选').replace('max="447"', f'max="{count-1}"')
        (target / 'index.html').write_text(html, encoding='utf-8')
        (target / 'viewer.js').write_text((templates / 'smplh_minimal_view.js').read_text(encoding='utf-8'), encoding='utf-8')
    metadata = {'frames': count, 'source_result': str(OUT / run / 'result.npz'), 'fresh_shared_scene': str(scene_dir),
        'raw_inputs': inp, 'tracking_rerun': False, 'fitting_rerun': False, 'old_dynamic_inputs_consumed': False,
        'geometry_max_difference_m': errors, 'real_topology': {'vertices': 6890, 'faces': 13776, 'index_entries': 41328},
        'frame_ids': ids.tolist(), 'same_task_scene_prefix': bool(np.array_equal(ids, np.arange(62))), 'smoothing': False, 'interpolation': False, 'coordinate_map': 'X,-Y,Z',
        'rejection': rejection, 'payload_bytes': len(compressed), 'page': str(target / 'index.html'), 'browser_status': 'pending'}
    if write_page:
        save(target / 'metadata.json', metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return payload, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    shared = OUT / 'visualization_20261016'
    if not args.export_only:
        prepare_scene(ROOT / 'skeleton_tracking_20261016', shared)
    export('toe_pixels_full448', '脚趾位置约束 · 448帧', '未通过：全片非髋残差中位数增加9.32%。', shared)
    export('toe_direction_prefix', '前掌方向约束 · 62帧', '未通过：35–61帧髋残差中位数增加7.79%。', shared)
