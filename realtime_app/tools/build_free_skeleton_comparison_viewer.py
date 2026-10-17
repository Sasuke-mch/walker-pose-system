"""Export saved free-skeleton/SMPL-H results with a fresh same-image scene."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy  # noqa: E402
from pose_app.body_observations import load_ordered_predictions as _load_predictions  # noqa: E402
from pose_app.replay_pair_timing import load_capture_pair_times as _read_capture_times  # noqa: E402
from run_stereo_pva_baseline import JOINT_NAMES  # noqa: E402


from walker_tools.visualization._scene_export import finite, packed, prepare_scene, save  # noqa: E402


def export(root, output):
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    inputs = protocol['tracking_inputs']
    source = Path(protocol['source_run_for_paths_and_later_comparison'])
    scene_dir = output/'scene'
    sources = json.loads((scene_dir/'scene_sources.json').read_text(encoding='utf-8'))
    if any(sources.get(key) for key in ('old_stage_jsonl_read', 'old_dynamic_ground_read', 'old_dynamic_walker_pose_read')):
        raise ValueError('old dynamic inputs consumed by scene')
    if any(Path(sources[f'raw_{side}_pmpose']).resolve() != Path(inputs[f'{side}_predictions']).resolve() for side in ('left', 'right')):
        raise ValueError('scene/track observations differ')
    if Path(sources['input_pair_dir']).resolve() != Path(inputs['input_pairs']).resolve():
        raise ValueError('scene/track images differ')
    if Path(sources['calibration']).resolve() != (Path(inputs['calibration_dir'])/'stereo_fisheye.json').resolve():
        raise ValueError('scene/track calibration differs')
    with np.load(scene_dir/'scene_transforms.npz', allow_pickle=False) as archive:
        scene = {k: archive[k].copy() for k in ('rotation_ground_from_left', 'translation_ground_from_left_mm',
            'detector_only_triangulated_points_mm', 'raw_triangulated_points_mm', 'accepted', 'pair_timestamp_sec')}
    R, t = scene['rotation_ground_from_left'], scene['translation_ground_from_left_mm']/1000
    xf = lambda a: np.einsum('tij,tkj->tki', R, a) + t[:, None]
    observations = np.stack([_load_predictions(Path(inputs[f'{side}_predictions']), side) for side in ('left', 'right')], axis=1)
    frame0 = json.loads(Path(protocol['first_frame_observations']).read_text(encoding='utf-8'))
    observations[0] = np.asarray(frame0['keypoints'])
    times, *_ = _read_capture_times(Path(inputs['input_pairs']), len(observations))
    if not np.allclose(scene['pair_timestamp_sec'], times, atol=1e-7, rtol=0):
        raise ValueError('scene/track times differ')
    cal = load_stereo_fisheye(Path(inputs['calibration_dir']))
    def project(points):
        other = points @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm/1000
        return np.stack([fisheye_project_numpy(points, cal.K0, cal.D0), fisheye_project_numpy(other, cal.K1, cal.D1)], axis=1)
    models = []
    for filename, label, color in (('pva_matched_pixel_sigma8', '独立骨架跟踪 · σ8', '#00786d'),
                                    ('pva_frozen_sigma4', '独立骨架跟踪 · σ4', '#245cb0')):
        with np.load(root/(filename+'.npz'), allow_pickle=False) as archive:
            if not np.allclose(archive['timestamp_s'], times, atol=1e-7, rtol=0):
                raise ValueError('skeleton/scene frame times differ')
            points = np.full((448, 17, 3), np.nan)
            points[:, 5:17] = archive['camera_points_mm']/1000
            pixels = project(points)
            errors = np.linalg.norm(pixels[:, :, 5:17]-observations[:, :, 5:17, :2], axis=-1)
            models.append({'name': filename, 'label': label, 'color': color, 'points': packed(xf(points)),
                'pixels': finite(pixels), 'errors': finite(errors), 'status': archive['status'].tolist(),
                'reason': archive['reason'].tolist(), 'segments': archive['segment_id'].tolist(), 'type': 'free'})
    with np.load(source/'result.npz', allow_pickle=False) as archive:
        if not np.allclose(archive['timestamp_s'], times, atol=1e-7, rtol=0):
            raise ValueError('body/track frame times differ')
        mesh, coco, faces = archive['vertices'].copy(), archive['predicted_coco'].copy(), archive['faces'].copy()
    if mesh.shape != (448, 6890, 3) or faces.shape != (13776, 3) or faces.max() >= 6890:
        raise ValueError('same-result real mesh topology is required')
    pixels = project(coco)
    records = [json.loads(line) for line in (source/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
    smpl_status = [row.get('iterated_update', {}).get('status', row['model_status']) for row in records]
    models.append({'name': 'smplh', 'label': 'SMPL-H · 原MAP', 'color': '#bd6527', 'points': packed(xf(coco)),
        'pixels': finite(pixels), 'errors': finite(np.linalg.norm(pixels[:, :, 5:17]-observations[:, :, 5:17, :2], axis=-1)),
        'status': [[value]*12 for value in smpl_status], 'reason': [[row.get('iterated_update', {}).get('termination', '')]*12 for row in records],
        'segments': [[row['segment_id']]*12 for row in records], 'type': 'smplh'})
    walker = json.loads((scene_dir/'static_walker_model.json').read_text(encoding='utf-8'))
    names = sorted({name for edge in walker['edges'] for name in edge})
    wr = np.asarray(walker['rotation_left_camera_from_walker'])
    wt = np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    nodes = np.asarray([walker['nodes_walker_mm'][name] for name in names])/1000 @ wr.T + wt
    walker_world = xf(np.broadcast_to(nodes, (448, len(names), 3)))
    c1 = -cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm/1000)
    cameras = xf(np.broadcast_to(np.stack([np.zeros(3), c1]), (448, 2, 3)))
    stages = [json.loads(line)['stage']['operational'] for line in (scene_dir/'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines()]
    ground = [json.loads(line)['pose_audit'] for line in (scene_dir/'dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines()]
    world_coco = xf(coco)
    center = np.nanmedian(world_coco[:, 11:13, :2].reshape(-1, 2), axis=0)
    raw_world = xf(scene['detector_only_triangulated_points_mm']/1000)
    payload = {'frames': 448, 'time': times.tolist(), 'models': models, 'names': JOINT_NAMES,
        'mesh': packed(xf(mesh)), 'faces': faces.tolist(), 'walker': packed(walker_world), 'walkerNames': names,
        'walkerEdges': [[names.index(a), names.index(b)] for a, b in walker['edges']], 'cameras': packed(cameras),
        'raw': packed(raw_world), 'accepted': scene['accepted'].tolist(), 'observed': finite(observations),
        'stage': stages, 'pose': [row['status'] for row in ground], 'poseReason': [row.get('reason') for row in ground],
        'center': center.tolist(), 'imageRoot': os.path.relpath(inputs['input_pairs'], output).replace('\\', '/'),
        'summaryImage': '../comparison.png'}
    compressed = gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    (output/'data.json.gz').write_bytes(compressed)
    template_dir = APP/'walker_tools/visualization'
    (output/'index.html').write_text((template_dir/'free_skeleton_comparison.html').read_text(encoding='utf-8'), encoding='utf-8')
    (output/'viewer.js').write_text((template_dir/'free_skeleton_comparison.js').read_text(encoding='utf-8'), encoding='utf-8')
    # This compares the newly replayed reference with the numerical experiment;
    # the preceding reference is never consumed by the displayed scene.
    old_scene = Path(protocol['reference']['scene'])/'scene_transforms.npz'
    with np.load(old_scene, allow_pickle=False) as archive:
        same_R = float(abs(R-archive['rotation_ground_from_left']).max())
        same_t = float(abs(t*1000-archive['translation_ground_from_left_mm']).max())
    metadata = {'frames': 448, 'source_tracking': str(root.resolve()), 'smplh_result': str(source/'result.npz'),
        'scene': str(scene_dir.resolve()), 'new_scene_replayed': True, 'old_dynamic_inputs_consumed': False,
        'same_ground_map': 'X,-Y,Z;all skeletons,mesh,raw points,walker and both camera centers',
        'mesh_vertices': 6890, 'mesh_faces': 13776, 'mesh_index_entries': int(faces.size),
        'raw_observation_source': 'new_scene.detector_only_triangulated_points_mm;finite rejected retained',
        'frame0_body': 'Sapiens2', 'frame0_raw_scene': 'PMPose;explicitly separate',
        'body_rerun': False, 'tracking_rerun': False, 'interpolation': False, 'display_smoothing': False,
        'reference_comparison_only': {'max_rotation_entry_difference': same_R, 'max_translation_mm_difference': same_t},
        'page': str(output/'index.html'), 'payload_bytes': len(compressed), 'browser_status': 'pending'}
    save(output/'metadata.json', metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-only', action='store_true', help='Use the fresh replay already made by this visualization task')
    args = parser.parse_args()
    if not args.export_only:
        prepare_scene(args.root.resolve(), args.output.resolve())
    export(args.root.resolve(), args.output.resolve())
