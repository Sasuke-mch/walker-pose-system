"""Same-run walker448 PhysHMR display with raw image reprojections."""

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from .physhmr_visualize import export


def build(out):
    from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
    from tools.run_stereo_pva_baseline import _load_predictions

    root = Path(__file__).resolve().parents[2]
    run = out / 'physics'
    view = out / 'view'
    export(run, root / 'third_party/PhysHMR/data/smpl', view)
    with gzip.open(view / 'data.json.gz', 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    runtime = json.loads((run / 'runtime.json').read_text(encoding='utf-8'))
    inputs = np.load(out / 'image_inputs.npz')
    height_shift = runtime['input_height_shift_m']

    def unpack(name, points):
        return np.frombuffer(base64.b64decode(data[name]), '<f4').copy().reshape(448, points, 3)

    def pack(values):
        return base64.b64encode(np.asarray(values, '<f4').tobytes()).decode('ascii')

    # One constant inverse of the author's height normalization. Preserve all
    # relative positions and show the simulation plane separately at this Z.
    vertices = unpack('mesh', 6890)
    vertices[..., 2] += height_shift
    for name in ('position', 'reference'):
        array = unpack(name, 24)
        array[..., 2] += height_shift
        data[name] = pack(array)
    data['mesh'] = pack(vertices)
    R, t = inputs['camera_rotation'], inputs['camera_translation_m']
    camera_vertices = np.einsum('nji,nvj->nvi', R, vertices-t[:, None])
    regressor = np.load(root / 'models/smpl/J_regressor_coco.npy')
    coco_camera = np.einsum('jv,nvc->njc', regressor, camera_vertices)
    cal = load_stereo_fisheye(root / 'realtime_app/calibration/results')
    right = coco_camera @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000
    predicted = np.stack([fisheye_project_numpy(coco_camera, cal.K0, cal.D0),
                          fisheye_project_numpy(right, cal.K1, cal.D1)], axis=1)
    predicted[(np.stack([coco_camera[..., 2], right[..., 2]], axis=1) <= 0)] = np.nan
    people = root / 'research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1'
    observed = np.stack([_load_predictions(people / f'c3_predictions/{side}/pmpose/raw_predictions.json', side)
                         for side in ('left', 'right')], axis=1)
    mask = np.isfinite(observed).all(-1) & (observed[..., 2] >= .2) & (observed[..., 2] <= 1)
    mask &= (observed[..., 0] >= 0) & (observed[..., 0] < 1920) & (observed[..., 1] >= 0) & (observed[..., 1] < 1080)
    residual = np.linalg.norm(predicted-observed[..., :2], axis=-1)
    target = mask[..., 5:17]
    difference = residual[..., 5:17]
    finite = target & np.isfinite(difference)
    summary = {
        'frames': 448, 'slots_total': 10752, 'raw_valid_slots': int(target.sum()),
        'finite_reprojection_slots': int(finite.sum()),
        'median_finite_reprojection_px': float(np.median(difference[finite])),
        'p95_finite_reprojection_px': float(np.percentile(difference[finite], 95)),
        'within25px_complete_fraction': float((finite & (difference <= 25)).sum()/10752),
        'scope': 'Raw stereo image consistency, not independent 3D accuracy; left only drives policy',
    }
    (out / 'image_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    def finite_list(values):
        values = np.asarray(values)
        return np.where(np.isfinite(values), values, None).tolist()

    timing = json.loads((out / 'scene_replay/frame_times.json').read_text(encoding='utf-8'))
    data['images'] = [[('/'+Path(row[side+'_image']).relative_to(root).as_posix())
                       for side in ('left', 'right')] for row in timing]
    data['predicted_pixels'] = finite_list(predicted)
    data['observed_pixels'] = finite_list(observed)
    data['source_times'] = inputs['timestamp_s'].tolist()
    data['simulation_plane_z_m'] = height_shift
    data['frame0_is_initialized'] = True
    data['scope'] = 'Adapted actual walker448; existing model supplies initial state and all-frame root orientation'
    data['ground'] = 'Fresh Stage conditional ground Z=0; separate author simulation plane'
    data['coordinates'] = 'Current ground frame, one constant inverse author height shift; display [X,-Y,Z]'
    data['image_median_px'] = [float(np.median(difference[i][finite[i]])) if finite[i].any() else None for i in range(448)]
    walker = json.loads((out / 'scene_replay/scene/static_walker_model.json').read_text(encoding='utf-8'))
    names = sorted({name for edge in walker['edges'] for name in edge})
    nodes = np.asarray([walker['nodes_walker_mm'][name] for name in names])/1000
    nodes = nodes @ np.asarray(walker['rotation_left_camera_from_walker']).T + np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    data['walker'] = (np.einsum('nij,vj->nvi', R, nodes)+t[:, None]).tolist()
    data['walker_edges'] = [[names.index(a), names.index(b)] for a, b in walker['edges']]
    centers = np.stack([np.zeros(3), -cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm/1000)])
    data['cameras'] = (np.einsum('nij,vj->nvi', R, centers)+t[:, None]).tolist()
    raw_points = np.full((448, 17, 3), np.nan)
    raw_valid = np.zeros((448, 17), bool)
    reasons = []
    for i, line in enumerate((out / 'scene_replay/first_instance_strict.jsonl').read_text(encoding='utf-8').splitlines()):
        row = json.loads(line)
        frame_reasons = ['missing']*17
        if row['persons_3d']:
            for point in row['persons_3d'][0]['keypoints_3d']:
                j = point['index']
                if point.get('xyz') is not None:
                    raw_points[i, j] = np.asarray(point['xyz'])/1000
                raw_valid[i, j] = point['valid']
                frame_reasons[j] = point.get('reason', '')
        reasons.append(frame_reasons)
    data['raw_points'] = finite_list(np.einsum('nij,nvj->nvi', R, raw_points)+t[:, None])
    data['raw_accepted'] = raw_valid.tolist()
    data['raw_reasons'] = reasons
    scene = np.load(out / 'scene_replay/scene/scene_transforms.npz', allow_pickle=True)
    data['stage'] = scene['stage'].astype(str).tolist()
    data['scene_status'] = {k: scene[k].astype(str).tolist() for k in scene.files
                            if scene[k].shape == (448,) and any(word in k for word in ('status', 'reason'))}
    with gzip.open(view / 'data.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))
    html = (view / 'index.html').read_text(encoding='utf-8')
    html = html.replace('官方示例三维回放', '你的448帧三维回放').replace('冻结预训练策略 / 官方示例', '冻结预训练策略 / 你的448帧')
    html = html.replace('这是官方 AIST 的 100 步示例，不是你的 448 帧视频。参考骨架不是独立真值；地面是仿真平面。',
                        '实际左目 RGB → HMR2/GVHMR 特征 → 冻结 PhysHMR；借用原模型初始化和根朝向。橙色为重定向输入参考。未加腕支撑或重置。')
    html = html.replace('<canvas id="scene" aria-label="PhysHMR 三维人体回放"></canvas>',
        '<div class="layout"><canvas id="scene" aria-label="PhysHMR 三维人体回放"></canvas><aside>'
        '<p>原图回投：绿点为物理模型，橙圈为 PMPose 观测</p><div class="inputs">'
        '<div>左目（策略输入）<canvas id="input0" width="216" height="384"></canvas></div>'
        '<div>右目（检查）<canvas id="input1" width="216" height="384"></canvas></div></div>'
        '<p id="sceneinfo"></p><details><summary>当前帧拒绝原因 / 场景状态</summary><pre id="details"></pre></details></aside></div>')
    html = html.replace('</style>', '.layout{display:grid;grid-template-columns:minmax(400px,1fr) 452px}.inputs{display:flex;gap:8px}.inputs canvas{width:216px;height:384px}aside{padding:10px;background:#fff;font-size:12px}pre{white-space:pre-wrap;font-size:10px}#scene{height:calc(100vh - 258px)}@media(max-width:950px){.layout{display:block}}\n</style>')
    html = html.replace('<script src="viewer.js"></script>', '<script src="viewer.js"></script><script src="walker.js"></script>')
    (view / 'index.html').write_text(html, encoding='utf-8')
    js = (view / 'viewer.js').read_text(encoding='utf-8')
    js = js.replace("if($('joints').checked)", "if(typeof drawWalkerScene==='function')drawWalkerScene();if($('joints').checked)")
    js = js.replace("canvas.dataset.frame=frame;", "canvas.dataset.frame=frame;if(typeof drawWalkerInputs==='function')drawWalkerInputs();")
    js = js.replace('((frame+1)*data.dt)', '(frame*data.dt)')
    (view / 'viewer.js').write_text(js, encoding='utf-8')
    (view / 'walker.js').write_text((Path(__file__).parent / 'physhmr_walker_view.js').read_text(encoding='utf-8'), encoding='utf-8')
    metadata = json.loads((view / 'metadata.json').read_text(encoding='utf-8'))
    metadata.update(scope='Adapted actual walker448, left input only; supplied existing model orientation and initialization',
                    coordinates='Current run ground frame; constant author height shift reversed for all physics output/reference',
                    simulation_plane_z_m=height_shift, ground='Same-run Stage conditional ground Z=0, not independent truth',
                    image_comparison=summary, scene=str(out / 'scene_replay/scene/scene_transforms.npz'))
    (view / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
