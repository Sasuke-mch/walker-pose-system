"""Decode saved human/walker foundation runs into a local, frame-indexed viewer."""
import argparse
import base64
import gzip
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .analyze import ROOT, load_model, smpl_rotations


def packed(array, dtype=np.float32):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')


def build(foundation, output, motion=None, rl=None):
    output.mkdir(parents=True, exist_ok=False)
    base = foundation.parent
    reference = np.load(base/'corrections_v1/reference_matched.npz', allow_pickle=False)
    provenance = json.loads((base/'corrections_v1/reference_matched.json').read_text(encoding='utf-8'))
    source = np.load(provenance['source'], allow_pickle=False)
    beta = torch.tensor(reference['betas'], dtype=torch.float32)
    hands = [torch.tensor(Rotation.from_rotvec(source[k][0].reshape(15, 3)).as_matrix()[None],
                          dtype=torch.float32) for k in ('left_hand_pose', 'right_hand_pose')]
    model = load_model()
    torch.set_num_threads(1)
    rest = model.v_template+torch.einsum('bl,vcl->bvc', beta, model.shapedirs)[0]
    offset = (model.J_regressor[0]@rest).detach().numpy()
    regressor = np.load(ROOT/'models/smpl/J_regressor_coco.npy')
    asset = json.loads((foundation/'synthetic_fixture_asset_v1/asset_metadata.json').read_text(encoding='utf-8'))
    static_rods = asset['rods']+asset['pads']
    endpoints = np.array([[rod['a'], rod['b']] for rod in static_rods])
    stereo_source = base/'corrections_v1/reference_exact_heel.npz'
    stereo = np.load(stereo_source, allow_pickle=False)
    relative = np.eye(4)
    relative[:3, :3], relative[:3, 3] = stereo['R01'], stereo['T01_m'].ravel()
    saved = dict(np.load(foundation/'nonzero_motion_restore_v3/saved_motion_state.npz', allow_pickle=False))
    cases = [('full', '完整1秒 · 固定关节目标', foundation/'armature_fixed_control_substeps2_v1/trajectory.npz'),
             ('resumed', '中途恢复 · 0.125–0.325秒', foundation/'nonzero_motion_restore_v3/resumed.npz'),
             ('uninterrupted', '未中断对照 · 同一时间窗', foundation/'nonzero_motion_restore_v3/uninterrupted.npz')]
    if motion is not None:
        cases = [('motion', '前2秒 · 运动目标驱动（非RL）', motion/'trajectory.npz')]
    if rl is not None:
        saved = dict(np.load(rl/'initial_state.npz', allow_pickle=False))
        cases = [('rl', '完整448帧 · 自由助步器图像PPO', rl/'trajectory.npz')]
    runs, roundtrip = [], None
    for key, label, path in cases:
        z = np.load(path, allow_pickle=False)
        if key in ('full', 'motion', 'rl'):
            p, r = z['human'][..., 0], z['human'][..., 1:]
            time = z['timestamp_s'] if key in ('motion', 'rl') else np.arange(len(p))/240
            if key == 'rl':
                force, gap = z['interval_force_peak_N'], z['grip_gap_m']
            else:
                force, gap = z['grip_force_N'], z['grip_gap_m']
        else:
            p, r, time = z['position'], z['rotation'], z['time_s']
            force, gap = z['force_N'], z['gap_m']
        walker = z['walker']
        root, body = smpl_rotations(r)
        vertices = []
        for start in range(0, len(p), 32):
            stop = min(start+32, len(p))
            count = stop-start
            with torch.inference_mode():
                mesh = model(betas=beta.expand(count, -1),
                    global_orient=torch.tensor(root[start:stop], dtype=torch.float32),
                    body_pose=torch.tensor(body[start:stop], dtype=torch.float32),
                    transl=torch.tensor(p[start:stop, 0]-offset, dtype=torch.float32),
                    left_hand_pose=hands[0].expand(count, -1, -1, -1),
                    right_hand_pose=hands[1].expand(count, -1, -1, -1)).vertices.numpy()
            vertices.append(mesh)
        vertices = np.concatenate(vertices)
        assert vertices.shape == (len(p), 6890, 3) and model.faces.shape == (13776, 3)
        assert np.isfinite(vertices).all() and np.all(np.diff(time) > 0)
        if key in ('full', 'motion', 'rl'):
            initial_delta = json.loads((path.parent/'results.json').read_text())['initialization']['contact_initialization']['human_root_z_delta_m']
            roundtrip = float(np.abs(vertices[0]-[0, 0, initial_delta]-source['world_vertices'][0]).max())
            assert roundtrip < 1e-5
        rw = Rotation.from_quat(walker[:, 3:7]).as_matrix()
        rods = np.einsum('nij,abj->nabi', rw, endpoints)+walker[:, None, None, :3]
        transform = np.tile(np.eye(4), (len(p), 1, 1))
        transform[:, :3, :3], transform[:, :3, 3] = rw, walker[:, :3]
        left = transform@np.asarray(asset['left_camera_mount']['T_W_from_C'])
        right = left@np.linalg.inv(relative)
        assert np.allclose(np.linalg.inv(right)@left, relative, atol=1e-6)
        if key in ('full', 'motion', 'rl'):
            np.testing.assert_allclose(left, z['cameras'], atol=1e-6)
            np.testing.assert_allclose(right, z['right_cameras'], atol=1e-6)
        cameras = np.stack([left[:, :3, 3], right[:, :3, 3]], axis=1)
        anchors = []
        for j, index in enumerate((17, 22)):
            a = p[:, index]+r[:, index]@saved['grip_local_wrist'][j]
            b = walker[:, :3]+rw@saved['grip_local_handle'][j]
            anchors.append(np.stack([a, b], axis=1))
        anchors = np.stack(anchors, axis=1)
        np.testing.assert_allclose(np.linalg.norm(anchors[:, :, 0]-anchors[:, :, 1], axis=2), gap, atol=1e-6)
        runs.append(dict(key=key, label=label, source=str(path.resolve()), frames=len(p), dt=1/240,
            time=time.tolist(), mesh=packed(vertices), coco=packed(np.einsum('jv,nvc->njc', regressor, vertices)),
            position=packed(p), rods=packed(rods), cameras=packed(cameras), anchors=packed(anchors),
            gap_mm=(gap*1000).tolist(),
            force_N=force.tolist() if key == 'rl' else np.linalg.norm(force, axis=2).tolist()+[[None, None]],
            description='实际PPO网络输出关节控制；双手持续连接为假设；失败状态保留' if key == 'rl' else
                        '已到达的连续姿态只驱动关节目标；自由根和助步器不覆盖；非RL' if key == 'motion' else
                        '首帧固定目标；无新图像观测；非行走跟踪' if key == 'full' else
                        '同一次0.125秒状态的恢复/未中断对照；保留全部49状态'))
        if key == 'motion':
            index = z['target_index']
            target = reference['reference_position_m'][index]
            runs[-1].update(reference=packed(target), target_index=index.tolist(),
                target_age_ms=(z['target_age_s']*1000).tolist(),
                root_reference_difference_mm=(np.linalg.norm(p[:, 0]-target[:, 0], axis=1)*1000).tolist())
        if key == 'rl':
            runs[-1].update(fallen=z['fallen'].tolist(), source_time=z['source_timestamp_s'].tolist(),
                status_text='实际图像PPO；决策60Hz / 握持240Hz / 物理480Hz；相机随自由助步器运动；'
                    '原视频非均匀时间回放，无插值；没有本轮Stage或三角化；合成物理参数；非真实精度证明',
                force_label='截至当前帧区间力峰',
                errors_median_px=[float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else None
                                  for e in z['pixel_error_px']])
    payload = dict(runs=runs, faces_u32=packed(model.faces.ravel(), np.uint32),
        radii=[rod['radius'] for rod in static_rods],
        center=np.r_[reference['reference_position_m'][0, 0, :2], .85].tolist(),
        scope='actual bounded PPO adaptation and full448 physics; not independent accuracy' if rl is not None else
              'controlled physical experiment; no new video/Stage/triangulation; no trained policy',
        default_rate=1 if motion is not None or rl is not None else .1,
        reference=str((base/'corrections_v1/reference_matched.npz').resolve()),
        stereo_source=str(stereo_source.resolve()), source_model=provenance['source'],
        initial_mesh_roundtrip_max_m=roundtrip)
    with gzip.open(output/'data.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    for src, dest in [('foundation_view.html', 'index.html'), ('foundation_view.js', 'viewer.js')]:
        shutil.copyfile(Path(__file__).with_name(src), output/dest)
    page = (output/'index.html').read_text(encoding='utf-8')
    for link in ('nonzero_motion_restore_v3/restore_comparison.png',
                 'armature_fixed_control_substeps2_v1/force_timing_diagnosis.png'):
        page = page.replace('../'+link, Path(os.path.relpath(foundation/link, output)).as_posix())
    if motion is not None:
        page = page.replace('最新物理基础回放', '前2秒运动目标驱动 · 非强化学习输出')
        page = page.replace('受控仿真：首帧姿态固定目标，没有新图像输入，也没有策略训练。',
            '关节目标随已到达的姿态变化；橙骨架为当前输入参考，非真实标注。人体整体与助步器由物理仿真产生。')
    if rl is not None:
        page = page.replace('最新物理基础回放', '448帧图像强化学习 · 自由助步器')
        page = page.replace('受控仿真：首帧姿态固定目标，没有新图像输入，也没有策略训练。这不是448帧视频重建；参数尚未实测，外观接近不等于真实承重。',
            '已实际进行PPO训练，冻结网络处理448帧。绿色为物理输出；所有摔倒和偏离保留，完整运行不代表跟踪成功。双手持续连接与动力学参数仍是假设。')
    (output/'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(output=str(output.resolve()), frames=[r['frames'] for r in runs],
        vertex_count=6890, face_count=13776, initial_mesh_roundtrip_max_m=roundtrip), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('foundation', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--motion', type=Path, help='saved bounded moving-target diagnostic, instead of static cases')
    parser.add_argument('--rl', type=Path, help='saved free-walker image-PPO448 replay')
    args = parser.parse_args()
    build(args.foundation, args.output, args.motion, args.rl)
