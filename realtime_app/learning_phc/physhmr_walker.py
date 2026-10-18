"""Prepare actual walker RGB features and calibrated rays for frozen PhysHMR.

This adapted experiment supplies the existing model's root orientation; it is
not an independent monocular reconstruction or a fair same-input comparison.
"""

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .contract import BASE_ROTATION, SMPL_INDICES

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'research_records/engineering_validation/G20261015_phc_pretrained448_v1'
GVHMR = ROOT / 'third_party/GVHMR'
VIRTUAL_R = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
VIRTUAL_K = np.array([[800., 0., 1024.], [0., 800., 5000.], [0., 0., 1.]])


def selected_source(path, names, namespace):
    """Execute unchanged author definitions without unrelated renderer imports."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing author definitions: {names}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102 - known local author definitions


def load_encoder():
    import pytorch_lightning as pl
    from yacs.config import CfgNode

    sys.path.insert(0, str(GVHMR))
    # Package-only registration bypasses HMR2's unused full-SMPL/PyTorch3D path.
    package = types.ModuleType('hmr4d.network.hmr2')
    package.__path__ = [str(GVHMR / 'hmr4d/network/hmr2')]
    sys.modules[package.__name__] = package
    from hmr4d.network.hmr2.configs import get_config
    from hmr4d.network.hmr2.smpl_head import SMPLTransformerDecoderHead
    from hmr4d.network.hmr2.vit import ViT
    namespace = {'pl': pl, 'torch': torch, 'CfgNode': CfgNode, 'ViT': ViT,
                     'SMPLTransformerDecoderHead': SMPLTransformerDecoderHead}
    selected_source(GVHMR / 'hmr4d/network/hmr2/hmr2.py', ['HMR2'], namespace)
    cfg = get_config(str(GVHMR / 'hmr4d/network/hmr2/configs/model_config.yaml'))
    cfg.defrost()
    cfg.SMPL.MEAN_PARAMS = str(GVHMR / 'hmr4d/network/hmr2/configs/smpl_mean_params.npz')
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.freeze()
    encoder = namespace['HMR2'](cfg)
    state = torch.load(ROOT / 'models/gvhmr/hmr2.ckpt', map_location='cpu', weights_only=False)['state_dict']
    encoder.load_state_dict({k: v for k, v in state.items() if k.split('.')[0] in ('backbone', 'smpl_head')}, strict=True)
    del state
    return encoder.cuda().eval().requires_grad_(False)


def prepare(out):
    from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
    from tools.run_stereo_pva_baseline import _load_predictions

    destination = out / 'image_inputs.npz'
    if destination.exists():
        raise FileExistsError(destination)
    cal = load_stereo_fisheye(ROOT / 'realtime_app/calibration/results')
    frames = json.loads((out / 'scene_replay/frame_times.json').read_text(encoding='utf-8'))
    detection = ROOT / ('research_records/engineering_validation/'
                       'V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/left/pmpose/raw_predictions.json')
    raw = _load_predictions(detection, 'left')
    scene = np.load(out / 'scene_replay/scene/scene_transforms.npz', allow_pickle=True)
    world_R = scene['rotation_ground_from_left']
    world_t = scene['translation_ground_from_left_mm'] / 1000
    # OpenCV 4.11 misreads this strided xy view; pass contiguous pixel pairs.
    uv = cv2.fisheye.undistortPoints(np.ascontiguousarray(raw[..., :2]).reshape(-1, 1, 2), cal.K0, cal.D0).reshape(448, 17, 2)
    rays = np.concatenate([uv, np.ones((448, 17, 1))], axis=-1)
    virtual = rays @ VIRTUAL_R.T
    pixels = virtual[..., :2] / virtual[..., 2:] * 800 + VIRTUAL_K[:2, 2]
    valid = (np.isfinite(raw).all(-1) & (raw[..., 2] >= .2) & (raw[..., 2] <= 1)
             & (raw[..., 0] >= 0) & (raw[..., 0] < 1920) & (raw[..., 1] >= 0) & (raw[..., 1] < 1080))
    kp = np.concatenate([pixels, np.where(valid, raw[..., 2], 0)[..., None]], axis=-1).astype(np.float32)
    bbx = []
    for i in range(448):
        pts = pixels[i, valid[i]]
        if len(pts) < 4:
            raise ValueError(f'Insufficient frame {i} observations; do not fill or skip')
        low, high = pts.min(0), pts.max(0)
        bbx.append([*(.5 * (low + high)), max(high-low) * 1.2])
    bbx = np.asarray(bbx, np.float32)
    # Render each crop directly from its virtual pinhole rays. A small virtual
    # viewport would clip this extreme peripheral fisheye subject. No clipping
    # to an intermediate canvas and no invented pixels outside the source.
    yy, xx = np.mgrid[:512, :512]
    crops = []
    for i, record in enumerate(frames):
        upright = cv2.imdecode(np.fromfile(record['left_image'], np.uint8), cv2.IMREAD_COLOR)
        if upright is None:
            raise ValueError(f'Image unavailable at {i}')
        image = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
        # Inverse of the author's center/size square crop, supersampled 2x.
        px = bbx[i, 0] + ((xx+.5)/512-.5) * bbx[i, 2]
        py = bbx[i, 1] + ((yy+.5)/512-.5) * bbx[i, 2]
        virtual_grid = np.stack([(px-1024)/800, (py-5000)/800, np.ones_like(px)], -1)
        raw_grid = fisheye_project_numpy(virtual_grid @ VIRTUAL_R, cal.K0, cal.D0).astype(np.float32)
        rect = cv2.remap(image, raw_grid[..., 0], raw_grid[..., 1], cv2.INTER_LINEAR)
        patch = cv2.resize(rect, (256, 256), interpolation=cv2.INTER_AREA)[..., ::-1]
        crops.append(patch)
        if i in (0, 143, 267, 447):
            cv2.imencode('.jpg', patch[..., ::-1])[1].tofile(str(out / f'input_crop_{i:03d}.jpg'))
        if i % 100 == 0:
            print('Prepared RGB', i, flush=True)
    crops = np.asarray(crops)
    virtual_R_world_to_camera = VIRTUAL_R @ world_R.transpose(0, 2, 1)
    delta = virtual_R_world_to_camera[1:] @ virtual_R_world_to_camera[:-1].transpose(0, 2, 1)
    angular = np.concatenate([delta[:, :2].reshape(447, 6), delta[-1:, :2].reshape(1, 6)], axis=0)
    rays_world = np.einsum('nij,nkj->nki', world_R, rays / np.linalg.norm(rays, axis=-1, keepdims=True))
    ray_points = rays_world + world_t[:, None]
    order = np.arange(5, 17)  # shoulders, elbows, wrists, hips, knees, ankles
    kp_world = np.concatenate([ray_points[:, order], np.where(valid[:, order], raw[:, order, 2], 0)[..., None]], -1)
    # Verify calibrated ray inversion against exactly the raw pixels it came from.
    reprojection = fisheye_project_numpy(rays, cal.K0, cal.D0)
    ray_error = np.linalg.norm(reprojection-raw[..., :2], axis=-1)[valid]
    if ray_error.max() > .01:
        raise ValueError(f'Fisheye ray round trip failed: {ray_error.max()} px')
    np.savez_compressed(destination, crops=crops, kp2d=kp, bbx_xys=bbx, cam_angvel=angular,
                        camera=world_t[:, None].astype(np.float32), kp_world=kp_world.astype(np.float32),
                        camera_rotation=world_R, camera_translation_m=world_t,
                        timestamp_s=scene['pair_timestamp_sec'], raw_left=raw,
                        virtual_K=VIRTUAL_K, virtual_R=VIRTUAL_R)
    (out / 'ray_check.json').write_text(json.dumps({'max_roundtrip_px': float(ray_error.max()),
        'frames': 448, 'valid_left_coco17_slots': int(valid.sum()), 'rgb_crops': [448, 256, 256, 3],
        'virtual_camera': 'upright pinhole f=800, principal=(1024,5000); direct crop from actual K/D rays, 2x supersampling',
        'black_border_is_not_filled': True}, indent=2), encoding='utf-8')


def features(out):
    if (out / 'features.npz').exists():
        raise FileExistsError(out / 'features.npz')
    z = np.load(out / 'image_inputs.npz')
    encoder = load_encoder()
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    values = []
    with torch.inference_mode():
        for i in range(0, 448, 4):
            batch = torch.from_numpy(z['crops'][i:i+4]).cuda().permute(0, 3, 1, 2).float() / 255
            token = encoder({'img': (batch-mean)/std})
            values.append(token.cpu())
            if i % 40 == 0:
                print('HMR2 RGB features', i, flush=True)
    image_features = torch.cat(values)
    del encoder
    torch.cuda.empty_cache()
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE
    network = NetworkEncoderRoPE().cuda().eval().requires_grad_(False)
    state = torch.load(ROOT / 'models/gvhmr/gvhmr_siga24_release.ckpt', map_location='cpu', weights_only=False)['state_dict']
    prefix = 'pipeline.denoiser3d.'
    network.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    namespace = {'torch': torch}
    selected_source(GVHMR / 'hmr4d/utils/geo/hmr_cam.py', ['normalize_kp2d', 'compute_bbox_info_bedlam'], namespace)
    kp = torch.tensor(z['kp2d'][None], device='cuda')
    bbx = torch.tensor(z['bbx_xys'][None], device='cuda')
    intrinsics = torch.tensor(np.broadcast_to(VIRTUAL_K, (1, 448, 3, 3)).copy(), device='cuda', dtype=torch.float32)
    angular = torch.tensor(z['cam_angvel'][None], device='cuda', dtype=torch.float32)
    average = torch.tensor([1., 0., 0., 0., 1., 0.], device='cuda')
    scale = torch.tensor([.001, .1, .1, .1, .001, .1], device='cuda')
    with torch.inference_mode():
        result = network(length=torch.tensor([448], device='cuda'),
                         obs=namespace['normalize_kp2d'](kp, bbx),
                         f_cliffcam=namespace['compute_bbox_info_bedlam'](bbx, intrinsics),
                         f_cam_angvel=(angular-average)/scale,
                         f_imgseq=image_features[None].cuda())
    context = result['pred_context'][0].cpu().numpy()
    if context.shape != (448, 512) or not np.isfinite(context).all():
        raise ValueError('Invalid actual GVHMR context')
    np.savez_compressed(out / 'features.npz', image1024=image_features.numpy(), context512=context)
    print('Actual pretrained context complete', context.shape, flush=True)


def sample(out):
    import smplx
    from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility

    destination = out / 'walker448_sample.npz'
    if destination.exists():
        raise FileExistsError(destination)
    _install_legacy_smpl_pickle_compatibility()
    inputs = np.load(out / 'image_inputs.npz')
    original = np.load(EXPERIMENT / 'reference448.npz')
    old_R = original['camera_rotation']
    new_R, new_t = inputs['camera_rotation'], inputs['camera_translation_m']
    change = new_R @ old_R.transpose(0, 2, 1)
    world = change[:, None] @ original['reference_rotation'][:, np.argsort(SMPL_INDICES)] @ BASE_ROTATION
    model = smplx.SMPL(str(ROOT / 'third_party/PhysHMR/data/smpl'), gender='neutral')
    local = np.stack([world[:, model.parents[j]].transpose(0, 2, 1) @ world[:, j] for j in range(1, 24)], 1)
    aa = np.concatenate([Rotation.from_matrix(world[:, 0]).as_rotvec(),
                         Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(448, 69)], -1)
    neutral_global = world[:, SMPL_INDICES] @ BASE_ROTATION.T
    neutral_local = []
    inverse = np.argsort(SMPL_INDICES)
    for j in range(24):
        neutral_local.append(neutral_global[:, j] if j == 0 else
                             neutral_global[:, inverse[int(model.parents[SMPL_INDICES[j]])]].transpose(0, 2, 1) @ neutral_global[:, j])
    root = np.einsum('nij,nj->ni', change, original['source_pelvis_world_m']-original['camera_translation_m']) + new_t
    quat = Rotation.from_matrix(neutral_global.reshape(-1, 3, 3)).as_quat().reshape(448, 24, 4)
    quat_local = Rotation.from_matrix(np.stack(neutral_local, 1).reshape(-1, 3, 3)).as_quat().reshape(448, 24, 4)
    np.savez_compressed(destination, sample_name='walker448_left_adapted', pose_quat_global=quat,
                        pose_quat=quat_local, root_trans_offset=root, pose_aa=aa,
                        img_feat=np.load(out / 'features.npz')['context512'], camera=inputs['camera'],
                        kp2d=inputs['kp_world'], fps=30, gender='neutral', scale=np.ones((1, 1)),
                        poses_g=aa, trans_g=root)
    print('Prepared adapted walker448 sample; real features, no training', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'features', 'sample'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    globals()[args.stage](args.output.resolve())


if __name__ == '__main__':
    main()
