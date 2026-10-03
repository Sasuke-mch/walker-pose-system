"""Local DA3 image-space diagnostic; never changes observations or stereo."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from experiment_wrist_processing import ANNOTATIONS, ROOT, read, sources, stats, write
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.wrist_processing import perspective_view, upper_box, upright_to_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--da3-root', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--dependencies', type=Path, required=True)
    ap.add_argument('--run-name', default='da3')
    args = ap.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError('run-name must be one directory name')
    out = args.output.resolve()/args.run_name
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    sys.path[:0] = [str(args.dependencies.resolve()), str(args.da3_root.resolve()/'src')]
    import torch
    from depth_anything_3.api import DepthAnything3
    from PIL import Image
    write(out/'protocol.json', {'model': str(args.model_dir.resolve()), 'repo': str(args.da3_root.resolve()),
        'dependencies': str(args.dependencies.resolve()), 'frames': read(ANNOTATIONS/'manifest.json')['frames'],
        'input': 'calibrated upper-limb perspective; previous Sapiens2 joints; frame0 diagnostic bootstrap uses existing same-model frame0',
        'process_res': 504, 'modes': ['paired', 'monocular'], 'known_intrinsics': True, 'extrinsics_supplied': False,
        'depth_role': 'relative local image diagnostic and occlusion/ROI research only',
        'not_used_for': ['calibration', 'person_association', 'keypoint_filter', 'triangulation', 'metric_depth', 'pose_loss'],
        'timing': 'CUDA synchronized;2warmup stereo pairs;includes perspective construction/remap and official preprocessing/forward/output conversion;no export, PNG load or pose-model cost',
        'api': 'official input_processor,_prepare_model_inputs,forward,output_processor;no export or pose scale alignment',
        'source_rule': 'AGENTS.md DA3 image-space restriction; single-pose-detector results remain exactly unchanged'})
    model = DepthAnything3.from_pretrained(str(args.model_dir.resolve())).to('cuda').eval()
    det, cached = sources()
    cal = load_stereo_fisheye(ROOT/'realtime_app/calibration/results')
    frames = read(ANNOTATIONS/'manifest.json')['frames']
    results = []
    def infer(images, Ks):
        cpu, ex, intr = model.input_processor(images, intrinsics=np.stack(Ks), process_res=504, process_res_method='upper_bound_resize', num_workers=1, sequential=True)
        imgs, _, intr_gpu = model._prepare_model_inputs(cpu, ex, intr)
        raw_output = model.forward(imgs, intrinsics=intr_gpu, export_feat_layers=[])
        prediction = model.output_processor(raw_output)
        flag = raw_output.get('is_metric')
        metric_flag = int(flag) if isinstance(flag, (int, float, bool, np.number)) else None
        torch.cuda.synchronize()
        return prediction, intr.numpy(), metric_flag
    for f in frames:
        loaded = {c: cv2.imread(det[c][f]['image_path']) for c in ('left', 'right')}
        if any(x is None for x in loaded.values()):
            raise FileNotFoundError(f'frame{f}')
        torch.cuda.synchronize()
        start = time.perf_counter()
        views, images, raw_joints = [], [], []
        for c in ('left', 'right'):
            previous = cached[c][max(f-1, 0)]
            box = upper_box(previous)
            if box is None:
                raise ValueError('DA3 diagnostic upper ROI unavailable')
            K, D = (cal.K0, cal.D0) if c == 'left' else (cal.K1, cal.D1)
            view = perspective_view(box, K, D, c, support=previous[5:11, :2])
            raw = cv2.rotate(loaded[c], cv2.ROTATE_90_CLOCKWISE if c == 'left' else cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.cvtColor(view.remap(raw), cv2.COLOR_BGR2RGB)
            views.append(view)
            images.append(Image.fromarray(image))
            raw_joints.append(upright_to_raw(cached[c][f, [7, 8, 9, 10], :2], c))
        view_ms = (time.perf_counter()-start)*1000
        if f == frames[0]:
            for _ in range(2):
                infer(images, [v.intrinsic for v in views])
        for mode in ('paired', 'monocular'):
            groups = [[0, 1]] if mode == 'paired' else [[0], [1]]
            for group in groups:
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction, inputKs, metric_flag = infer([images[i] for i in group], [views[i].intrinsic for i in group])
                elapsed = (time.perf_counter()-start)*1000
                samples = []
                for n, i in enumerate(group):
                    depth, conf = prediction.depth[n], prediction.conf[n]
                    virtual = views[i].from_raw(raw_joints[i])
                    rays = np.column_stack((virtual, np.ones(len(virtual)))) @ np.linalg.inv(views[i].intrinsic).T
                    q = rays @ inputKs[n].T
                    q = q[:, :2]/q[:, 2:3]
                    H, W = depth.shape
                    for j, (x, y) in zip((7, 8, 9, 10), q):
                        available = 0 <= x < W and 0 <= y < H
                        item = {'camera': ('left', 'right')[i], 'joint_coco': j, 'available': bool(available), 'depth_pixel': [float(x), float(y)]}
                        if available:
                            ix, iy = int(round(x)), int(round(y))
                            ix, iy = min(ix, W-1), min(iy, H-1)
                            patch = depth[max(0, iy-4):iy+5, max(0, ix-4):ix+5]
                            item.update(relative_depth=float(depth[iy, ix]), confidence=float(conf[iy, ix]), local_depth_range_ratio=float((np.max(patch)-np.min(patch))/max(np.median(patch), 1e-8)))
                        samples.append(item)
                    np.savez_compressed(out/f'{f:04d}_{mode}_{i}.npz', depth=depth, confidence=conf, processed_intrinsic=inputKs[n], virtual_intrinsic=views[i].intrinsic, camera_from_virtual=views[i].rotation)
                    if mode == 'paired':
                        im = np.asarray(images[i])
                        dd = (depth-np.percentile(depth, 2))/max(float(np.percentile(depth, 98)-np.percentile(depth, 2)), 1e-8)
                        color = cv2.applyColorMap((np.clip(dd, 0, 1)*255).astype('uint8'), cv2.COLORMAP_TURBO)
                        color = cv2.resize(color, (im.shape[1], im.shape[0]))
                        Image.fromarray(np.concatenate((im, color[..., ::-1]), axis=1)).save(out/f'{f:04d}_{i}_depth.png')
                results.append({'frame': f, 'mode': mode, 'views': group, 'elapsed_ms': elapsed, 'view_ms_stereo': view_ms, 'finite_fraction': float(np.isfinite(prediction.depth).mean()), 'is_metric': metric_flag, 'shape': list(prediction.depth.shape), 'samples': samples})
        print(f'DA3 {f} done', flush=True)
    summary = {'paired_inference_ms': stats([r['elapsed_ms'] for r in results if r['mode'] == 'paired']),
        'monocular_inference_ms': stats([r['elapsed_ms'] for r in results if r['mode'] == 'monocular']),
        'paired_with_view_ms': stats([r['elapsed_ms']+r['view_ms_stereo'] for r in results if r['mode'] == 'paired']),
        'finite_fraction_min': min(r['finite_fraction'] for r in results),
        'available_samples': sum(x['available'] for r in results for x in r['samples']),
        'total_samples': sum(len(r['samples']) for r in results), 'pose_points_modified': 0,
        'metric_flags': list({r['is_metric'] for r in results}), 'records': results}
    write(out/'results.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
