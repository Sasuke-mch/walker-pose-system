"""Prepare identical, masked raw-fisheye inputs for two independent SLAM backends."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
from pose_app.calibration import StereoCalibration  # noqa: E402


def linux_path(path: Path) -> str:
    path = path.resolve()
    if path.drive:
        return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]
    return str(path)


def raw_points(points: np.ndarray, side: str, raw_size: tuple[int, int]) -> np.ndarray:
    points = np.asarray(points, dtype=float).copy()
    u, v = points[:, 0].copy(), points[:, 1].copy()
    if side == 'left':
        points[:, 0], points[:, 1] = raw_size[0] - 1 - v, u
    elif side == 'right':
        points[:, 0], points[:, 1] = v, raw_size[1] - 1 - u
    else:
        raise ValueError(side)
    return points


def scaled_K(K: np.ndarray, scale: float) -> np.ndarray:
    result = K.copy()
    result[0, 0] *= scale
    result[1, 1] *= scale
    result[0, 2] = (K[0, 2] + .5) * scale - .5
    result[1, 2] = (K[1, 2] + .5) * scale - .5
    return result


def walker_mask(model: dict, cal: StereoCalibration, side: str, size: tuple[int, int]) -> np.ndarray:
    """Project sampled reference rods; padding expresses uncertain installation/rod thickness."""
    mask = np.zeros(size[::-1], np.uint8)
    names = model['nodes_walker_mm']
    nodes = {key: np.array(value, float) for key, value in names.items()}
    R = np.array(model['rotation_left_camera_from_walker'])
    t = np.array(model['translation_left_camera_from_walker_mm'])
    radius = max(12, round(size[0] * .035))
    K, D = (cal.left_K, cal.left_D) if side == 'left' else (cal.right_K, cal.right_D)
    for start, end in model['edges']:
        samples = np.linspace(nodes[start], nodes[end], 301)
        xyz = samples @ R.T + t
        if side == 'right':
            xyz = xyz @ cal.R.T + cal.T
        # Both selected backends use the forward hemisphere; do not bridge invalid rays.
        valid = xyz[:, 2] > 1.0
        previous = None
        for point, ok in zip(xyz, valid):
            if not ok:
                previous = None
                continue
            uv = cv2.fisheye.projectPoints(point.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(2)
            if not np.isfinite(uv).all():
                previous = None
                continue
            pixel = tuple(np.rint(np.clip(uv, -10000, 10000)).astype(int))
            if previous is not None:
                cv2.line(mask, previous, pixel, 255, 2 * radius + 1)
            cv2.circle(mask, pixel, radius, 255, -1)
            previous = pixel
    return mask


def reserve_cells(size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed held-out pixel cells; exclusion includes descriptor-support padding."""
    width, height = size
    yy, xx = np.indices((height, width))
    selected = (((yy * 6 // height) * 10 + (xx * 10 // width)) % 7 == 0)
    holdout = selected.astype(np.uint8) * 255
    padded = cv2.dilate(holdout, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    # Avoid ambiguous image/mask boundaries for the test feature detector.
    test = cv2.erode(holdout, np.ones((9, 9), np.uint8))
    return test, padded


def forward_mask(K: np.ndarray, D: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Common forward-hemisphere domain required by cuVSLAM fisheye projection."""
    yy, xx = np.indices(size[::-1])
    radius = np.hypot((xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1])
    theta = np.deg2rad(89)
    limit = theta * (1 + sum(float(k) * theta ** (2 * i + 2) for i, k in enumerate(D)))
    return (radius >= limit).astype(np.uint8) * 255


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory is not empty; preserve previous run and choose a new directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.source_plan.read_text(encoding='utf-8'))
    image_dir = Path(plan['inputs']['images'])
    originals = StereoCalibration.load(args.calibration)
    if originals.length_unit != 'mm':
        raise ValueError('This comparison uses the project millimetre calibration bundle.')
    raw_size = originals.left_image_size
    scale = args.width / raw_size[0]
    size = (args.width, round(raw_size[1] * scale))
    cal = StereoCalibration('fisheye', size, size, scaled_K(originals.left_K, scale),
                            originals.left_D, scaled_K(originals.right_K, scale), originals.right_D,
                            originals.R, originals.T, originals.length_unit)
    model = json.loads(args.walker_model.read_text(encoding='utf-8'))
    predictions = {side: {int(row['image_id']): row for row in
                   json.loads(Path(plan['inputs'][side]).read_text(encoding='utf-8'))['images']}
                   for side in ('left', 'right')}
    ids = sorted(predictions['left'])
    if ids != list(range(448)) or sorted(predictions['right']) != ids:
        raise ValueError('Comparison requires all original 448 paired frames, 0..447.')
    source_rows = list(csv.DictReader((image_dir / 'selection_manifest.csv').open(encoding='utf-8-sig')))
    if len(source_rows) != len(ids):
        raise ValueError('Source selection and prediction counts differ.')
    if [int(row['sequence_index']) for row in source_rows] != ids:
        raise ValueError('Source selection frame order differs from prediction order.')
    heldout, padded = reserve_cells(size)
    walker = {side: walker_mask(model, cal, side, size) for side in ('left', 'right')}
    forward = {side: forward_mask(K, D, size) for side, K, D in
               [('left', cal.left_K, cal.left_D), ('right', cal.right_K, cal.right_D)]}
    frames = []
    fractions = []
    for frame_id in ids:
        frame = {'frame_id': frame_id, 'pair_id': int(source_rows[frame_id]['source_pair_index']),
                 'timestamp_ns': round(frame_id * 1e9 / 30), 'timestamp_type': 'nominal_30fps_not_exposure'}
        frame['source_left_frame_id'] = int(source_rows[frame_id]['left_frame_id'])
        frame['source_right_frame_id'] = int(source_rows[frame_id]['right_frame_id'])
        frame['abs_host_return_delta_ms'] = float(source_rows[frame_id]['abs_host_delta_ms'])
        views = []
        for side, folder, rotation in [('left', 'left_ccw90', cv2.ROTATE_90_CLOCKWISE),
                                       ('right', 'right_cw90', cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            source_path = image_dir / folder / f'pair_{frame_id:04d}.png'
            upright = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if upright is None:
                raise FileNotFoundError(source_path)
            raw = cv2.rotate(upright, rotation)
            if raw.shape[1::-1] != raw_size:
                raise ValueError(f'Image size differs from calibration: {source_path}')
            image = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
            semantic = walker[side] | forward[side]
            points = raw_points(np.array(predictions[side][frame_id]['keypoints'][0]), side, raw_size)
            points[:, :2] = (points[:, :2] + .5) * scale - .5
            finite = np.isfinite(points).all(axis=1) & (points[:, 2] >= .25)
            selected = points[finite, :2]
            if len(selected) < 3:
                # Missing person evidence must not become permission to use a dynamic image.
                semantic[:] = 255
                frame[f'{side}_mask_reason'] = 'insufficient_person_keypoints_full_exclusion'
            else:
                person = np.zeros_like(semantic)
                cv2.fillConvexPoly(person, cv2.convexHull(np.rint(selected).astype(np.int32)), 255)
                radius = round(.055 * max(size))
                person = cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
                # Hands and shoes extend beyond the skeleton hull. Keep their uncertain fringes excluded.
                for joint in (9, 10, 15, 16):
                    if len(points) > joint and finite[joint]:
                        cv2.circle(person, tuple(np.rint(points[joint, :2]).astype(int)), round(.085 * max(size)), 255, -1)
                semantic |= person
                frame[f'{side}_mask_reason'] = 'all_channel_person_hull_plus_projected_reference_rods'
            semantic[:8] = semantic[-8:] = 255
            semantic[:, :8] = semantic[:, -8:] = 255
            excluded = semantic | padded if args.holdout_cells else semantic
            test = cv2.bitwise_and(heldout if args.holdout_cells else np.full_like(heldout, 255), cv2.bitwise_not(semantic))
            for kind, array in [('image', image), ('mask', excluded), ('test_mask', test)]:
                path = args.output / side / kind / f'{frame_id:04d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), array):
                    raise OSError(path)
                key = f'{side}_path' if kind == 'image' else f'{side}_{kind}_path'
                frame[key] = linux_path(path)
            fractions.append({'frame_id': frame_id, 'side': side,
                              'train_fraction': float(np.mean(excluded == 0)),
                              'test_fraction': float(np.mean(test > 0))})
            overlay = image.copy()
            overlay[semantic > 0] = overlay[semantic > 0] // 3
            overlay[test > 0] = (overlay[test > 0] * .4 + np.array([0, 160, 0]) * .6).astype(np.uint8)
            views.append(overlay)
        frames.append(frame)
        if frame_id in (0, 224, 447):
            cv2.imwrite(str(args.output / f'mask_preview_{frame_id:04d}.jpg'), np.concatenate(views, axis=1))
    manifest = {'schema_version': 1, 'source_plan': str(args.source_plan.resolve()),
                'timestamp_source': 'nominal_30fps; original host-return skew retained; exposure synchronization unknown',
                'calibration_source': str(args.calibration.resolve()), 'walker_model_source': str(args.walker_model.resolve()),
                'stage_used': False, 'mask_nonzero': 'excluded',
                'mask_certified': False, 'mask_evidence': 'conservative keypoint hull and coarse rigid model projection; visual review only',
                'calibration': {'model': 'opencv_fisheye', 'image_size': list(size),
                                'left_K': cal.left_K.tolist(), 'right_K': cal.right_K.tolist(),
                                'left_D': cal.left_D.tolist(), 'right_D': cal.right_D.tolist(),
                                'R_right_from_left': cal.R.tolist(),
                                'translation_right_from_left_m': (cal.T / 1000).tolist()},
                'holdout': {'enabled': args.holdout_cells, 'grid': [6, 10], 'cell_modulus': 7,
                            'descriptor_padding_px': 20,
                            'scope': 'withheld feature centers' if args.holdout_cells else 'same-source background diagnostic; not held-out or independent truth'},
                'frames': frames, 'mask_coverage': fractions}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--walker-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--holdout-cells', action='store_true')
    args = parser.parse_args()
    if args.width <= 0:
        parser.error('width must be positive')
    cv2.setNumThreads(1)
    result = prepare(args)
    print(json.dumps({'frames': len(result['frames']), 'manifest': str(args.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
