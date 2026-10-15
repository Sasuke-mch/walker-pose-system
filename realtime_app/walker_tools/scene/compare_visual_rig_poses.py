"""Compare native stereo world-pose outputs without Stage or pose fitting on test tracks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.static_background_stereo_frontend import (  # noqa: E402
    StereoFrontendCriteria, mutual_ratio_matches, triangulate_indexed_matches,
)


def local_path(value: str) -> Path:
    if sys.platform == 'win32' and value.startswith('/mnt/'):
        return Path(value[5].upper() + ':/' + value[7:])
    return Path(value)


def read_rows(path: Path, frames: list[dict]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['frame_id'] for row in rows] != [frame['frame_id'] for frame in frames]:
        raise ValueError(f'{path}: missing, repeated, reordered or foreign frame IDs')
    if [row['pair_id'] for row in rows] != [frame['pair_id'] for frame in frames]:
        raise ValueError(f'{path}: source pair IDs differ')
    return rows


def measured_pose(row: dict) -> np.ndarray | None:
    value = row.get('world_from_left_camera_m')
    if value is None or row.get('status') != 'tracked':
        return None
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Backend returned malformed measured pose')
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Backend returned invalid homogeneous row')
    R = pose[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
        raise ValueError('Backend returned invalid measured SO(3)')
    return pose


def same_segment(a: dict, b: dict) -> bool:
    return a.get('map_segment_id', a.get('segment_id', 0)) == b.get('map_segment_id', b.get('segment_id', 0))


def stats(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'count': int(len(values)), 'median': float(np.median(values)) if len(values) else None,
            'p95': float(np.percentile(values, 95)) if len(values) else None,
            'max': float(np.max(values)) if len(values) else None}


def longest_run(rows: list[dict]) -> int:
    best = length = 0
    previous = None
    for row in rows:
        if measured_pose(row) is None:
            length = 0
        else:
            length = length + 1 if previous is not None and same_segment(previous, row) else 1
            best = max(best, length)
        previous = row
    return best


def support_summary(rows: list[dict]) -> dict:
    """Count exported per-frame support; this is not the complete native map."""
    support_counts, shared_counts = [], []
    for row in rows:
        if measured_pose(row) is None:
            continue
        observations = row.get('observations', [])
        if observations and isinstance(observations[0], list):
            support = row.get('finite_landmark_count')
            left = {item['id'] for item in observations[0]}
            right = {item['id'] for item in observations[1]} if len(observations) > 1 else set()
        else:
            inliers = [item for item in observations if not item.get('outlier', False)]
            left = {item['map_point_id'] for item in inliers if item['camera_index'] == 0}
            right = {item['map_point_id'] for item in inliers if item['camera_index'] == 1}
            support = len(left | right)
        if support is not None:
            support_counts.append(support)
        shared_counts.append(len(left & right))
    return {'exported_3d_support_per_pose_frame': stats(support_counts),
            'same_frame_left_right_shared_observation_ids': stats(shared_counts),
            'scope': 'cuVSLAM last-frame finite 3D tracks; ORB distinct inlier map-point IDs; not full-map counts or calibrated uncertainty'}


def angle_deg(R: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def extract(frame: dict, calibration: StereoCalibration, criteria: StereoFrontendCriteria) -> dict:
    """The diagnostic pixels are identified independently of either estimated trajectory."""
    features = []
    excluded_depth = []
    orb = cv2.ORB_create(nfeatures=criteria.orb_features, fastThreshold=10, edgeThreshold=19)
    for side in ('left', 'right'):
        gray = cv2.imread(str(local_path(frame[f'{side}_path'])), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(local_path(frame[f'{side}_test_mask_path'])), cv2.IMREAD_GRAYSCALE)
        excluded = cv2.imread(str(local_path(frame[f'{side}_mask_path'])), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or excluded is None:
            raise FileNotFoundError(f'Missing diagnostic input at {frame["frame_id"]}/{side}')
        excluded_depth.append(cv2.distanceTransform((excluded != 0).astype(np.uint8), cv2.DIST_L2, 3))
        kp, desc = orb.detectAndCompute(gray, mask)
        features.append((tuple(kp or []), desc))
    (left, dl), (right, dr) = features
    stereo = mutual_ratio_matches(dl, dr, ratio=criteria.ratio_test)
    xyz, counts = triangulate_indexed_matches(calibration, left, right, stereo, criteria)
    paired = {m.queryIdx: m.trainIdx for m in stereo if m.queryIdx in xyz}
    return {'left': left, 'right': right, 'dl': dl, 'dr': dr, 'xyz': xyz,
            'paired': paired, 'counts': counts, 'excluded_depth': excluded_depth}


def observation_mask_diagnostics(row: dict, excluded_depth: list[np.ndarray]) -> dict:
    """Separate deeply excluded native observations from subpixel boundary overlap."""
    observations = row.get('observations', [])
    if observations and isinstance(observations[0], list):
        views = observations
    else:
        views = [[item for item in observations if item['camera_index'] == side]
                 for side in (0, 1)]
    results = []
    for view, depth in zip(views, excluded_depth):
        distances = []
        for item in view:
            x, y = int(round(item['u'])), int(round(item['v']))
            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                distances.append(float(depth[y, x]))
        results.append({'native_observation_count': len(view),
                        'rounded_center_excluded_count': sum(value > 0 for value in distances),
                        'deep_excluded_center_count_gt_3px': sum(value > 3 for value in distances),
                        'excluded_depth_px': stats([value for value in distances if value > 0])})
    return {'per_camera': results,
            'scope': 'Actual native observation centers versus supplied exclusion mask; not a correction to native pose'}


def observations(previous: dict, current: dict, criteria: StereoFrontendCriteria) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_matches = mutual_ratio_matches(previous['dl'], current['dl'], ratio=criteria.ratio_test)
    right_links = {m.queryIdx: m.trainIdx for m in
                   mutual_ratio_matches(previous['dr'], current['dr'], ratio=criteria.ratio_test)}
    xyz, left_pixels, right_pixels = [], [], []
    for match in left_matches:
        source = match.queryIdx
        if source not in previous['paired']:
            continue
        right_source = previous['paired'][source]
        if right_source not in right_links:
            continue
        xyz.append(previous['xyz'][source])
        left_pixels.append(current['left'][match.trainIdx].pt)
        right_pixels.append(current['right'][right_links[right_source]].pt)
    return (np.asarray(xyz).reshape(-1, 3), np.asarray(left_pixels).reshape(-1, 2),
            np.asarray(right_pixels).reshape(-1, 2))


def projected_errors(delta: np.ndarray, xyz: np.ndarray, left: np.ndarray, right: np.ndarray,
                     cal: StereoCalibration) -> tuple[list[float], int]:
    current_xyz = xyz @ delta[:3, :3].T + delta[:3, 3] * 1000
    current_right = current_xyz @ cal.R.T + cal.T
    valid = np.isfinite(current_xyz).all(axis=1) & (current_xyz[:, 2] > 0) & (current_right[:, 2] > 0)
    # Do not reject large residuals according to either backend's prediction.
    if not np.any(valid):
        return [], int(len(xyz))
    errors = np.maximum(np.linalg.norm(cal.project_left(current_xyz[valid]) - left[valid], axis=1),
                        np.linalg.norm(cal.project_right(current_xyz[valid]) - right[valid], axis=1))
    return errors.tolist(), int((~valid).sum())


def export_world_geometry(rows: list[dict], body_fit: Path, walker_model: Path,
                          cal: StereoCalibration, output: Path) -> None:
    """Use one native visual transform for body, rigid walker and both optical centers."""
    model = json.loads(walker_model.read_text(encoding='utf-8'))
    names = list(model['nodes_walker_mm'])
    nodes = np.array([model['nodes_walker_mm'][name] for name in names]) / 1000
    R = np.array(model['rotation_left_camera_from_walker'])
    t = np.array(model['translation_left_camera_from_walker_mm']) / 1000
    walker_camera = nodes @ R.T + t
    centers_camera = np.array([[0., 0., 0.], -cal.R.T @ cal.T / 1000])
    with np.load(body_fit, allow_pickle=False) as source:
        body = np.array(source['predicted_coco'])
        ids = np.array(source['pair_id'])
        vertices = np.array(source['vertices'])
        faces = np.array(source['faces'])
    if body.shape != (len(rows), 17, 3) or not np.array_equal(ids, [row['pair_id'] for row in rows]):
        raise ValueError('Frozen body and visual source pair IDs differ')
    world_body = np.full_like(body, np.nan)
    world_vertices = np.full_like(vertices, np.nan)
    world_walker = np.full((len(rows), len(nodes), 3), np.nan)
    world_centers = np.full((len(rows), 2, 3), np.nan)
    transforms = np.full((len(rows), 4, 4), np.nan)
    for index, row in enumerate(rows):
        pose = measured_pose(row)
        if pose is None:
            continue
        transforms[index] = pose
        world_body[index] = body[index] @ pose[:3, :3].T + pose[:3, 3]
        world_vertices[index] = vertices[index] @ pose[:3, :3].T + pose[:3, 3]
        world_walker[index] = walker_camera @ pose[:3, :3].T + pose[:3, 3]
        world_centers[index] = centers_camera @ pose[:3, :3].T + pose[:3, 3]
    np.savez_compressed(output, pair_id=ids, world_from_left_camera_m=transforms,
                        world_body_coco_m=world_body, world_body_vertices_m=world_vertices, faces=faces,
                        world_walker_nodes_m=world_walker,
                        walker_node_names=np.array(names), world_camera_centers_m=world_centers,
                        map_segment_id=np.array([row.get('map_segment_id', row.get('segment_id', 0)) for row in rows]),
                        status=np.array([row['status'] for row in rows]),
                        coordinate_frame='native_visual_map_per_segment_not_certified_ground',
                        world_pose_certified=False, metric_pose_usable=False, stage_used=False)


def compare(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite an existing comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    frames = manifest['frames']
    calibration = manifest['calibration']
    if calibration['model'] != 'opencv_fisheye':
        raise ValueError('raw-fisheye diagnostics and frozen raw-camera mesh export cannot consume rectified poses; use the explicit original-camera pose bridge')
    size = tuple(calibration['image_size'])
    cal = StereoCalibration('fisheye', size, size, np.array(calibration['left_K']), np.array(calibration['left_D']),
                            np.array(calibration['right_K']), np.array(calibration['right_D']),
                            np.array(calibration['R_right_from_left']),
                            np.array(calibration['translation_right_from_left_m']) * 1000, 'mm')
    methods = {'orbslam3': read_rows(args.orbslam, frames), 'cuvslam': read_rows(args.cuvslam, frames)}
    criteria = StereoFrontendCriteria(orb_features=6000)
    residuals = {name: [] for name in methods}
    translations = {name: [] for name in methods}
    rotations = {name: [] for name in methods}
    edges_used = Counter()
    point_counts = Counter()
    invalid_counts = Counter()
    mask_deep_counts = {name: [0, 0] for name in methods}
    mask_observation_counts = {name: [0, 0] for name in methods}
    common_deltas = []
    diagnostics = []
    previous = None
    for index, frame in enumerate(frames):
        current = extract(frame, cal, criteria)
        row = {'frame_id': frame['frame_id'], 'stereo_geometry': current['counts'], 'edge_test_point_count': 0, 'methods': {}, 'native_mask_diagnostics': {}}
        for name, records in methods.items():
            mask_diagnostic = observation_mask_diagnostics(records[index], current['excluded_depth'])
            row['native_mask_diagnostics'][name] = mask_diagnostic
            for side, values in enumerate(mask_diagnostic['per_camera']):
                mask_deep_counts[name][side] += values['deep_excluded_center_count_gt_3px']
                mask_observation_counts[name][side] += values['native_observation_count']
        if previous is not None:
            xyz, left, right = observations(previous, current, criteria)
            row['edge_test_point_count'] = len(xyz)
            deltas = {}
            for name, records in methods.items():
                old_pose, new_pose = measured_pose(records[index - 1]), measured_pose(records[index])
                if old_pose is None or new_pose is None or not same_segment(records[index - 1], records[index]):
                    row['methods'][name] = {'status': 'unavailable_or_new_map'}
                    continue
                delta = np.linalg.inv(new_pose) @ old_pose
                deltas[name] = delta
                translations[name].append(float(np.linalg.norm(delta[:3, 3]) * 1000))
                rotations[name].append(angle_deg(delta[:3, :3]))
                errors, invalid = projected_errors(delta, xyz, left, right, cal)
                point_counts[name] += len(xyz)
                invalid_counts[name] += invalid
                residuals[name].extend(errors)
                if len(errors) >= 4:
                    edges_used[name] += 1
                row['methods'][name] = {'status': 'measured', 'reprojection_px': stats(errors),
                                        'prediction_nonpositive_depth_count': invalid}
            if len(deltas) == 2:
                difference = np.linalg.inv(deltas['orbslam3']) @ deltas['cuvslam']
                common_deltas.append({'frame_id': frame['frame_id'],
                                      'translation_delta_disagreement_mm': float(np.linalg.norm(difference[:3, 3]) * 1000),
                                      'rotation_delta_disagreement_deg': angle_deg(difference[:3, :3])})
        diagnostics.append(row)
        previous = current
        if index % 64 == 0:
            print(f'diagnostic frame {index}/{len(frames)}', flush=True)
    summary = {'frames': len(frames), 'stage_used': False,
               'measurement_scope': manifest['holdout']['scope'], 'world_pose_certified': False,
               'tracked_pose_scope': 'Native pose proposal with nonempty exported 3D support; not a certified metric pose or proof of full observability',
               'timestamp_source': manifest.get('timestamp_source'),
               'promotion_gate': {'tracking_fraction_minimum': .9, 'tracking_call_p95_ms_maximum': 33.333,
                                  'diagnostic_edge_points_minimum': 4,
                                  'metric_translation_accuracy': 'requires independent trajectory, unavailable'},
               'methods': {}, 'common_measured_edges': len(common_deltas)}
    for name, rows in methods.items():
        count = sum(measured_pose(row) is not None for row in rows)
        track_ms = [row['track_ms'] for row in rows if row.get('track_ms') is not None]
        all_segments = {row.get('map_segment_id', row.get('segment_id', 0)) for row in rows if measured_pose(row) is not None}
        tracking_times = stats(track_ms)
        timing_passed = tracking_times['p95'] is not None and tracking_times['p95'] <= 33.333
        summary['methods'][name] = {'measured_frames': count, 'tracking_fraction': count / len(rows),
                                   'status_counts': dict(Counter(row['status'] for row in rows)),
                                   'conservative_output_segments': len(all_segments), 'longest_measured_run_frames': longest_run(rows),
                                   'tracking_call_ms': tracking_times,
                                   'frame_total_ms_if_native_available': stats([row['frame_total_ms'] for row in rows if row.get('frame_total_ms') is not None]),
                                   'tracking_call_amortized_fps': 1000 / np.mean(track_ms) if track_ms else None,
                                   'reprojection_px': stats(residuals[name]),
                                   'diagnostic_point_count': point_counts[name],
                                   'prediction_nonpositive_depth_count': invalid_counts[name],
                                   'evaluated_point_fraction': len(residuals[name]) / point_counts[name] if point_counts[name] else None,
                                   'edges_with_at_least_four_diagnostic_points': edges_used[name],
                                   'translation_step_mm': stats(translations[name]), 'rotation_step_deg': stats(rotations[name]),
                                   'coverage_gate_passed': count / len(rows) >= .9,
                                   'tracking_call_timing_gate_passed': timing_passed,
                                   'coverage_and_tracking_time_gate_passed': count / len(rows) >= .9 and timing_passed,
                                   'exported_geometry_support': support_summary(rows),
                                   'native_observation_counts_by_camera': mask_observation_counts[name],
                                   'deep_excluded_native_observation_centers_gt_3px_by_camera': mask_deep_counts[name],
                                   'numerically_zero_translation_edges': sum(step <= 1e-5 for step in translations[name]),
                                   'zero_translation_interpretation': 'May be genuine stationary motion, solver hold or prediction; native update state is insufficient to decide',
                                   'independent_accuracy_comparison': 'unavailable', 'promoted': False}
    summary['relative_method_disagreement'] = {
        'translation_mm': stats([row['translation_delta_disagreement_mm'] for row in common_deltas]),
        'rotation_deg': stats([row['rotation_delta_disagreement_deg'] for row in common_deltas])}
    if args.body_fit is not None:
        if args.walker_model is None:
            raise ValueError('body export requires the fixed walker model')
        for name, rows in methods.items():
            export_world_geometry(rows, args.body_fit, args.walker_model, cal, args.output / f'{name}_world_geometry.npz')
        summary['frozen_body_fit'] = str(args.body_fit.resolve())
        summary['world_geometry_scope'] = 'Same native visual map transform for frozen real body mesh, COCO17, walker and optical centers; failed frames NaN, new maps kept separate'
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with (args.output / 'background_diagnostics.jsonl').open('w', encoding='utf-8') as stream:
        for row in diagnostics:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    (args.output / 'method_disagreement.json').write_text(json.dumps(common_deltas, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--orbslam', type=Path, required=True)
    parser.add_argument('--cuvslam', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--body-fit', type=Path)
    parser.add_argument('--walker-model', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = compare(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
