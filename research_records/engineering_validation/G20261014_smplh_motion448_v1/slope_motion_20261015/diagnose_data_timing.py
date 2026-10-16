"""Bounded read-only image/content and host-time diagnosis; no tracker changes."""
from pathlib import Path
import csv
import json
import re
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'realtime_app'), str(REPO / 'realtime_app/tools')]
from run_stereo_pva_baseline import _load_predictions, _read_capture_times  # noqa: E402
from pose_app.fisheye_camera import load_stereo_fisheye  # noqa: E402


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'median', 'p95', 'max'),
                    np.percentile(values, (0, 50, 95, 100)).tolist()))


def add_observation_semantics(report):
    """Read raw scalar fields/local source; never decode images or rerun fitting."""
    references = {}

    def source(relative, anchors):
        lines = (REPO / relative).read_text(encoding='utf-8').splitlines()
        found = {name: [{'line': i + 1, 'text': line.strip()}
                        for i, line in enumerate(lines) if anchor in line]
                 for name, anchor in anchors.items()}
        references[relative] = found
        return lines

    dataset_path = 'third_party/bboxmaskpose/mmpose/configs/_base_/datasets/merged_COCO_AIC_MPII_mergable.py'
    dataset_lines = source(dataset_path, {'extra23_channels': 'dict(name=', 'ankles': 'name="left_ankle"'})
    names = {int(index): name for index, name in re.findall(r'^\s+(\d+): dict\(name="([^"]+)"', '\n'.join(dataset_lines), re.MULTILINE)}
    source('third_party/bboxmaskpose/mmpose/configs/ProbMaskPose/PMPose-b-1.0.0.py',
           {'head_channels': 'out_channels=23', 'predicted_oks_enabled': 'freeze_oks=False',
            'dataset_definition': 'merged_COCO_AIC_MPII_mergable.py'})
    source('third_party/bboxmaskpose/mmpose/models/heads/hybrid_heads/multi_head.py',
           {'preserve_heatmap_confidence': '"keypoints_conf"', 'visibility_field': '"keypoints_visible"',
            'oks_field': '"keypoints_oks"', 'score_switch': 'if not self.freeze_oks:',
            'score_is_oks': 'p.set_field(oks[pi], "keypoint_scores")'})
    source('third_party/bboxmaskpose/pmpose/api.py',
           {'get_keypoint_scores': 'kpt_scores = pred_instances.keypoint_scores',
            'get_separate_visibility': 'kpt_visibiliteies = pred_instances.keypoints_visible',
            'concatenate_xy_score': 'kpts_with_scores = np.concatenate([kpts, kpt_scores]'})
    source('realtime_app/pose_app/body_observations.py', {'use_first17': 'if kp.shape == (23,3): kp=kp[:17]'})
    source('realtime_app/pose_app/smplh_motion_tracker.py',
           {'score_gate': 'score = (pixels[..., 2] >= self.config.min_score)',
            'reject_epipolar': 'distance > self.config.max_stereo_epipolar_px'})
    source('realtime_app/pose_app/smplh_iterated_motion_tracker.py',
           {'uniform_pixel_sigma': 'config.pixel_sigma_px**2', 'huber': 'config.huber_px'})
    parameters = json.loads((Path(report['source_run']) / 'run_parameters.json').read_text(encoding='utf-8'))
    config = parameters['configuration']
    raw = {side: json.loads(Path(report['inputs'][f'{side}_predictions']).read_text(encoding='utf-8'))['images']
           for side in ('left', 'right')}
    converted = {side: _load_predictions(Path(report['inputs'][f'{side}_predictions']), side)
                 for side in ('left', 'right')}
    trajectory = [json.loads(line) for line in (Path(report['source_run']) / 'trajectory.jsonl').read_text().splitlines()]
    cal = load_stereo_fisheye(Path(report['inputs']['calibration_dir']))
    tx, ty, tz = cal.T_cam0_to_cam1_mm
    essential = np.array([[0., -tz, ty], [tz, 0., -tx], [-ty, tx, 0.]]) @ cal.R_cam0_to_cam1

    def sampson(frame, joint):
        rays, jacobians = [], []
        for side, K, D in (('left', cal.K0, cal.D0), ('right', cal.K1, cal.D1)):
            xy = converted[side][frame, joint, :2]
            samples = np.r_[xy[None], xy + np.eye(2)*.25, xy - np.eye(2)*.25]
            normalized = cv2.fisheye.undistortPoints(samples.reshape(-1, 1, 2), K, D).reshape(-1, 2)
            rays.append(np.r_[normalized[0], 1.])
            jacobians.append(((normalized[1:3] - normalized[3:])/.5).T)
        gradient = np.r_[jacobians[0].T @ (essential.T @ rays[1])[:2],
                         jacobians[1].T @ (essential @ rays[0])[:2]]
        return float(abs(rays[1] @ essential @ rays[0]) / np.linalg.norm(gradient))

    events = []
    for frame in (35, 36, 37, 38, 58, 59, 60, 61):
        row = {'frame': frame, 'right_knee_sampson_raw_px': sampson(frame, 14),
               'right_knee_gate_pass': sampson(frame, 14) <= config['max_stereo_epipolar_px'],
               'right_knee_used_lr': np.asarray(trajectory[frame]['used_views'])[:, 9].tolist(),
               'right_knee_view_reason_lr': np.asarray(trajectory[frame]['view_reason'])[:, 9].tolist(),
               'per_view': {}}
        for side in ('left', 'right'):
            item = raw[side][frame]
            row['per_view'][side] = {names[joint]: {
                'coco17_index': joint, 'keypoint_score': float(item['keypoints'][0][joint][2]),
                'separate_visibility': float(item['visibility'][0][joint][0]),
                'separate_presence': float(item['presence'][0][joint][0]),
                'raw_fisheye_xy_px': converted[side][frame, joint, :2].tolist()}
                for joint in (11, 12, 14, 15, 16)}
        events.append(row)
    report['observation_semantics'] = {
        'source_scope': 'Existing raw export fields plus inspected local current source/config; no historical Docker runtime revision is independently certified by these checks',
        'raw_exports': {side: {'frames': len(rows), 'keypoint_channels_all_frames': sorted({len(row['keypoints'][0]) for row in rows})}
                        for side, rows in raw.items()},
        'coco17_lower_body_names': {str(i): names[i] for i in range(11, 17)},
        'extra_channels17_22': {str(i): names[i] for i in range(17, 23)},
        'extra_channel_conclusion': 'PMPose23 merges COCO/AIC/MPII; channels17..22 are torso/head landmarks, not toe/heel landmarks. Current first17 selection matches this schema.',
        'score_contract': 'Local PMPose-b config has freeze_oks=False. MultiHead.predict preserves heatmap score as keypoints_conf, exports predicted OKS as keypoint_scores, and exports visibility/presence separately. api concatenates keypoint_scores as xy third channel. Do not interpret it as visibility probability, pixel standard deviation, or calibrated confidence.',
        'current_tracker_score_use': {'min_score': config['min_score'], 'max_score': 1.,
            'pixel_sigma_px': config['pixel_sigma_px'], 'huber_px': config['huber_px'],
            'epipolar_gate_px': config['max_stereo_epipolar_px'],
            'interpretation': 'Finite/bounds/score and stereo gate decide support; admitted points use uniform pixel_sigma8 before residual-dependent Huber downweighting. No score-to-variance calibration and no separate visibility/presence field enter this tracker.'},
        'gate_event_rows': events,
        'event_interpretation': 'Right knee crosses35px support gate36->37 and58->59 while raw scores remain high. Right-knee visibility is also high at these events; low predicted visibility is not supported as its cause. Right-view hip scores remain high while their separate visibility can be low, exposing different scalar meanings without establishing true anatomical visibility.',
        'source_references_1based': references}


def main():
    out = Path(__file__).resolve().parent
    run = out.parent / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    parameters = json.loads((run / 'run_parameters.json').read_text(encoding='utf-8'))
    inputs = parameters['inputs']
    directory = Path(inputs['input_pairs'])
    predictions = {side: _load_predictions(Path(inputs[f'{side}_predictions']), side)
                   for side in ('left', 'right')}
    times, skew, pair_ids, capture_path = _read_capture_times(directory, 448)
    selection = list(csv.DictReader((directory / 'selection_manifest.csv').open(encoding='utf-8-sig')))
    capture = list(csv.DictReader(capture_path.open(encoding='utf-8-sig')))
    metadata = json.loads((capture_path.parent / 'metadata.json').read_text(encoding='utf-8'))
    trajectory = [json.loads(line) for line in (run / 'trajectory.jsonl').read_text().splitlines()]
    report = {
        'question': 'Are distinct host-return records always new image evidence with a known exposure time?',
        'source_run': str(run), 'inputs': inputs,
        'scope': 'All448 saved PMPose keypoints; raw upright PNG frames35..61 only; source AVI verification at35,36,58,59,60,61; no network/model/optimization or observation changes',
        'timestamp_definition': metadata['timestamp_type'],
        'timestamp_warning': metadata['timestamp_warning'],
        'capture_csv_columns': list(capture[0]),
        'frame_csv_columns': {},
        'selection_identity': 'Existing _read_capture_times verifies sequence order and selected frame IDs against source CSV; all448 retained',
        'whole448_host_timing': {'span_s': float(times[-1]), 'dt_ms': stats(np.diff(times) * 1000),
                                 'absolute_host_skew_ms': stats(abs(skew))},
        'whole448_keypoint_adjacent_comparison': {},
        'png_window': {'frame_range_inclusive': [35, 61], 'decoded_images': 54, 'adjacent_rows': []},
        'source_avi_selected_verification': [],
        'code_evidence': {
            'frame_id': 'realtime_app/pose_app/stereo_camera.py:_capture_loop; frame_id=self.captured_count; increment once per successful validated read',
            'timestamp': 'realtime_app/pose_app/stereo_camera.py:_capture_loop; perf_counter_ns immediately after VideoCapture.read',
            'estimator_time': 'realtime_app/tools/run_stereo_pva_baseline.py:_read_capture_times; mean left/right host-return ns minus first pair',
            'pixel_gate': 'realtime_app/pose_app/smplh_motion_tracker.py:_supported_pixels; finite/bounds/score/epipolar only; no image-content or exposure-ID input',
            'extraction': 'realtime_app/walker_tools/stereo/prepare_continuous_stereo_segment.py:main; frame-ID to CSV AVI-position mapping and sequential decoding'},
        'limits': ['Exact saved image equality proves repeated image content, not a specific hardware/driver cause or actual exposure ID.',
                   'Whole448 keypoint equality is a cheap screen; only frames35..61 are verified against image pixels.',
                   'Host skew cannot certify exposure synchronization; no sensor timestamp/exposure counter is stored in the inspected CSV schemas.',
                   'No frame is deleted/replaced and no nominal30fps is substituted.',
                   'This identifies a data-time and independence problem; it does not yet measure its causal effect on ankle height or justify blaming events37/59 on duplicates alone.']}
    for side, array in predictions.items():
        exact = np.all(array[1:] == array[:-1], axis=(1, 2))
        near = np.all(abs(array[1:, :, :2] - array[:-1, :, :2]) <= 1e-3, axis=(1, 2))
        near &= np.all(array[1:, :, 2] == array[:-1, :, 2], axis=1)
        report['whole448_keypoint_adjacent_comparison'][side] = {
            'denominator_adjacent_pairs': 447, 'exact_all17_xy_score_count': int(exact.sum()),
            'exact_frames': (np.flatnonzero(exact) + 1).tolist(),
            'xy_tolerance_0p001px_same_score_count': int(near.sum()),
            'tolerance_frames': (np.flatnonzero(near) + 1).tolist(),
            'exact_dt_ms': stats(np.diff(times)[exact] * 1000)}
    images = {}
    for side, folder in (('left', 'left_ccw90'), ('right', 'right_cw90')):
        previous = None
        for frame in range(35, 62):
            image = cv2.imread(str(directory / folder / f'pair_{frame:04d}.png'))
            if image is None:
                raise ValueError(f'Missing image: {side} {frame}')
            if frame in (35, 36, 58, 59, 60, 61):
                images[(side, frame)] = image.copy()
            if previous is not None:
                difference = np.abs(image.astype(np.int16) - previous.astype(np.int16))
                delta = predictions[side][frame] - predictions[side][frame - 1]
                source = capture[int(selection[frame]['source_pair_index'])]
                report['png_window']['adjacent_rows'].append({
                    'side': side, 'previous_frame': frame - 1, 'frame': frame,
                    'image_equal': bool(np.array_equal(image, previous)),
                    'image_max_abs_channel_difference': int(difference.max()),
                    'image_mean_abs_channel_difference': float(difference.mean()),
                    'keypoint_max_abs_xy_difference_px': float(abs(delta[:, :2]).max()),
                    'keypoint_max_abs_score_difference': float(abs(delta[:, 2]).max()),
                    'host_midpoint_dt_ms': float((times[frame] - times[frame - 1]) * 1000),
                    'signed_host_skew_ms': float(skew[frame]), 'capture_pair_id': int(pair_ids[frame]),
                    'capture_frame_id': int(source[f'{side}_frame_id']),
                    'read_duration_ms': float(source[f'{side}_read_duration_ms']),
                    'right_knee_used_lr': np.asarray(trajectory[frame]['used_views'])[:, 9].tolist()})
            previous = image
    for side in ('left', 'right'):
        csv_path = capture_path.parent / f'{side}_frames.csv'
        frames = list(csv.DictReader(csv_path.open(encoding='utf-8-sig')))
        report['frame_csv_columns'][side] = list(frames[0])
        positions = {int(row['frame_id']): index for index, row in enumerate(frames)}
        targets = {}
        for frame in (35, 36, 58, 59, 60, 61):
            frame_id = int(selection[frame][f'{side}_frame_id'])
            targets[positions[frame_id]] = (frame, frame_id)
        video = cv2.VideoCapture(str(capture_path.parent / f'{side}_capture.avi'))
        video.set(cv2.CAP_PROP_POS_FRAMES, min(targets))
        try:
            for position in range(min(targets), max(targets) + 1):
                ok, image = video.read()
                if not ok:
                    raise ValueError(f'Cannot decode source {side} AVI position {position}')
                if position in targets:
                    frame, frame_id = targets[position]
                    upright = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE if side == 'left' else cv2.ROTATE_90_CLOCKWISE)
                    report['source_avi_selected_verification'].append({
                        'side': side, 'frame': frame, 'capture_frame_id': frame_id,
                        'avi_position': position, 'equals_saved_png': bool(np.array_equal(upright, images[(side, frame)]))})
        finally:
            video.release()
    rows = report['png_window']['adjacent_rows']
    report['png_window']['summary'] = {side: {
        'adjacent_pairs': 26, 'pixel_identical_count': sum(r['image_equal'] for r in rows if r['side'] == side),
        'pixel_identical_frames': [r['frame'] for r in rows if r['side'] == side and r['image_equal']]}
        for side in ('left', 'right')}
    trace_path = out / 'motion_chain_trace.json'
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    repeated = report['png_window']['summary']['left']['pixel_identical_frames']
    repeated_rows = [row for row in trace['rows'] if row['frame'] in repeated]
    report['existing_trace_on_repeated_content'] = {
        'source': str(trace_path),
        'scope': 'Reads existing exact-baseline trace only; no counterfactual replay',
        'rows': [{key: row[key] for key in ('frame', 'prediction_height_increment_mm',
                                           'update_height_increment_mm')}
                 for row in repeated_rows],
        'prediction_mean_ankle_height_increment_sum_mm': sum(row['prediction_height_increment_mm'] for row in repeated_rows),
        'update_mean_ankle_height_increment_sum_mm': sum(row['update_height_increment_mm'] for row in repeated_rows),
        'interpretation': 'All8 repeated-content updates lower mean ankle height locally; they are not the two direct rise steps. Their counterfactual influence on covariance/history is unmeasured.'}
    report['conclusion'] = 'Distinct read counters and host-return times demonstrably include repeated image content. Current update treats every valid supplied pixel record as another measurement; independence and physical derivative time are not established. The inspected local image repeats occur before37 and after59, while35->37 and58->59 contain new image content; duplicate content alone is not a demonstrated source of the two rise events.'
    path = out / 'data_timing_diagnostic.json'
    add_observation_semantics(report)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(path), 'png_summary': report['png_window']['summary'],
                      'avi_matches': sum(r['equals_saved_png'] for r in report['source_avi_selected_verification']),
                      'avi_checked': len(report['source_avi_selected_verification'])}, ensure_ascii=False))


if __name__ == '__main__':
    if '--append-semantics-only' in sys.argv:
        path = Path(__file__).resolve().parent / 'data_timing_diagnostic.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        add_observation_semantics(report)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Appended observation_semantics only: {path}')
    else:
        main()
