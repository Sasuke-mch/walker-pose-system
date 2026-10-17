"""Shared export helpers for the skeleton/SMPL-H viewer build CLIs.

These are the encoding helpers and the same-image strict/Stage/ground scene
preparation used by more than one viewer CLI. Each build CLI keeps only its own
argument parsing and orchestration; the shared encoding and scene preparation
live here.

The scene preparation deliberately consumes only same-image raw observations
and frozen static assets; it never reads old dynamic Stage/ground arrays. The
strict association gates and replay command semantics are unchanged from the
original viewer implementation.
"""
from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from pose_app.calibration import StereoCalibration
from pose_app.stereo_prediction_records import records as _records, result as _result
from pose_app.triangulation import triangulate_matches

APP = Path(__file__).resolve().parents[2]
REPO = APP.parent


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def finite(values):
    return np.where(np.isfinite(values), values, None).tolist()


def packed(values):
    return base64.b64encode(np.ascontiguousarray(values, dtype=np.float32).tobytes()).decode('ascii')


def prepare_scene(root, output):
    """Only raw observations/static assets enter this new Stage replay."""
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    inputs = protocol['tracking_inputs']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'scene').exists():
        raise FileExistsError('fresh scene already exists; use --export-only to reuse this task replay')
    points = {side: _records(Path(inputs[f'{side}_predictions']), 'pmpose', 'score') for side in ('left', 'right')}
    cal_path = Path(inputs['calibration_dir']) / 'stereo_fisheye.json'
    cal = StereoCalibration.load(cal_path).for_runtime_sizes((1920, 1080), (1920, 1080))
    strict = []
    for i, (left, right) in enumerate(zip(points['left'], points['right'])):
        pair = []
        for frame, rotation in ((left, 'ccw90'), (right, 'cw90')):
            if frame['id'] != i or frame['name'] != f'pair_{i:04d}.png':
                raise ValueError('detector/image frame identity differs')
            pair.append(_result(dict(frame, persons=frame['persons'][:1]), 'pmpose', (1920, 1080), rotation, i/30))
        people = triangulate_matches(pair[0].persons, pair[1].persons, cal, .25, .05, 10., max_matches=1)
        strict.append({'index': i, 'original_pair_id': i, 'pair_outcome': 'associated' if people else 'association_rejected',
                       'persons_3d': [person.to_dict() for person in people]})
    strict_path = output / 'strict.jsonl'
    strict_path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in strict), encoding='utf-8')
    pairs = Path(inputs['input_pairs'])
    with (pairs / 'selection_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        selection = list(csv.DictReader(stream))
    with Path(protocol['capture_times']).open(encoding='utf-8-sig', newline='') as stream:
        capture = list(csv.DictReader(stream))
    frame_times = []
    for i, selected in enumerate(selection):
        row = capture[int(selected['source_pair_index'])]
        frame_times.append({'index': i, 'left_image': str(pairs/'left_ccw90'/selected['file_name']),
            'right_image': str(pairs/'right_cw90'/selected['file_name']),
            'left_host_return_timestamp_ns': int(row['left_host_return_timestamp_ns']),
            'right_host_return_timestamp_ns': int(row['right_host_return_timestamp_ns']),
            'signed_host_delta_ms_right_minus_left': float(row['signed_host_delta_ms_right_minus_left'])})
    time_path = output / 'frame_times.json'
    save(time_path, frame_times)
    # Read the frozen static replay configuration, never any old dynamic arrays.
    old = json.loads((root.parent/'slope_motion_20261015/scene_replay_command.json').read_text(encoding='utf-8'))
    argv = old['argv'].copy()
    for flag, value in (('--strict-jsonl', strict_path), ('--frames-metadata', time_path), ('--output', output/'scene')):
        argv[argv.index(flag)+1] = str(value.resolve())
    save(output/'scene_replay_command.json', {'executable': sys.executable, 'argv': argv,
        'purpose': 'New visualization task: fresh strict/Stage/ground from the same ordered images;no body tracking/model fit',
        'old_dynamic_inputs_consumed': False})
    with (output/'scene_replay.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, *argv], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'scene replay failed; retained log: {output / "scene_replay.log"}')
    print('fresh strict/Stage/ground replay completed', len(strict), flush=True)
