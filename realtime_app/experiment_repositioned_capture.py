"""Run the existing cold SMPL-H route on the confirmed September 27 capture.

Only dataset resources, frame count and the authorized absent wrist reference
change. Pose runners, ROI, WiLoR and fitting arithmetic remain existing code.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

from pose_app.person_roi import RULE, foot_inclusive_box
from walker_tools.pose2d.compare_sapiens2_mainline import infer, sapiens_points

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT/'realtime_app/outputs/stereo_capture/20260927_201716_220'
EXTRACT = ROOT/'realtime_app/outputs/pipeline_20260927_201716_pmpose_mainline/upright_extract'
STEREO = ROOT/'realtime_app/calibration/results/stereo_fisheye_20260927_195919.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(command, log, cwd=ROOT):
    write(log.parent/'active_command.json', {'command': command, 'cwd': str(cwd), 'log': str(log)})
    started = time.perf_counter()
    with log.open('w', encoding='utf-8') as h:
        subprocess.run(command, cwd=cwd, stdout=h, stderr=subprocess.STDOUT, check=True)
    write(log.with_suffix('.execution.json'), {'command': command, 'cwd': str(cwd),
          'seconds': time.perf_counter()-started, 'status': 'completed',
          'includes_process_start_and_model_loading': True})


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    pairs = list(csv.DictReader((CAPTURE/'stereo_pairs.csv').open(encoding='utf-8-sig')))
    assert len(pairs) == 360
    assert [int(r['pair_id']) for r in pairs] == list(range(1157, 1517))
    metadata = read(CAPTURE/'metadata.json')
    assert metadata['calibration_camera_mapping']['selection_mode'] == 'physical_registry'
    assert metadata['calibration_camera_mapping']['left']['logical_camera'] == 'cam0'
    assert metadata['calibration_camera_mapping']['right']['logical_camera'] == 'cam1'
    identity = metadata['calibration_camera_mapping']['camera_registry_resolution']
    assert identity['left']['expected_instance_id'] != identity['right']['expected_instance_id']
    frames = []
    for i, row in enumerate(pairs):
        frames.append({'index': i, **row})
    verified = {}
    for side, rotation in (('left', cv2.ROTATE_90_COUNTERCLOCKWISE), ('right', cv2.ROTATE_90_CLOCKWISE)):
        folder = f'{side}_{"ccw90" if side == "left" else "cw90"}'
        input_dir = out/'input'/folder
        input_dir.mkdir(parents=True)
        frame_rows = list(csv.DictReader((CAPTURE/f'{side}_frames.csv').open(encoding='utf-8-sig')))
        by_id = {int(r['frame_id']): j for j, r in enumerate(frame_rows)}
        assert len(by_id) == len(frame_rows)
        wanted = {by_id[int(row[side+'_frame_id'])]: i for i, row in enumerate(pairs)}
        assert len(wanted) == 360
        cap = cv2.VideoCapture(str(CAPTURE/f'{side}_capture.avi'))
        n, checked = 0, 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if n in wanted:
                index = wanted[n]
                path = EXTRACT/folder/f'pair_{int(pairs[index]["pair_id"]):04d}.png'
                saved = cv2.imread(str(path))
                assert image.shape == (1080, 1920, 3)
                assert saved is not None and np.array_equal(cv2.rotate(image, rotation), saved)
                frames[index][side+'_avi_index'] = n
                target = input_dir/f'pair_{index:04d}.png'
                shutil.copyfile(path, target)
                frames[index][side+'_source_image'] = str(path)
                frames[index][side+'_image'] = str(target)
                checked += 1
            n += 1
        cap.release()
        assert n == len(frame_rows) and checked == 360
        verified[side] = {'avi_frames': n, 'paired_images_exact': checked}
    cal = out/'calibration'
    cal.mkdir()
    shutil.copyfile(STEREO, cal/'stereo_fisheye_source.json')
    portable = read(STEREO)
    portable['cam0_intrinsics'] = 'cam0_fisheye.json'
    portable['cam1_intrinsics'] = 'cam1_fisheye.json'
    write(cal/'stereo_fisheye.json', portable)
    for name in ('cam0_fisheye.json', 'cam1_fisheye.json'):
        shutil.copyfile(STEREO.parent/name, cal/name)
    write(out/'frames.json', frames)
    write(out/'protocol.json', {'capture': str(CAPTURE), 'extrinsic': str(STEREO),
        'calibration': str(cal), 'frames': 360, 'pair_ids': [1157, 1516], 'mapping': 'frames.json',
        'source_images': str(EXTRACT), 'images': str(out/'input'), 'image_audit': verified, 'metadata': metadata,
        'detector': {'weights': 'models/yolo26/yolo26x.pt', 'imgsz': 1280, 'conf': .05, 'iou': .7},
        'roi': RULE, 'transport': 'same decoded PNG for both pose models, no JPEG transport',
        'pose': {'PMPose': 'PMPose-b original bbox-mask benchmark', 'Sapiens2': '0.4B original runner, flip_test=True, batch2'},
        'fit_profile': 'realtime_app/configs/cold_smplh_joint300.json',
        'authorized_difference': 'No new manual wrist reference; both wrist terms disabled per user confirmation',
        'wilor': 'new capture inference; original native MANO local pose; shared hand PCA12',
        'initialization': 'shared beta zeros; zero latent; current observation-derived root and translation',
        'steps': {'A': 30, 'B': 30, 'C': 300, 'D1': 100, 'D2': 100, 'D3': 0},
        'excluded': ['old fitted parameters', 'old wrist reference', 'old hand predictions', 'body temporal loss', 'contact loss'],
        'accepted_for_main_fit': False,
        'success': 'complete all360 frames for both detectors and same solver; audited controls and rejection states',
        'stop': 'frame/calibration/coordinate mismatch, missing primary body observation, active nonfinite optimization or resource failure',
        'boundary': 'No manual or independent 3D truth on this sequence; engineering consistency only',
        'seed': 0,
        'instrumentation': 'Existing solver observer saves stage A/B/C joint snapshots; no loss or gradient change',
        'calibration_copy': 'Original source bytes retained; usable copy changes only cam0/cam1 relative intrinsic references',
        'code': ['realtime_app/experiment_repositioned_capture.py','realtime_app/walker_tools/pose2d/compare_sapiens2_mainline.py',
                 'realtime_app/walker_tools/hands/run_wilor_sequence.py','realtime_app/pose_app/smplh_fitting/pipeline.py']})
    write(out/'preparation_audit.json', {'pairs': 360, 'exact_avi_png': verified, 'registered_camera_roles': True})


def detect(out):
    for side, folder in (('left', 'left_ccw90'), ('right', 'right_cw90')):
        path = out/f'{side}_detections.json'
        if path.exists():
            raise FileExistsError(path)
        container_root = '/project/'+str(out.relative_to(ROOT)).replace('\\', '/')
        run(['docker', 'run', '--rm', '--gpus', 'all', '--shm-size', '8g',
             '-v', str(ROOT)+':/project', '-w', '/project', 'yolo26-realtime:latest',
             'python', 'sequence_pipeline/container_tasks/yolo26_detector.py',
             '--input-dir', container_root+'/input/'+folder,
             '--weights', '/project/models/yolo26/yolo26x.pt',
             '--output-json', container_root+f'/{side}_detector_unexpanded.json',
             '--device', '0', '--imgsz', '1280', '--conf', '0.05', '--iou', '0.7'],
            out/f'{side}_detector.log')
        predictions = read(out/f'{side}_detector_unexpanded.json')['images']
        assert len(predictions) == 360
        entries = []
        for i, frame in enumerate(read(out/'frames.json')):
            image_path = Path(frame[side+'_image'])
            assert predictions[i]['file_name'] == image_path.name
            detections = []
            for detection in predictions[i]['detections']:
                box = detection['bbox_xyxy']
                expanded, roi_stats = foot_inclusive_box(box, 1080, 1920, RULE['threshold'], RULE['power'])
                detections.append({'class_id': 0, 'class_name': 'person', 'bbox_xyxy': expanded,
                    'score': detection['score'], 'original_bbox': box, 'roi_stats': roi_stats})
            if not detections:
                raise ValueError(f'no person detection at {side}/{i}; no silent completion')
            entries.append({'image_id': i, 'frame_index': i, 'pair_id': int(frame['pair_id']),
                'file_name': image_path.name, 'image_path': str(image_path), 'width': 1080, 'height': 1920,
                'detections': detections})
            if i % 30 == 0:
                print(side, i, 'detected', len(detections), flush=True)
        write(path, {'images': entries})
        container = copy.deepcopy(entries)
        for row in container:
            row['image_path'] = '/project/'+str(Path(row['image_path']).relative_to(ROOT)).replace('\\', '/')
        write(out/f'{side}_detections_container.json', {'images': container})


def pmpose(out):
    for side in ('left', 'right'):
        target = out/f'{side}_pmpose_raw.json'
        if target.exists():
            raise FileExistsError(target)
        container_root = '/project/'+str(out.relative_to(ROOT)).replace('\\', '/')
        command = ['docker', 'run', '--rm', '--gpus', 'all', '--shm-size', '8g',
            '-v', str(ROOT)+':/project', '-v', str(ROOT/'third_party/bboxmaskpose')+':/workspace/BBoxMaskPose',
            '-w', '/workspace/BBoxMaskPose', '-e', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1', 'bboxmaskpose:latest',
            'python', 'tools/benchmark_from_yolo26_bboxes.py', '--det-json', container_root+f'/{side}_detections_container.json',
            '--method', 'pmpose', '--output-csv', container_root+f'/{side}_pmpose_timing.csv',
            '--summary-csv', container_root+f'/{side}_pmpose_summary.csv', '--save-pred-json', container_root+f'/{side}_pmpose_raw.json',
            '--device', 'cuda:0', '--pose-variant', 'PMPose-b', '--warmup', '0', '--repeat', '1',
            '--pose-checkpoint', '/project/models/pmpose/PMPose-b-1.0.0.pth']
        run(command, out/f'{side}_pmpose.log')


def sapiens(out):
    infer(SimpleNamespace(output=out))
    for side in ('left', 'right'):
        rows = [json.loads(s) for s in (out/f'{side}_sapiens_raw.jsonl').read_text().splitlines()]
        assert len(rows) == 360
        images = []
        for i, row in enumerate(rows):
            if not row['instances']:
                raise ValueError(f'no Sapiens person at {side}/{i}')
            images.append({'image_id': i, 'file_name': f'pair_{i:04d}.png',
                'keypoints': [sapiens_points(instance).tolist() for instance in row['instances']]})
        write(out/f'{side}_sapiens_coco17_model_input.json', {'images': images})


def wilor(out):
    for side, folder in (('left', 'left_ccw90'), ('right', 'right_cw90')):
        run([sys.executable, '-B', '-u', 'realtime_app/tools/run_wilor_sequence.py',
             '--image-dir', str(out/'input'/folder), '--output', str(out/f'wilor_{side}.jsonl'),
             '--orientation', folder, '--device', 'cuda'], out/f'wilor_{side}.log')


def fit_commands(out):
    commands, controls = {}, {}
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser
    for method in ('pmpose', 'sapiens2'):
        settings = read(ROOT/'realtime_app/configs/cold_smplh_joint300.json')
        for flag in ('--wrist-reference', '--wrist-reference-weight'):
            index = settings.index(flag)
            del settings[index:index+2]
        settings.remove('--allow-diagnostic-wrist-reference')
        body = {s: out/(f'{s}_pmpose_raw.json' if method == 'pmpose' else f'{s}_sapiens_coco17_model_input.json') for s in ('left', 'right')}
        changes = {'--left-raw': body['left'], '--right-raw': body['right'],
            '--wilor-left': out/'wilor_left.jsonl', '--wilor-right': out/'wilor_right.jsonl',
            '--calibration-dir': out/'calibration', '--output-dir': out/method/'fit'}
        for flag, value in changes.items():
            settings[settings.index(flag)+1] = str(value)
        options = build_parser().parse_args(settings)
        assert options.wrist_reference is None and options.wrist_reference_weight == 0
        assert options.body_temporal_weight == options.hand_2d_weight == 0
        controls[method] = {k: str(v) if isinstance(v, Path) else v for k,v in vars(options).items()
                            if k not in ('left_raw', 'right_raw', 'output_dir')}
        commands[method] = [sys.executable, '-B', '-u', 'realtime_app/tools/fit_smplh_wilor_sequence.py', *settings]
    assert controls['pmpose'] == controls['sapiens2']
    reference = build_parser().parse_args(read(ROOT/'realtime_app/configs/cold_smplh_joint300.json'))
    allowed = {'left_raw', 'right_raw', 'wilor_left', 'wilor_right', 'calibration_dir', 'output_dir',
               'wrist_reference', 'wrist_reference_weight', 'allow_diagnostic_wrist_reference'}
    for key, value in vars(reference).items():
        if key not in allowed:
            assert controls['pmpose'][key] == (str(value) if isinstance(value, Path) else value), key
    return commands, controls


def fit(out):
    commands, controls = fit_commands(out)
    write(out/'fit_controls.json', {'equal_except_body_paths_output': True, 'controls': controls})
    for method, command in commands.items():
        target = out/method
        if target.exists():
            raise FileExistsError(target)
        target.mkdir()
        write(target/'fit_command.json', command)
        run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'fit-child',
             '--output', str(out), '--method', method], target/'fit.log')
        result = np.load(target/'fit/result.npz')
        assert result['vertices'].shape == (360, 6890, 3)
        assert result['betas'].shape == (1, 10)
        assert np.isfinite(result['vertices']).all()


def fit_child(out, method):
    import torch
    from check_mainline_regression import fit_arguments
    from pose_app.smplh_fitting.pipeline import run_fit
    options = fit_arguments(out/method/'fit_command.json')
    np.random.seed(0)
    torch.manual_seed(0)
    def observe(stage, step, arrays):
        np.savez_compressed(options.output_dir/f'checkpoint_{stage}_{step}.npz', **arrays)
    run_fit(options, observer=observe)


def audit(out):
    from check_mainline_regression import fit_arguments
    from pose_app.mainline_regression import observation_snapshot, audit_contract, replay_parameters
    from walker_tools.pose2d.compare_sapiens2_mainline import assert_frozen_prompts
    for side in ('left', 'right'):
        prompts = read(out/f'{side}_detections.json')['images']
        predicted = read(out/f'{side}_pmpose_raw.json')['images']
        assert len(predicted) == len(prompts) == 360
        for entry, row in zip(prompts, predicted):
            assert entry['file_name'] == row['file_name']
            assert_frozen_prompts(entry, row)
        native = [json.loads(s) for s in (out/f'{side}_sapiens_raw.jsonl').read_text().splitlines()]
        assert len(native) == len(prompts) == 360
        for i, row in enumerate(native):
            row['frame_index'] = i
            assert row['pair_id'] == prompts[i]['pair_id']
            assert row['file_name'] == prompts[i]['file_name']
            # Both production pose runners consume float32 boxes/scores.
            # Comparing JSON doubles before that cast invents prompt changes.
            assert_frozen_prompts(prompts[i], {
                'boxes_from_yolo26x': [x['bbox'] for x in row['instances']],
                'bbox_scores_from_yolo26x': [x['bbox_score'] for x in row['instances']]})
            for instance in row['instances']:
                instance['bbox_xyxy_from_yolo26x'] = instance['bbox']
                instance['bbox_score_from_yolo26x'] = instance['bbox_score']
        write(out/f'{side}_sapiens_stereo_adapter.json', {'images': native})
    for method in ('pmpose', 'sapiens2'):
        target = out/method
        options = fit_arguments(target/'fit_command.json')
        observed = observation_snapshot(options)
        np.savez_compressed(target/'body_input_diagnostics.npz', **observed)
        paths = {s: out/(f'{s}_pmpose_raw.json' if method == 'pmpose' else f'{s}_sapiens_stereo_adapter.json')
                 for s in ('left', 'right')}
        if not (target/'strict_stereo/offline_stereo_summary.json').exists():
            run([sys.executable, '-B', 'realtime_app/tools/evaluate_offline_stereo_predictions.py',
                '--model', method, '--left-json', str(paths['left']), '--right-json', str(paths['right']),
                '--calibration', str(out/'calibration/stereo_fisheye.json'),
                '--output-dir', str(target/'strict_stereo'), '--keypoint-threshold', '.25',
                '--max-association-cost', '.05', '--max-reprojection-error-px', '10',
                '--max-matches', '1', '--triangulation-mode', 'strict'], target/'strict_stereo_retry.log')
        if not (target/'fit/result.npz').exists():
            failure = read(out/f'fit-child_{method}_failure.json')
            write(target/'implementation_audit.json', {'status': 'fit_failed', 'failure': failure,
                  'initialization': read(target/'fit/initialization_audit.json'),
                  'parameter_replay': {'unavailable': {'passed': False, 'reason': 'no fitted result'}},
                  'evidence': 'failure_preserved; strict_geometry_diagnostics_only'})
            continue
        with np.load(target/'fit/result.npz') as source:
            result = {k: source[k] for k in source.files}
        with np.load(target/'fit/cold_initialization.npz') as source:
            initial = {k: source[k] for k in source.files}
        report = audit_contract(result, initial, read(target/'fit/fit_summary.json'), observed, options)
        assert all(row['passed'] for row in report.values()), report
        replay = replay_parameters(result, options, 32)
        assert all(row['passed'] for row in replay.values()), replay
        checkpoint = np.load(target/'fit/checkpoint_C_body_vposer_refine_300.npz')
        assert np.array_equal(checkpoint['betas'], result['betas'])
        assert np.array_equal(checkpoint['smplh_joints'][:, :22], result['smplh_joints'][:, :22])
        write(target/'implementation_audit.json', {'controls': report, 'parameter_replay': replay,
              'body_frozen_during_hand_stages': True, 'evidence': 'implementation_consistency_only'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'detect', 'pmpose', 'sapiens', 'wilor', 'fit', 'fit-child', 'audit'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=('pmpose', 'sapiens2'))
    args = ap.parse_args()
    out = args.output.resolve()
    record_stage = f'fit-child_{args.method}' if args.stage == 'fit-child' else args.stage
    try:
        if args.stage == 'fit-child':
            if args.method is None:
                raise ValueError('fit-child requires --method')
            fit_child(out, args.method)
        else:
            globals()[args.stage](out)
        write(out/f'{record_stage}_completed.json', {'stage': args.stage, 'method': args.method, 'status': 'completed'})
    except Exception as exc:
        if out.exists():
            write(out/f'{record_stage}_failure.json', {'stage': args.stage, 'method': args.method,
                  'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
