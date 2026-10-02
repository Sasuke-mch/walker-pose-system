"""Compare Sapiens2 against the frozen C3 PMPose body observation sequence.

Use all stored detector prompts, the original Sapiens runner, existing strict
geometry and the unchanged body scene entry. Never use pose outputs for ROI
selection, alter scores, overwrite results or promote consistency to accuracy.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
from collections import Counter
import csv
import importlib
import json
import os
import subprocess
from pathlib import Path
import sys

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
ROOT = APP.parent
sys.path.insert(0, str(APP))
from pose_app.calibration import StereoCalibration
from pose_app.geometry_input import count_out_of_raw_image_bounds
from pose_app.rotation import rotate_image_for_model, restore_model_result_to_raw
from pose_app.schema import InferenceResult, PersonPose
from pose_app.stereo_sources import StereoFramePair
from pose_app.sources import SourceFrame
from pose_app.stereo_output import StereoOutputWriter
from walker_tools.pose2d.build_continuous_foot_inclusive_roi import RULE
from walker_tools.stereo.evaluate_offline_stereo_predictions import SAPIENS_TO_COCO17
from pose_app.triangulation import triangulate_matches


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def raw_result(value):
    return InferenceResult(
        source_frame_id=value['source_frame_id'], source_timestamp_sec=value['source_timestamp_sec'],
        image_width=value['image_width'], image_height=value['image_height'], model_name=value['model_name'],
        model_ms=value['model_ms'], roundtrip_ms=value['roundtrip_ms'],
        persons=[PersonPose.from_dict(x) for x in value['persons']],
        dropped_before=value.get('dropped_before', 0), stage_times_ms=value.get('stage_times_ms', {}))


def sapiens_points(item):
    xy = np.asarray(item['keypoints308'], dtype=float)
    scores = np.asarray(item['keypoint_scores'], dtype=float)
    if xy.shape != (308,2) or scores.shape != (308,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
        raise ValueError('invalid Sapiens output')
    return np.column_stack((xy[list(SAPIENS_TO_COCO17)],scores[list(SAPIENS_TO_COCO17)]))


def assert_frozen_prompts(entry,predicted):
    expected = [x['bbox_xyxy'] for x in entry['detections']]
    scores = [x['score'] for x in entry['detections']]
    if (not np.array_equal(np.asarray(expected,dtype=np.float32),predicted['boxes_from_yolo26x'])
            or not np.array_equal(np.asarray(scores,dtype=np.float32),predicted['bbox_scores_from_yolo26x'])):
        raise ValueError('stored C3 and PMPose input prompts differ')
    return len(expected)


def infer(args):
    json.loads((args.output / 'preparation_audit.json').read_text(encoding='utf-8'))
    for side in ('left','right'):
        if not (args.output / f'{side}_detections.json').is_file():
            raise FileNotFoundError(f'{side} prepared prompts missing')
    repo = ROOT / 'third_party/sapiens2/repo'
    pose = repo / 'sapiens/pose'
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(pose / 'tools/vis'))
    runner = importlib.import_module('run_sapiens2_from_yolo26_bboxes')
    old_cwd = Path.cwd()
    try:
        os.chdir(pose)
        model = runner.prepare_model('configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py',
            str(ROOT / 'models/sapiens2/pose-0.4b/sapiens2_0.4b_pose.safetensors'), 'cuda:0')
        if not model.cfg.val_cfg.get('flip_test', False):
            raise ValueError('expected existing flip-test configuration')
        for side in ('left', 'right'):
            path = args.output / f'{side}_sapiens_raw.jsonl'
            if path.exists():
                raise FileExistsError(path)
            entries = json.loads((args.output / f'{side}_detections.json').read_text(encoding='utf-8'))['images']
            with path.open('w', encoding='utf-8') as handle:
                for index, entry in enumerate(entries):
                    image = cv2.imread(entry['image_path'])
                    if image is None:
                        raise FileNotFoundError(entry['image_path'])
                    boxes, scores = runner.extract_person_boxes(entry)
                    xy, confidence, elapsed = runner.infer_image(model, image, boxes, scores, 2, 'cuda:0', True)
                    if len(xy) != len(boxes):
                        raise ValueError('Sapiens output/prompt count mismatch')
                    value = dict(pair_id=entry['pair_id'], file_name=entry['file_name'], elapsed_ms=elapsed,
                        instances=[{'bbox':b.tolist(), 'bbox_score':float(s), 'keypoints308':np.asarray(p).tolist(),
                                    'keypoint_scores':np.asarray(c).tolist()} for b,s,p,c in zip(boxes,scores,xy,confidence)])
                    handle.write(json.dumps(value, allow_nan=False)+'\n')
                    handle.flush()
                    if index % 20 == 0:
                        print(f'{side}: {index+1}/{len(entries)} {elapsed:.1f} ms', flush=True)
    finally:
        os.chdir(old_cwd)


def prepare_frozen(args):
    """Use all immutable C3 prompts from the current body-fitting sequence."""
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    baseline_path = args.baseline / 'pmpose_strict_stereo/offline_stereo_results.jsonl'
    baseline = rows(baseline_path)
    scene_source_path = ROOT/'research_records/engineering_validation/G20261001_annotated_shared_grip_v1/active_constructed_grip_v2/balanced_body_full448_v1/scene/scene_sources.json'
    scene_source = json.loads(scene_source_path.read_text(encoding='utf-8'))
    protocol = {'baseline':str(args.baseline), 'baseline_jsonl':str(baseline_path),
        'input_dir':str(args.baseline/'input_448pairs'), 'pairs':len(baseline),
        'calibration':str(APP/'calibration/results/stereo_fisheye.json'),
        'ground_reference':scene_source['static_ground_reference'],
        'walker_model':scene_source['static_walker_model'],
        'raw_videos':[scene_source['raw_left_video'],scene_source['raw_right_video']],
        'scene_entry':str(ROOT/'research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/pipeline/scene/replay_current_run.py'),
        'rotations':{'left':'ccw90','right':'cw90'}, 'roi':RULE,
        'source_mode':'frozen_c3_body_mainline', 'transport':'original PNG, exactly as frozen sequence PMPose input',
        'person_filter':'none, matching existing evaluate_offline_stereo_predictions._records',
        'keypoint_threshold':.25, 'association_threshold':.05,'reprojection_threshold_px':10.,'max_matches':1,
        'sapiens_mapping':list(SAPIENS_TO_COCO17),'sapiens_flip_test':True,
        'execution':'native CUDA using existing Sapiens runner; runtime differs from archived Docker',
        'success':'complete engineering paired comparison; not an accuracy or physical claim',
        'stop':'prompt, frame, mapping or baseline geometric gate mismatch; do not expand to fitting',
        'boundary':'Sapiens is a comparison, not ground truth. No smoothing, interpolation, score rescaling or failure removal.'}
    dump(args.output/'protocol.json',protocol)
    checks = []
    for side, folder in [('left','left_ccw90'),('right','right_cw90')]:
        c3 = args.baseline/'c3_predictions'/side
        detections = json.loads((c3/'continuous_roi/detections.json').read_text(encoding='utf-8-sig'))
        prediction = json.loads((c3/'pmpose/raw_predictions.json').read_text(encoding='utf-8-sig'))['images']
        entries = detections['images']
        if len(entries) != len(baseline) or len(prediction) != len(baseline):
            raise ValueError('frozen image counts differ')
        output = []
        for record, entry, predicted in zip(baseline,entries,prediction):
            if not record['file_name'] == entry['file_name'] == predicted['file_name']:
                raise ValueError('frozen image ordering mismatch')
            # Both existing runners convert their prompts to float32 before inference.
            count = assert_frozen_prompts(entry,predicted)
            path = args.baseline/'input_448pairs'/folder/entry['file_name']
            if not path.is_file():
                raise FileNotFoundError(path)
            output.append(dict(entry, pair_id=record['pair_id'], image_path=str(path.resolve())))
            checks.append({'pair_id':record['pair_id'],'side':side,'input_boxes':count})
        dump(args.output/f'{side}_detections.json',{'images':output})
    dump(args.output/'preparation_audit.json',{'pairs':len(baseline),'prompts_exact':True,'checks':checks})
    print(f'frozen C3 preparation passed: {len(baseline)} pairs',flush=True)


def saved_pair(record, protocol):
    images = []
    for side, folder, inverse in [('left','left_ccw90','cw90'),('right','right_cw90','ccw90')]:
        path = Path(protocol['input_dir'])/folder/record['file_name']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = rotate_image_for_model(image,inverse)
        value = record[side]
        images.append(SourceFrame(value['source_frame_id'],value['source_timestamp_sec'],image))
    return StereoFramePair(record['pair_id'],*images,float(record['timestamp_skew_ms'])/1000.,
        timestamp_type=record['timestamp_type'])


def replay(args):
    from pose_app.lower_limb_pipeline import build_lower_limb_pipeline
    from pose_app.lower_limb_live_status import LowerLimbLiveStatusWriter
    protocol = json.loads((args.output / 'protocol.json').read_text(encoding='utf-8'))
    for key,path in protocol.get('frozen_static_input_copies',{}).items():
        original = Path(protocol[key])
        if original.read_bytes() != Path(path).read_bytes():
            raise ValueError(f'{key} changed since static-input freezing')
        # Keep original path resolution: stereo JSON references mono-calibration
        # files relative to the app. A byte copy is an audit, not a relocation.
    if protocol.get('source_mode') != 'frozen_c3_body_mainline':
        raise ValueError('only the audited frozen C3 body sequence is supported')
    baseline = rows(Path(protocol.get('baseline_jsonl',args.baseline / 'stereo_results.jsonl')))
    sapiens = {side:rows(args.output / f'{side}_sapiens_raw.jsonl') for side in ('left','right')}
    for side, items in sapiens.items():
        if [x['pair_id'] for x in items] != [x['pair_id'] for x in baseline]:
            raise ValueError(f'{side} pair ordering differs')
    calibration = StereoCalibration.load(protocol['calibration']).for_runtime_sizes((1920,1080),(1920,1080))
    summaries = {}
    per_joint = []
    for name in ('pmpose', 'sapiens2'):
        out = args.output / name
        if out.exists():
            raise FileExistsError(out)
        out.mkdir()
        writer = StereoOutputWriter(out, False, True, 30., str(args.baseline), calibration, max_matches=1)
        live = LowerLimbLiveStatusWriter(out/'lower_limb_live_status.jsonl')
        geometry_checks = 0
        counts, reasons = Counter(), Counter()
        try:
            for index, record in enumerate(baseline):
                pair = saved_pair(record,protocol)
                if pair is None or pair.pair_id != record['pair_id']:
                    raise ValueError('pair alignment failed')
                results = []
                for side, rotation in protocol['rotations'].items():
                    original = raw_result(record[side])
                    if name == 'pmpose':
                        result = original
                    else:
                        saved = sapiens[side][index]
                        persons = []
                        for item in saved['instances']:
                            points = sapiens_points(item)
                            persons.append(PersonPose(len(persons), item['bbox'], item['bbox_score'], float(points[:,2].mean()), points.tolist()))
                        model_result = InferenceResult(original.source_frame_id, original.source_timestamp_sec, 1080,1920,
                            'Sapiens2-0.4B', saved['elapsed_ms'], 0., persons, original.dropped_before,
                            {'pose_ms':saved['elapsed_ms']})
                        result = restore_model_result_to_raw(model_result, raw_width=1920, raw_height=1080, rotation=rotation)
                    results.append(result)
                people = triangulate_matches(results[0].persons, results[1].persons, calibration, .25,.05,10.,max_matches=1)
                if name == 'pmpose':
                    computed = [x.to_dict() for x in people]
                    if computed != record['persons_3d']:
                        # Numeric replay tolerates only floating roundoff, never a gate change.
                        old = record['persons_3d']
                        if len(old) != len(computed):
                            raise ValueError('baseline association changed')
                        for a,b in zip(old,computed):
                            for x,y in zip(a['keypoints_3d'],b['keypoints_3d']):
                                if x['valid'] != y['valid'] or x['reason'] != y['reason']:
                                    raise ValueError('baseline point gate changed')
                                if x['xyz'] is not None and not np.allclose(x['xyz'],y['xyz'],rtol=0.,atol=1e-6):
                                    raise ValueError('baseline xyz changed')
                    geometry_checks += 1
                bounds = {s:count_out_of_raw_image_bounds(r.persons, calibration.left_image_size) for s,r in zip(('left','right'),results)}
                payload = writer.build_payload(pair, *results, people, bounds, max_matches=1)
                payload['file_name'] = record.get('file_name',f'pair_{pair.pair_id:04d}.png')
                live.consume(payload)
                writer.json_file.write(json.dumps(payload,allow_nan=False)+'\n')
                for person in people:
                    for point in person.keypoints_3d:
                        counts[point['name']] += int(point['valid'])
                        if not point['valid']:
                            reasons[point['reason']] += 1
                if not people:
                    reasons['no_stereo_person'] += 1
                if index % 60 == 0:
                    print(f'replay {name}: {index+1}/{len(baseline)}',flush=True)
            writer.json_file.close()
            writer.json_file = None
            summary = {'pairs':len(baseline),'valid_by_joint':dict(counts),'rejection_counts':dict(reasons),
                       'baseline_geometry_reproduced_pairs':geometry_checks,
                       'live':live.close(completed=True)}
            paths = []
            for side in ('left','right'):
                if name == 'pmpose':
                    paths.append(args.baseline/'c3_predictions'/side/'pmpose/raw_predictions.json')
                else:
                    # Only an explicit schema adapter; retain full source separately.
                    converted = []
                    for index,item in enumerate(sapiens[side]):
                        instances = item['instances']
                        if len(instances) != 1:
                            raise ValueError('body mainline requires exactly one frozen C3 prompt')
                        instance = instances[0]
                        converted.append({'image_id':index,'file_name':item['file_name'],
                            'keypoints':[sapiens_points(instance).tolist()]})
                    path = args.output/f'{side}_sapiens_coco17_model_input.json'
                    dump(path, {'source_model':'Sapiens2-0.4B','source_raw_jsonl':str(args.output/f'{side}_sapiens_raw.jsonl'),
                        'mapping':list(SAPIENS_TO_COCO17),'coordinate_space':'upright model input pixels','images':converted})
                    paths.append(path)
            scene = out/'scene'
            command = [sys.executable,protocol['scene_entry'],'--left-video',protocol['raw_videos'][0],
                '--right-video',protocol['raw_videos'][1],'--input-pair-dir',protocol['input_dir'],
                '--left-pmpose',str(paths[0]),'--right-pmpose',str(paths[1]),
                '--calibration',protocol['calibration'],'--ground-reference',protocol['ground_reference'],
                '--walker-model',protocol['walker_model'],'--output',str(scene)]
            dump(out/'scene_command.json',command)
            with (out/'scene_console.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            scene_metadata = json.loads((scene/'scene_sources.json').read_text(encoding='utf-8'))
            summary['stage'] = scene_metadata['stage_summary']
            summary['ground'] = scene_metadata['ground_summary']
            summary['scene_input_model'] = name
            summary['lower_limb'] = build_lower_limb_pipeline(out/'stereo_results.jsonl', out/'lower_limb_pipeline')
            dump(out/'summary.json', summary)
            summaries[name] = summary
        finally:
            if writer.json_file is not None:
                writer.json_file.close()
        for joint,count in counts.items():
            per_joint.append({'model':name, 'joint':joint,'accepted':count,'total':len(baseline),'coverage':count/len(baseline)})
    dump(args.output/'comparison.json', summaries)
    with (args.output/'joint_comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['model','joint','accepted','total','coverage'])
        writer.writeheader(); writer.writerows(per_joint)


def statistics(values):
    data = np.asarray(values,dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return {'count':0,'median':None,'p95':None,'mean':None,'mad':None}
    median = float(np.median(data))
    return {'count':int(data.size),'median':median,'p95':float(np.percentile(data,95)),
            'mean':float(data.mean()),'mad':float(np.median(np.abs(data-median)))}


def analyze(args):
    from tools.compare_pose_model_bone_ratios import BONES
    names = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
             'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
             'left_knee','right_knee','left_ankle','right_ankle']
    sequences = {m:rows(args.output/m/'stereo_results.jsonl') for m in ('pmpose','sapiens2')}
    a,b = sequences.values()
    if [r['pair_id'] for r in a] != [r['pair_id'] for r in b]:
        raise ValueError('analysis pair mismatch')
    joint_rows = []
    details = []
    models = {}
    for m,sequence in sequences.items():
        all_error, accepted_error = [],[]
        model_summary = {'pairs':len(sequence),'matched_pairs':0,'bounds':{},'joint':{},'bones':{}}
        points_by_frame = []
        for record in sequence:
            people = record['persons_3d']
            model_summary['matched_pairs'] += bool(people)
            points = {p['name']:p for p in people[0]['keypoints_3d']} if people else {}
            points_by_frame.append(points)
            for p in points.values():
                error = p.get('reprojection_error_mean_px')
                if error is not None:
                    all_error.append(error)
                    if p['valid']:
                        accepted_error.append(error)
        for side in ('left','right'):
            model_summary['bounds'][side] = sum(r['geometry_rejected_out_of_raw_bounds_keypoints'][side] for r in sequence)
        for j,name in enumerate(names):
            accepted = 0; reason = Counter(); errors = []; scores = {'left':[],'right':[]}
            for record,points in zip(sequence,points_by_frame):
                p = points.get(name)
                valid = bool(p and p['valid'])
                accepted += valid
                why = None if valid else (p['reason'] if p else 'no_stereo_person')
                if why:
                    reason[why] += 1
                if p and p.get('reprojection_error_mean_px') is not None:
                    errors.append(p['reprojection_error_mean_px'])
                for side in scores:
                    people = record[side]['persons']
                    if len(people) == 1:
                        scores[side].append(people[0]['keypoints'][j][2])
                details.append({'model':m,'pair_id':record['pair_id'],'joint':name,'valid':valid,
                    'reason':why,'reprojection_mean_px':None if p is None else p.get('reprojection_error_mean_px')})
            model_summary['joint'][name] = {'accepted':accepted,'coverage':accepted/len(sequence),
                'reasons':dict(reason),'all_finite_reprojection_px':statistics(errors),
                'native_scores':{s:statistics(v) for s,v in scores.items()}}
            joint_rows.append({'model':m,'joint':name,'accepted':accepted,'total':len(sequence),'coverage':accepted/len(sequence)})
        for bone,(start,end) in BONES.items():
            lengths = []
            for points in points_by_frame:
                x,y = points.get(start),points.get(end)
                if x and y and x['valid'] and y['valid']:
                    lengths.append(np.linalg.norm(np.asarray(x['xyz'])-y['xyz']))
            model_summary['bones'][bone] = statistics(lengths)
        model_summary['all_finite_reprojection_px'] = statistics(all_error)
        model_summary['accepted_reprojection_px'] = statistics(accepted_error)
        model_summary['total_accepted'] = sum(x['accepted'] for x in model_summary['joint'].values())
        if m == 'sapiens2':
            model_summary['pose_correctness_elapsed_ms'] = statistics([r['elapsed_ms'] for side in ('left','right')
                for r in rows(args.output/f'{side}_sapiens_raw.jsonl')])
        models[m] = model_summary
    common = {}; differences2d = {}
    for j,name in enumerate(names):
        xyz_difference = []; common_error = {m:[] for m in sequences}; counts = Counter(); offset = {'left':[],'right':[]}
        for x,y in zip(a,b):
            pa = {p['name']:p for p in x['persons_3d'][0]['keypoints_3d']} if x['persons_3d'] else {}
            pb = {p['name']:p for p in y['persons_3d'][0]['keypoints_3d']} if y['persons_3d'] else {}
            va = bool(pa.get(name,{}).get('valid')); vb = bool(pb.get(name,{}).get('valid'))
            counts['both' if va and vb else ('pmpose_only' if va else ('sapiens_only' if vb else 'neither'))] += 1
            if va and vb:
                xyz_difference.append(np.linalg.norm(np.asarray(pa[name]['xyz'])-pb[name]['xyz']))
                for m,p in [('pmpose',pa[name]),('sapiens2',pb[name])]:
                    common_error[m].append(p['reprojection_error_mean_px'])
            for side in offset:
                if len(x[side]['persons']) == len(y[side]['persons']) == 1:
                    px,py = x[side]['persons'][0]['keypoints'][j],y[side]['persons'][0]['keypoints'][j]
                    offset[side].append(np.linalg.norm(np.asarray(px[:2])-py[:2]))
        common[name] = {'coverage_partition':dict(counts),'cross_model_3d_difference_mm':statistics(xyz_difference),
                        'same_frames_reprojection_px':{m:statistics(v) for m,v in common_error.items()}}
        differences2d[name] = {s:statistics(v) for s,v in offset.items()}
    common_bones = {}
    for bone,(start,end) in BONES.items():
        lengths = {m:[] for m in sequences}
        for x,y in zip(a,b):
            points = {m:({p['name']:p for p in r['persons_3d'][0]['keypoints_3d']} if r['persons_3d'] else {})
                      for m,r in [('pmpose',x),('sapiens2',y)]}
            if all(points[m].get(j,{}).get('valid') for m in points for j in (start,end)):
                for m in points:
                    lengths[m].append(np.linalg.norm(np.asarray(points[m][start]['xyz'])-points[m][end]['xyz']))
        common_bones[bone] = {m:statistics(v) for m,v in lengths.items()}
    dump(args.output/'metrics.json', {'models':models,'common_accepted':common,'common_frame_bones_mm':common_bones,
        'cross_model_2d_difference_px':differences2d,'boundary':'Cross-model differences are not errors to ground truth. Finite, accepted, common and all-frame denominators kept separately. Timing is native correctness inference, not paired live latency.'})
    with (args.output/'point_quality.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    # The body fit uses the original soft-weight raw triangulation, a distinct
    # contract from the strict live gate. Audit it through the same entry too.
    import importlib.util
    entry = ROOT/'research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py'
    spec = importlib.util.spec_from_file_location('raw_observation_comparison',entry)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.cv2 = cv2
    protocol = json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
    calibration = module.load_stereo_fisheye(Path(protocol['calibration']).parent)
    body = {}
    for model in sequences:
        paths = ([args.baseline/'c3_predictions'/s/'pmpose/raw_predictions.json' for s in ('left','right')]
                 if model == 'pmpose' else [args.output/f'{s}_sapiens_coco17_model_input.json' for s in ('left','right')])
        left = module.raw_side(paths[0],'left');right = module.raw_side(paths[1],'right')
        if sorted(left) != list(range(len(a))) or sorted(right) != list(range(len(a))):
            raise ValueError('body observation frame indices differ')
        result = module.raw_triangulate(np.stack([left[i] for i in range(len(a))]),np.stack([right[i] for i in range(len(a))]),calibration)
        raw,el,er,gap,dl,dr,accepted,reason,q,parts = result
        np.savez_compressed(args.output/model/'body_raw_observations.npz',raw=raw,left_reprojection_error=el,
            right_reprojection_error=er,ray_gap=gap,depth_left=dl,depth_right=dr,accepted=accepted,
            reject_reason=reason,confidence=q,confidence_2d=parts[0],confidence_ray=parts[1],confidence_reprojection=parts[2])
        body[model] = {'accepted_positive_depth_points':int(accepted.sum()),'accepted_per_joint':dict(zip(names,accepted.sum(0).astype(int).tolist())),
            'ray_gap_mm':statistics(gap[accepted]),'mean_reprojection_px':statistics(((el+er)/2)[accepted]),
            'weight':statistics(q[accepted]),'definition':'Existing body raw_triangulate accepts finite in-bounds positive-depth geometry and weights reprojection softly; distinct from strict live observations.'}
    dump(args.output/'body_observation_comparison.json',body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare-frozen','infer','replay','analyze'])
    parser.add_argument('--baseline', type=Path, default=ROOT/'research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.baseline = args.baseline.resolve()
    try:
        {'prepare-frozen':prepare_frozen,'infer':infer,'replay':replay,'analyze':analyze}[args.stage](args)
    except Exception as exc:
        if args.output.exists():
            dump(args.output/f'{args.stage}_failure.json', {'type':type(exc).__name__,'reason':str(exc)})
        raise


if __name__ == '__main__':
    main()
