"""Single bounded replay of accepted-iterate covariance conditioning."""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_iterated_motion_tracker import (
    IteratedMAPConfig,
    SMPLHIteratedMotionTracker,
)
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, SMPLHObservationMap
from run_stereo_pva_baseline import _load_predictions, _read_capture_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hip-adaptation', action='store_true')
    parser.add_argument('--frames', type=int, choices=(62,448), default=62)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.analyze_only:
        analyze()
        return
    if args.frames == 448 and not args.hip_adaptation:
        raise ValueError('The covariance-only candidate failed the prefix gate; no full replay')
    root = APP.parent / 'research_records/engineering_validation/G20261014_smplh_motion448_v1'
    source = root / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    name = 'hip_operator_feedback' if args.hip_adaptation else 'covariance_feedback'
    output = root / 'hip_pose_repair_20261016' / (name if args.frames == 62 else name+'_full448')
    output.mkdir(parents=True, exist_ok=False)
    p = json.loads((source / 'run_parameters.json').read_text())
    inp = p['inputs']
    cfg = SMPLHMotionConfig(**p['configuration'])
    opt = replace(IteratedMAPConfig(**p['iterated_map_configuration']), condition_accepted_iterate=True)
    protocol = {'source_run': str(source), 'frames': [0, args.frames-1],
        'command': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        'code': ['realtime_app/pose_app/smplh_iterated_motion_tracker.py', 'realtime_app/tools/probe_smplh_covariance_feedback.py'],
        'configuration': asdict(cfg), 'iterated_map_configuration':asdict(opt),
        'single_change': 'Condition full q/v/a covariance ONCE on each accepted final linearization, including nonconverged improving iterates. Keep derivative publication blocked for nonconverged iterates.',
        'unchanged': 'frame0 pose, beta, pixels, dt, VPoser sigma0.75, all gates,15-step optimizer and conditional state mean',
        'hypothesis': 'Withholding every pixel information update lets Pqq grow and weakens the temporal prior, magnifying pose jumps.',
        'success': '35..61 mean ankle rise <30mm AND same-window hip pixel median improves >=20%, all62 poses available; no new restart. Otherwise do not promote or run448.',
        'limits': 'Linearized covariance is an approximation, not proof of exact MAP convergence or calibrated uncertainty. No ground/contact/3D target.',
        'stop': 'One62-frame pass, no repeats/weight search.'}
    if args.hip_adaptation:
        protocol.update(single_change='Relative to covariance_feedback: replace only two COCO hip observation rows by one symmetric, frozen2-parameter virtual-joint map after frame10.',
            calibration='Frames1..10 only: fit same raw stereo pixels to frozen pre-adaptation poses; alpha in[0,.5] widens, thigh_fraction in[0,.4] moves observation hip toward knee. Freeze thereafter. No shape/native kinematic joint change.',
            validation='Frames11..61, primary35..61. Do not use calibration residual for success.',
            research='https://openaccess.thecvf.com/content/CVPR2023W/RHOBIN/papers/Zioulis_KBody_Towards_General_Robust_and_Aligned_Monocular_Whole-Body_Estimation_CVPRW_2023_paper.pdf',
            limits=protocol['limits']+' Virtual COCO points are detector landmarks, NEVER anatomical joint centers. Evaluate native pelvis separately; this diagnostic cannot certify anatomy.')
    frozen_calibration = None
    if args.frames == 448:
        prefix_protocol = root/'hip_pose_repair_20261016/hip_operator_feedback/protocol.json'
        frozen_calibration = json.loads(prefix_protocol.read_text())['hip_calibration_result']
        protocol.update(calibration='Read two frozen weights from the completed prefix calibration, apply after frame10; no refit or later-frame calibration.',
            calibration_source=str(prefix_protocol),stop='One complete448 pass of the successful prefix candidate, no sweep/retry.',
            generalization_gate='Full raw-slot median <= original*1.05, P95 <= original; hip median decreases>=20%; no new missing/restarted pose. Inspect every long static segment, retain all failures. These are engineering gates, not anatomical accuracy.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    model = SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], device='cuda')
    model.torch.set_num_threads(2)
    tr = SMPLHIteratedMotionTracker(model, load_stereo_fisheye(inp['calibration_dir']), cfg, opt)
    left = _load_predictions(Path(inp['left_predictions']), 'left')
    right = _load_predictions(Path(inp['right_predictions']), 'right')
    first = np.array(json.loads(Path(p['first_frame_observations']).read_text())['keypoints'])
    left[0], right[0] = first
    times = _read_capture_times(Path(inp['input_pairs']), len(left))[0]
    points, errors, sigmas, avail, segments, rows = [], [], [], [], [], []
    joints, translations, roots, latents, bodies, calibration = [], [], [], [], [], []
    started = perf_counter()
    with (output / 'trajectory.jsonl').open('w') as stream:
        for i in range(args.frames):
            px = np.array([left[i, 5:17], right[i, 5:17]])
            result = tr.step(float(times[i]), *px)
            points.append(result['coco_m'])
            joints.append(result['joints_m'])
            translations.append(result['translation_m'])
            roots.append(result['root_rotation'])
            latents.append(result['vposer_latent'])
            bodies.append(result['body_rotations'])
            error = np.linalg.norm(result['projected_pixels']-px[..., :2], axis=-1) if result['projected_pixels'] is not None else np.full((2,12), np.nan)
            errors.append(error)
            sigmas.append(np.sqrt(np.diag(tr.covariance)[:38]))
            avail.append(result['pose_available'])
            segments.append(result['segment_id'])
            row = {'frame': i, 'status': result['status'], 'available': bool(result['pose_available']),
                'optimizer': result['iterated_update'], 'used': result['used_views'].tolist(), 'reason': result['view_reason'].tolist(),
                'error': error.tolist(), 'derivative_count': int(result['derivative_valid'].sum())}
            stream.write(json.dumps(row)+'\n')
            rows.append(row)
            if args.hip_adaptation and 1 <= i <= 10:
                calibration.append((result['coco_m'].copy(), px.copy(), result['used_views'].copy()))
                if i == 10:
                    from pose_app.fisheye_camera import fisheye_project_numpy
                    def residual(weights):
                        alpha, beta = weights
                        residuals = []
                        for co, px_cal, mask in calibration:
                            hips = co[11:13] + alpha*(co[11:13]-co[[12,11]]) + beta*(co[13:15]-co[11:13])
                            other = hips @ tr.cal.R_cam0_to_cam1.T + tr.cal.T_cam0_to_cam1_mm/1000
                            projected = np.stack((fisheye_project_numpy(hips,tr.cal.K0,tr.cal.D0),fisheye_project_numpy(other,tr.cal.K1,tr.cal.D1)))
                            residuals.extend(((projected-px_cal[:,6:8,:2])[mask[:,6:8]]/8).ravel())
                        return residuals
                    if frozen_calibration is None:
                        fit = least_squares(residual, [.15,.15], bounds=([0.,0.],[.5,.4]), max_nfev=30)
                        if not fit.success or np.any(np.isclose(fit.x,[0,0],atol=1e-5)) or np.any(np.isclose(fit.x,[.5,.4],atol=1e-5)):
                            raise RuntimeError('Virtual hip calibration failed or hit bounds; do not replay held-out frames')
                        weights = fit.x
                        result_calibration = {'alpha':weights[0],'thigh_fraction':weights[1],'nfev':fit.nfev,'cost':fit.cost,'optimality':fit.optimality,
                            'all_later_frames_excluded_from_calibration':True}
                    else:
                        weights = [frozen_calibration['alpha'],frozen_calibration['thigh_fraction']]
                        result_calibration = dict(frozen_calibration)
                    original = model.regressor.clone()
                    alpha, beta = weights
                    model.regressor[11:13] = original[11:13]+alpha*(original[11:13]-original[[12,11]])+beta*(original[13:15]-original[11:13])
                    tr.last_point_jacobian = None
                    protocol['hip_calibration_result'] = dict(result_calibration,
                        regressor_row_sum=model.regressor[11:13].sum(dim=1).cpu().numpy().tolist())
                    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
                    print('frozen virtual hip weights',weights,flush=True)
            if i%10 == 0:
                print('frame', i, 'hip', np.median(error[:,6:8]), 'Pstd t/z', sigmas[-1][0], np.median(sigmas[-1][6:]), flush=True)
    np.savez_compressed(output/'result.npz', predicted_coco=points, reprojection_px=errors,
        state_sigma=sigmas, pose_available=avail, segment_id=segments, timestamp_s=times[:args.frames],
        smplh_joints=joints,transl=translations,root_rotation=roots,vposer_latent=latents,body_rotations=bodies,
        betas=model.betas.cpu().numpy(),regressor=model.regressor.cpu().numpy())
    old = np.load(source/'result.npz')['predicted_coco'][:args.frames]
    scene = np.load(root/'slope_motion_20261015/scene/scene_transforms.npz')
    R, t = scene['rotation_ground_from_left'][:args.frames], scene['translation_ground_from_left_mm'][:args.frames]
    old_rows = [json.loads(v) for v in (source/'trajectory.jsonl').read_text().splitlines()[:args.frames]]
    summary = {'runtime_s': perf_counter()-started, 'available': sum(avail), 'segments': len(set(segments)), 'variants': {}}
    for name, co, err in [('original', old, np.array([v['reprojection_px'] for v in old_rows])), ('conditioned', np.array(points), np.array(errors))]:
        world = np.einsum('nij,nkj->nki', R, co)*1000+t[:,None]
        z = world[:,15:17,2]
        summary['variants'][name] = {'ankle35_61_change_mm': (z[61]-z[35]).tolist(),
            'mean_ankle35_61_change_mm': float((z[61]-z[35]).mean()),
            'median5_ankle35_61_change_mm': float((np.median(z[57:62],axis=0)-np.median(z[35:40],axis=0)).mean()),
            'hip35_61_median_px': float(np.median(err[35:62,:,6:8])),
            'all_frames_median_px': float(np.median(err)), 'all_frames_p95_px': float(np.percentile(err,95))}
    x, y = summary['variants']['original'], summary['variants']['conditioned']
    summary['primary_passed'] = bool(sum(avail)==args.frames and len(set(segments))==1 and y['mean_ankle35_61_change_mm']<30 and y['hip35_61_median_px']<=.8*x['hip35_61_median_px'])
    if args.frames == 448:
        for name, err in [('original',np.array([v['reprojection_px'] for v in old_rows])),('candidate',np.array(errors))]:
            summary[name+'_all448'] = {'median_px':float(np.nanmedian(err)),'p95_px':float(np.nanpercentile(err,95)),
                'hip_median_px':float(np.nanmedian(err[:,:,6:8])), 'finite_slots':int(np.isfinite(err).sum())}
        x,y = summary['original_all448'],summary['candidate_all448']
        summary['generalization_passed'] = bool(summary['primary_passed'] and y['median_px']<=x['median_px']*1.05 and y['p95_px']<=x['p95_px'] and y['hip_median_px']<=x['hip_median_px']*.8)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary,indent=2), flush=True)


def analyze():
    """Common raw-slot denominator and native-pelvis checks on saved runs."""
    from collections import Counter
    root = APP.parent/'research_records/engineering_validation/G20261014_smplh_motion448_v1'
    base = root/'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    out = root/'hip_pose_repair_20261016'
    p = json.loads((base/'run_parameters.json').read_text())
    inp = p['inputs']
    sides = [_load_predictions(Path(inp[k]),side) for k,side in [('left_predictions','left'),('right_predictions','right')]]
    first = np.array(json.loads(Path(p['first_frame_observations']).read_text())['keypoints'])
    sides[0][0],sides[1][0] = first
    pixels = np.stack([s[:,5:17] for s in sides],axis=1)
    eligible = (np.isfinite(pixels).all(axis=-1) & (pixels[...,2]>=.2) & (pixels[...,2]<=1)
        & (pixels[...,0]>=0) & (pixels[...,0]<1920) & (pixels[...,1]>=0) & (pixels[...,1]<1080))
    scene = root/'slope_motion_20261015/scene'
    transform = np.load(scene/'scene_transforms.npz')
    R,t = transform['rotation_ground_from_left'],transform['translation_ground_from_left_mm']
    stages = [json.loads(line)['stage']['operational'] for line in (scene/'stage1_stage2.jsonl').read_text().splitlines()]
    segments = []
    for i,s in enumerate(stages):
        if s == 'stage1_walker_static_human_moving':
            if i==0 or stages[i-1]!=s:
                start = i
            if i==len(stages)-1 or stages[i+1]!=s:
                segments.append((start,i))
    free = np.load(root/'skeleton_tracking_20261016/pva_matched_pixel_sigma8.npz')['camera_points_mm']
    report = {'source_run':str(base),'height_reference':str(scene),
        'limits':'Same-source reference-conditioned engineering comparison; PVA COCO hips are not anatomical pelvis truth; no independent position/contact/velocity truth.',
        'raw_slot_rule':'All finite,in-bounds score>=.2<=1 slots before epipolar gate, common to all variants. Missing estimates stay missing and count against coverage.',
        'runs':{}}
    names = {'original':base,'covariance_only':out/'covariance_feedback',
        'hip_prefix':out/'hip_operator_feedback','hip_full448':out/'hip_operator_feedback_full448'}
    baseline_joints = np.load(base/'result.npz')['smplh_joints']
    baseline_rows = [json.loads(line) for line in (base/'trajectory.jsonl').read_text().splitlines()]
    for name,path in names.items():
        if not (path/'result.npz').exists():
            continue
        a = np.load(path/'result.npz')
        count = len(a['predicted_coco'])
        rows = baseline_rows if name=='original' else [json.loads(line) for line in (path/'trajectory.jsonl').read_text().splitlines()]
        err = np.array([r['reprojection_px'] for r in rows]) if name=='original' else a['reprojection_px'].copy()
        err = err[:count]
        available = a['pose_available']
        err[~eligible[:count] | ~available[:,None,None]] = np.nan
        world = np.einsum('nij,nkj->nki',R[:count],a['predicted_coco'])*1000+t[:count,None]
        ankle = world[:,15:17,2]
        def metric(values, denominator):
            good = values[np.isfinite(values)]
            return {'finite_slots':len(good),'eligible_slots':int(denominator),
                'median_px':float(np.median(good)) if len(good) else None,
                'p95_px':float(np.percentile(good,95)) if len(good) else None}
        result = {'frames':count,'pose_available_frames':int(available.sum()),'segments':len(np.unique(a['segment_id'])),
            'all':metric(err,int(eligible[:count].sum())),
            'heldout11_end':metric(err[11:],int(eligible[11:count].sum())),
            'hip':metric(err[:,:,6:8],int(eligible[:count,:,6:8].sum())),
            'hip35_61':metric(err[35:62,:,6:8],int(eligible[35:62,:,6:8].sum())),
            'nonhip':metric(np.concatenate((err[:,:,:6],err[:,:,8:]),axis=-1),int(eligible[:count].sum()-eligible[:count,:,6:8].sum())),
            'static_segments':[]}
        for start,end in segments:
            if end>=count:
                continue
            z = ankle[start:end+1]
            delta = z[-1]-z[0]
            width = min(5,len(z))
            robust = np.median(z[-width:],axis=0)-np.median(z[:width],axis=0)
            result['static_segments'].append({'frames':[start,end],'ankle_change_mm':delta.tolist(),'mean_change_mm':float(delta.mean()),
                'median5_mean_change_mm':float(robust.mean()),'poses_available':int(available[start:end+1].sum()),
                'both_ankles_persistent_rise20mm':bool(end-start+1>=10 and np.all(delta>=20) and np.all(robust>=20))})
        if 'smplh_joints' in a:
            joints = a['smplh_joints']
            native = np.einsum('nij,nj->ni',R[:count],joints[:,0])*1000+t[:count]
            baseline_native = np.einsum('nij,nj->ni',R[:count],baseline_joints[:count,0])*1000+t[:count]
            result['native_pelvis_forward_shift35_61_median_mm'] = float(np.median((native-baseline_native)[35:62,1]))
            result['native_pelvis_shift61_mm'] = (native-baseline_native)[61].tolist()
            pairs = [(1,4),(4,7),(2,5),(5,8)]
            result['native_leg_length_range_mm'] = [float(np.ptp(np.linalg.norm(joints[:,i]-joints[:,j],axis=-1))*1000) for i,j in pairs]
            hips = world[:,11:13].mean(axis=1)
            free_hip = np.einsum('nij,nj->ni',R[:count],free[:count,6:8].mean(axis=1))+t[:count]
            result['observation_hip_back_offset35_61_median_mm'] = float(np.median((hips-free_hip)[35:62,1]))
        opt = [r.get('iterated_update') if name=='original' else r.get('optimizer') for r in rows[:count]]
        result['optimizer_statuses'] = dict(Counter(v['status'] for v in opt if v))
        result['posterior_information_updates'] = sum(v.get('posterior_updates',0) for v in opt if v)
        if name != 'original':
            result['published_joint_derivative_slots'] = sum(r['derivative_count'] for r in rows[:count])
        report['runs'][name] = result
    if 'hip_full448' in report['runs']:
        b,c = report['runs']['original'],report['runs']['hip_full448']
        report['full_gate'] = {'all_poses_available':c['pose_available_frames']==448,'no_new_restart':c['segments']==b['segments'],
            'all_median_within5pct':c['all']['median_px']<=b['all']['median_px']*1.05,
            'all_p95_not_worse':c['all']['p95_px']<=b['all']['p95_px'],
            'hip_median_improves20pct':c['hip']['median_px']<=b['hip']['median_px']*.8}
        report['full_gate_passed'] = all(report['full_gate'].values())
        # The candidate lowers ankles: check whether this trades apparent lift
        # for model-sole penetration against the same operational reference.
        from pose_app.smplh_ground_constraints import load_sole_vertex_indices
        sole_path = APP.parent/'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json'
        sole = load_sole_vertex_indices(sole_path)
        model = SMPLHObservationMap(Path(p['effective_initial_fit']),
            *[Path(inp[k]) for k in ('smplh_model','regressor','vposer_dir','mano_left','mano_right')],device='cuda')
        torch = model.torch
        torch.set_num_threads(2)
        original = np.load(base/'result.npz')
        candidate = np.load(names['hip_full448']/'result.npz')
        prefix = np.load(names['hip_prefix']/'result.npz')
        report['frozen_contracts'] = {'shape_beta_exactly_equal_to_original':bool(np.array_equal(candidate['betas'],original['betas'])),
            'full448_first62_point_max_difference_m':float(np.max(abs(candidate['predicted_coco'][:62]-prefix['predicted_coco']))),
            'hip_parameters_frozen_after_frame10':True}
        heights = {}
        for name,a in [('original',original),('candidate',candidate)]:
            values = []
            for start in range(0,448,32):
                stop = min(start+32,448)
                size = stop-start
                if name=='original':
                    vertices = a['vertices'][start:stop]
                else:
                    def tensor(x):
                        return torch.as_tensor(x,dtype=torch.float32,device='cuda')
                    with torch.inference_mode():
                        body = model.model(betas=model.betas.expand(size,-1),transl=tensor(a['transl'][start:stop]),
                            global_orient=tensor(a['root_rotation'][start:stop]),body_pose=tensor(a['body_rotations'][start:stop]),
                            left_hand_pose=model.hand_rotations[0].expand(size,-1,-1,-1),right_hand_pose=model.hand_rotations[1].expand(size,-1,-1,-1))
                        vertices = body.vertices.cpu().numpy()
                world = np.einsum('nij,nkj->nki',R[start:stop],vertices)*1000+t[start:stop,None]
                values.extend(np.stack([world[:,side,2].min(axis=1) for side in sole],axis=-1))
            values = np.array(values)
            heights[name] = values
            report[name+'_sole_reference'] = {'minimum_mm':float(values.min()),'median_mm':float(np.median(values)),
                'below_minus5mm_slots':int((values< -5).sum()),'total_slots':int(values.size),
                'primary35_61_minimum_mm':float(values[35:62].min()),
                'primary35_61_endpoint_change_mm':(values[61]-values[35]).tolist(),
                'limits':'54 model sole candidates per foot against reused operational Stage/ground reference; not actual contact/penetration truth.'}
        np.savez_compressed(out/'sole_reference_diagnostic.npz',**heights)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='runs'},indent=2))


if __name__ == '__main__':
    main()
