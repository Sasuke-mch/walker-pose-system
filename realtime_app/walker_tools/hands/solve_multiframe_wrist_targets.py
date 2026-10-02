"""Audit sparse wrist labels; share a walker-local point without bypassing gates."""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from walker_tools.hands.solve_annotated_wrist_targets import fisheye_project_numpy, load_stereo_fisheye, solve_point, upright_to_raw, wrist_points


def project(x, camera, cal):
    y = x if camera == 'left' else cal.R_cam0_to_cam1 @ x + cal.T_cam0_to_cam1_mm / 1000
    K, D = (cal.K0, cal.D0) if camera == 'left' else (cal.K1, cal.D1)
    return fisheye_project_numpy(y[None] * 1000, K, D)[0]


def raw_to_upright(p, camera):
    u, v = p
    return np.array([v, 1919-u]) if camera == 'left' else np.array([1079-v, u])


def fit_shared(observations, initials, cal):
    """All present points contribute; robust loss does not remove observations."""
    def residual(x):
        return np.concatenate([project(x, o['camera'], cal)-o['raw_px'] for o in observations])
    fits = [least_squares(residual, x, loss='soft_l1', f_scale=5, max_nfev=300) for x in initials]
    valid = [f for f in fits if f.success and np.isfinite(f.x).all() and np.isfinite(f.cost)]
    if not valid:
        return None
    fit = min(valid, key=lambda f: f.cost)
    return fit.x, np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1), float(fit.cost)


def gate_shared(pair_states, errors, x, cal):
    reasons = []
    if sum(s == 'accepted_manual_reference_geometry' for s in pair_states) < 3:
        reasons.append('fewer_than_3_accepted_stereo_pairs')
    if np.any(np.asarray(errors) > 10):
        reasons.append('shared_reprojection_over_10px')
    if not np.isfinite(x).all() or x[2] <= 0 or (cal.R_cam0_to_cam1 @ x + cal.T_cam0_to_cam1_mm/1000)[2] <= 0:
        reasons.append('nonfinite_or_nonpositive_depth')
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--walker-model', type=Path, required=True)
    ap.add_argument('--calibration-dir', type=Path, default=Path('realtime_app/calibration/results'))
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError('refuse_existing_output')
    manifest = json.loads((args.annotations/'manifest.json').read_text(encoding='utf8'))
    cal = load_stereo_fisheye(args.calibration_dir)
    if (cal.image_width, cal.image_height) != (1920, 1080):
        raise ValueError('unexpected_calibration_size')
    walker = json.loads(args.walker_model.read_text(encoding='utf8'))
    R = np.asarray(walker['rotation_left_camera_from_walker'])
    t = np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError('invalid_walker_rotation')
    frames = manifest['frames']
    labels, entries = {}, {}
    for e in manifest['images']:
        key = (e['frame_index'], e['camera'])
        if key in entries:
            raise ValueError('duplicate_frame_camera')
        entries[key] = e
        image = cv2.imread(e['image'])
        source = cv2.imread(e['source'])
        if image is None or source is None or image.shape[:2] != (1920, 1080) or not np.array_equal(image, source):
            raise ValueError('image_source_or_size_mismatch')
        path = Path(e['annotation'])
        labels[key] = wrist_points(path) if path.exists() else {s: {'state':'unavailable','reason':'missing_annotation_file'} for s in ('left','right')}
    if set(entries) != {(f,c) for f in frames for c in ('left','right')}:
        raise ValueError('incomplete_manifest')
    result = {'schema':'annotated_wrist_targets_v1', 'source_annotations':str(args.annotations.resolve()),
              'walker_model':str(args.walker_model.resolve()), 'calibration_dir':str(args.calibration_dir.resolve()),
              'frames':frames, 'finger_parameters_used':False, 'hands':{},
              'assumption':'wrist_fixed_relative_to_walker_with_rigid_camera_mount',
              'limits':'manual operational reference consistency; not anatomical joint truth',
              'protocol':{'loss':'soft_l1','scale_px':5,'min_accepted_pairs':3,'max_reprojection_px':10}}
    for side in ('left','right'):
        pairs, observations, initials = [], [], []
        for frame in frames:
            points = {c:labels[frame,c][side] for c in ('left','right')}
            for c, point in points.items():
                if point['state']=='present':
                    observations.append({'frame_index':frame,'camera':c,'upright_px':point['upright_px'],
                                         'raw_px':upright_to_raw(point['upright_px'],c).tolist()})
            row = {'frame_index':frame,'annotation_audit':points}
            if all(p['state']=='present' for p in points.values()):
                row.update(solve_point(*[upright_to_raw(points[c]['upright_px'],c) for c in ('left','right')],cal))
                if 'wrist_left_camera_m' in row:
                    x=np.asarray(row['wrist_left_camera_m'])
                    row['diagnostic_wrist_walker_m']=((x-t)@R).tolist()
                    if np.isfinite(x).all(): initials.append(x)
            else:
                row.update(state='unavailable',reasons=['missing_explicit_stereo_wrist_points'])
            pairs.append(row)
        hand={'pairs':pairs,'observations':observations,'state':'unavailable'}
        fit=fit_shared(observations,initials,cal) if observations and initials else None
        if fit is not None:
            x, errors, cost=fit
            reasons=gate_shared([p['state'] for p in pairs],errors,x,cal)
            hand.update(state='rejected' if reasons else 'accepted_manual_reference_geometry',reasons=reasons,
                        diagnostic_wrist_walker_m=((x-t)@R).tolist(),diagnostic_wrist_left_camera_m=x.tolist(),
                        shared_reprojection_px=errors.tolist(),median_px=float(np.median(errors)),p95_px=float(np.percentile(errors,95)),cost=cost)
            if not reasons: hand['wrist_walker_m']=hand['diagnostic_wrist_walker_m']
            xyz=np.array([p['diagnostic_wrist_walker_m'] for p in pairs if 'diagnostic_wrist_walker_m' in p])
            hand['diagnostic_pair_dispersion_mm']=dict(rms=float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(0))**2,axis=1)))*1000),axis_range=(np.ptp(xyz,axis=0)*1000).tolist())
            loo=[]
            for frame in frames:
                train=[o for o in observations if o['frame_index']!=frame]
                held=[o for o in observations if o['frame_index']==frame]
                if not held: loo.append({'frame_index':frame,'state':'unavailable'}); continue
                f=fit_shared(train,[x],cal)
                loo.append({'frame_index':frame,'state':'diagnostic_only','heldout_errors_px':[float(np.linalg.norm(project(f[0],o['camera'],cal)-o['raw_px'])) for o in held]} if f else {'frame_index':frame,'state':'failed'})
            hand['leave_one_frame_out']=loo
            cv_errors=[]
            for o, err in zip(observations, errors):
                o['shared_error_px']=float(err)
                p=project(x,o['camera'],cal)
                o['shared_upright_px']=raw_to_upright(p,o['camera']).tolist()
                y=x if o['camera']=='left' else cal.R_cam0_to_cam1@x+cal.T_cam0_to_cam1_mm/1000
                K,D=(cal.K0,cal.D0) if o['camera']=='left' else (cal.K1,cal.D1)
                cvp=cv2.fisheye.projectPoints(y.reshape(1,1,3),np.zeros(3),np.zeros(3),K,D)[0].reshape(2)
                cv_errors.append(float(np.linalg.norm(cvp-p)))
            hand['opencv_projection_max_difference_px']=max(cv_errors)
        result['hands'][side]=hand
    result['status']='accepted_manual_reference_geometry' if all(h['state']=='accepted_manual_reference_geometry' for h in result['hands'].values()) else 'incomplete_or_rejected'
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'wrist_targets.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    # Two wrist crops per original view: present points only; no invented markers.
    tiles=[]
    for frame in frames:
        cells=[]
        for c in ('left','right'):
            image=cv2.imread(entries[frame,c]['image'])
            for side in ('left','right'):
                point=labels[frame,c][side]
                canvas=np.zeros((250,350,3),np.uint8)
                if point['state']=='present':
                    u,v=map(int,point['upright_px']); x0=max(0,min(u-175,730));y0=max(0,min(v-110,1700))
                    canvas[:220]=image[y0:y0+220,x0:x0+350]
                    cv2.circle(canvas,(u-x0,v-y0),6,(0,255,0),2)
                    obs=next((o for o in result['hands'][side]['observations'] if o['frame_index']==frame and o['camera']==c),None)
                    if obs and 'shared_upright_px' in obs:
                        p=np.rint(np.array(obs['shared_upright_px'])-[x0,y0]).astype(int)
                        cv2.drawMarker(canvas,tuple(p),(0,0,255),cv2.MARKER_CROSS,14,2)
                cv2.putText(canvas,f'{frame} {c} {side} '+('DIAGNOSTIC' if point['state']=='present' else 'MISSING'),(5,240),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cells.append(canvas)
        tiles.append(np.concatenate(cells,axis=1))
    cv2.imwrite(str(args.output_dir/'wrist_diagnostic_crops.png'),np.concatenate(tiles,axis=0))
    print(json.dumps({s:{k:h.get(k) for k in ('state','reasons','median_px','p95_px','diagnostic_pair_dispersion_mm')} for s,h in result['hands'].items()},ensure_ascii=False))


if __name__=='__main__':
    main()
