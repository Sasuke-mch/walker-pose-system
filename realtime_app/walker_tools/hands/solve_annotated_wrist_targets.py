"""Stereo wrist-point reference independent of finger and grip optimization."""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pose_app.fisheye_camera import load_stereo_fisheye,fisheye_project_numpy


def upright_to_raw(point,camera):
    u,v=np.asarray(point,float)
    return np.array([1919-v,u]) if camera=='left' else np.array([v,1079-u])


def wrist_points(path):
    label=json.loads(path.read_text(encoding='utf-8'))
    if (label['imageWidth'],label['imageHeight'])!=(1080,1920):
        raise ValueError('unexpected_annotation_size')
    result={}
    for side in ('left','right'):
        shapes=[s for s in label['shapes'] if s['label']==f'{side}_wrist']
        if len(shapes)!=1 or shapes[0]['shape_type']!='point' or len(shapes[0]['points'])!=1:
            result[side]={'state':'unavailable','reason':'one_explicit_wrist_point_required'}
            continue
        p=np.asarray(shapes[0]['points'][0],float)
        if p.shape!=(2,) or not np.isfinite(p).all() or not (0<=p[0]<1080 and 0<=p[1]<1920):
            result[side]={'state':'unavailable','reason':'invalid_or_out_of_bounds_point'}
            continue
        result[side]={'state':'present','upright_px':p.tolist()}
    return result


def solve_point(left,right,cal):
    """Triangulate then refine only XYZ using the real fisheye projections."""
    left=np.asarray(left,float);right=np.asarray(right,float)
    nl=cv2.fisheye.undistortPoints(left.reshape(1,1,2),cal.K0,cal.D0).reshape(2)
    nr=cv2.fisheye.undistortPoints(right.reshape(1,1,2),cal.K1,cal.D1).reshape(2)
    R=cal.R_cam0_to_cam1;t=cal.T_cam0_to_cam1_mm/1000
    P0=np.column_stack([np.eye(3),np.zeros(3)])
    P1=np.column_stack([R,t])
    homogeneous=cv2.triangulatePoints(P0,P1,nl[:,None],nr[:,None])[:,0]
    if abs(homogeneous[3])<1e-12:return {'state':'rejected','reason':'triangulation_at_infinity'}
    initial=homogeneous[:3]/homogeneous[3]
    def residual(x):
        return np.concatenate([fisheye_project_numpy(x[None]*1000,cal.K0,cal.D0)[0]-left,fisheye_project_numpy((R@x+t)[None]*1000,cal.K1,cal.D1)[0]-right])
    fit=least_squares(residual,initial,max_nfev=100)
    x=fit.x;xr=R@x+t
    error=np.linalg.norm(residual(x).reshape(2,2),axis=1)
    ray0=np.r_[nl,1];ray1=R.T@np.r_[nr,1]
    angle=float(np.degrees(np.arccos(np.clip(ray0@ray1/(np.linalg.norm(ray0)*np.linalg.norm(ray1)),-1,1))))
    reasons=[]
    if not fit.success or not np.isfinite(x).all():reasons.append('nonfinite_or_optimization_failure')
    if x[2]<=0 or xr[2]<=0:reasons.append('nonpositive_depth')
    if np.any(error>10):reasons.append('reprojection_over_10px')
    if angle<1:reasons.append('ray_angle_under_1deg')
    return {'state':'accepted_manual_reference_geometry' if not reasons else 'rejected','reasons':reasons,'wrist_left_camera_m':x.tolist(),'reprojection_error_px':error.tolist(),'ray_angle_deg':angle,'limits':'agreement of manually chosen wrist-center projections, not independent anatomical 3D truth'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--annotations',type=Path,required=True)
    ap.add_argument('--walker-model',type=Path,required=True)
    ap.add_argument('--calibration-dir',type=Path,default=Path('realtime_app/calibration/results'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('refuse_existing_output')
    labels={c:wrist_points(args.annotations/'labels'/f'{c}_pair_0000.json') for c in ('left','right')}
    cal=load_stereo_fisheye(args.calibration_dir)
    walker=json.loads(args.walker_model.read_text(encoding='utf-8'))
    R=np.asarray(walker['rotation_left_camera_from_walker']);t=np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1):raise ValueError('invalid_walker_rotation')
    result={'schema':'annotated_wrist_targets_v1','source_annotations':str(args.annotations.resolve()),'frame_index':0,'assumption':'wrist_position_fixed_relative_to_walker_entire_video','finger_parameters_used':False,'hands':{},'annotation_audit':labels}
    for side in ('left','right'):
        if any(labels[c][side]['state']!='present' for c in ('left','right')):
            result['hands'][side]={'state':'unavailable','reason':'missing_explicit_stereo_wrist_points'}
            continue
        raw={c:upright_to_raw(labels[c][side]['upright_px'],c) for c in ('left','right')}
        solved=solve_point(raw['left'],raw['right'],cal)
        solved['raw_pixels']={c:p.tolist() for c,p in raw.items()}
        if solved['state']=='accepted_manual_reference_geometry':
            solved['wrist_walker_m']=((np.asarray(solved['wrist_left_camera_m'])-t)@R).tolist()
        result['hands'][side]=solved
    result['status']='accepted_manual_reference_geometry' if all(s['state']=='accepted_manual_reference_geometry' for s in result['hands'].values()) else 'incomplete_or_rejected'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
