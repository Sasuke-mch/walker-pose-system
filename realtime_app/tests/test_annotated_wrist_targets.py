import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

tools=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(tools))
spec=importlib.util.spec_from_file_location('wrists',tools/'solve_annotated_wrist_targets.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_upright_inverse_is_not_mirrored():
    assert np.array_equal(mod.upright_to_raw([200,1819],'left'),[100,200])
    assert np.array_equal(mod.upright_to_raw([879,100],'right'),[100,200])


def test_stereo_reference_roundtrip_with_real_calibration():
    from pose_app.fisheye_camera import load_stereo_fisheye,fisheye_project_numpy
    root=Path(__file__).resolve().parents[2]
    cal=load_stereo_fisheye(root/'realtime_app/calibration/results')
    x=np.array([.08,.12,.8])
    y=cal.R_cam0_to_cam1@x+cal.T_cam0_to_cam1_mm/1000
    l=fisheye_project_numpy(x[None]*1000,cal.K0,cal.D0)[0]
    r=fisheye_project_numpy(y[None]*1000,cal.K1,cal.D1)[0]
    result=mod.solve_point(l,r,cal)
    assert result['state']=='accepted_manual_reference_geometry'
    assert np.allclose(result['wrist_left_camera_m'],x,atol=1e-7)
    assert max(result['reprojection_error_px'])<1e-6


def test_contour_does_not_silently_become_wrist(tmp_path):
    path=tmp_path/'annotation.json'
    data={'imageWidth':1080,'imageHeight':1920,'shapes':[{'label':'left_hand','shape_type':'polygon','points':[[10,10],[20,20],[30,10]]}]}
    path.write_text(json.dumps(data))
    result=mod.wrist_points(path)
    assert result['left']['state']=='unavailable'
    assert result['right']['state']=='unavailable'


def test_ambiguous_or_outside_wrist_rejected(tmp_path):
    path=tmp_path/'annotation.json'
    data={'imageWidth':1080,'imageHeight':1920,'shapes':[
        {'label':'left_wrist','shape_type':'point','points':[[10,10]]},
        {'label':'left_wrist','shape_type':'point','points':[[12,12]]},
        {'label':'right_wrist','shape_type':'point','points':[[1080,10]]}]}
    path.write_text(json.dumps(data))
    result=mod.wrist_points(path)
    assert result['left']['state']=='unavailable'
    assert result['right']['reason']=='invalid_or_out_of_bounds_point'
