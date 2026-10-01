"""Shared wrist solver must preserve inconsistent labels as failures."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from solve_multiframe_wrist_targets import (
    fit_shared, gate_shared, load_stereo_fisheye, project,
    raw_to_upright, upright_to_raw,
)


def calibration():
    return load_stereo_fisheye(Path(__file__).resolve().parents[2]/'realtime_app/calibration/results')


def test_shared_known_point_and_sparse_monocular_observation():
    cal=calibration()
    x=np.array([.08,.12,.8])
    observations=[{'camera':c,'raw_px':project(x,c,cal)} for _ in range(3) for c in ('left','right')]
    observations.append({'camera':'right','raw_px':project(x,'right',cal)})
    result=fit_shared(observations,[x+[.01,-.01,.02]],cal)
    assert np.allclose(result[0],x,atol=1e-7)
    assert len(result[1])==7
    assert max(result[1])<1e-6


def test_shared_fit_never_overrides_failed_pair_gate():
    cal=calibration()
    reasons=gate_shared(['rejected']*5,[0]*10,np.array([.08,.12,.8]),cal)
    assert 'fewer_than_3_accepted_stereo_pairs' in reasons


def test_one_large_residual_cannot_hide_in_average():
    reasons=gate_shared(['accepted_manual_reference_geometry']*3,[0,0,0,0,0,11],np.array([.08,.12,.8]),calibration())
    assert 'shared_reprojection_over_10px' in reasons


def test_upright_raw_roundtrip_all_corners():
    for camera in ('left','right'):
        for point in ([0,0],[1079,0],[0,1919],[1079,1919],[800,300]):
            assert np.array_equal(raw_to_upright(upright_to_raw(point,camera),camera),point)
