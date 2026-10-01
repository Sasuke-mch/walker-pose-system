import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from construct_mirrored_grip_initialization import mirrored_left_parameters


def test_reflected_pose_rotation_and_points_preserve_proper_rotation():
    pose=np.arange(45).reshape(1,45)/100
    R=Rotation.from_rotvec([.1,-.2,.3]).as_matrix()
    t=np.array([-.23,.02,.89])
    p=np.array([.03,.01,-.04])
    left_pose,left_R,left_t=mirrored_left_parameters(pose,R,t)
    S=np.diag([-1.,1.,1.])
    assert np.allclose(left_R@(S@p)+left_t,S@(R@p+t))
    assert np.isclose(np.linalg.det(left_R),1)
    assert np.allclose(Rotation.from_rotvec(left_pose.reshape(15,3)).as_matrix(),S@Rotation.from_rotvec(pose.reshape(15,3)).as_matrix()@S)


def test_double_reflection_restores_parameters():
    pose=np.arange(45).reshape(1,45)/100
    R=Rotation.from_rotvec([.3,-.5,.8]).as_matrix()
    t=np.array([-.23,.02,.89])
    restored=mirrored_left_parameters(*mirrored_left_parameters(pose,R,t))
    for actual,expected in zip(restored,(pose,R,t)):
        assert np.allclose(actual,expected)
