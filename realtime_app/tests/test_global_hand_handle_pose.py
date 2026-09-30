import numpy as np

from pose_app.global_hand_handle_pose import estimate_global_hand_handle_pose


def test_global_pose_recovers_fixed_offset():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(40, 3)) * np.array([0.03, 0.01, 0.02])
    n = 20
    ends = np.zeros((n, 2, 3), float)
    ends[:, 0] = [-0.2, 0.0, 1.0]
    ends[:, 1] = [0.2, 0.0, 1.0]
    palms = np.repeat(base[None], n, axis=0) + np.array([0.0, 0.04, 0.0])
    palms += rng.normal(scale=1e-4, size=palms.shape)
    got = estimate_global_hand_handle_pose(palms, ends)
    assert got["status"] == "engineering_candidate"
    assert np.linalg.norm(np.asarray(got["palm_offset_handle_m"]) - [0, .04, -1.0]) < 1e-2
    assert len(got["valid_frames"]) == n


def test_outlier_is_retained_as_rejected_frame():
    base = np.zeros((10, 3)); base[:, 0] = np.linspace(-.02, .02, 10)
    palms = np.repeat(base[None], 8, axis=0)
    palms[3] += 0.5
    ends = np.zeros((8, 2, 3)); ends[:, 0] = [-.2, 0, 1]; ends[:, 1] = [.2, 0, 1]
    got = estimate_global_hand_handle_pose(palms, ends)
    assert any(x["frame"] == 3 for x in got["rejected_frames"])
