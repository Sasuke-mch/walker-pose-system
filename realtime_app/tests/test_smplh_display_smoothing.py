"""Mild display smoothing must respect SO3 charts, caps and missing support."""
import numpy as np
from scipy.spatial.transform import Rotation
from tools.smooth_smplh_display import smooth_rotations, smooth_translation


def test_rotation_smoothing_crosses_pi_correctly_and_respects_gaps():
    angles = np.deg2rad([179, -179, 179, 60, -60])
    vectors = np.column_stack((np.zeros(5), np.zeros(5), angles))
    eligible = np.array([True, False, True])
    result = smooth_rotations(vectors, eligible)
    original = Rotation.from_rotvec(vectors)
    corrections = (original.inv() * Rotation.from_rotvec(result)).magnitude()
    assert np.rad2deg(corrections).max() <= 2 + 1e-10
    assert abs(np.rad2deg(result[1, 2])) > 178
    np.testing.assert_array_equal(result[[0, 2, 4]], vectors[[0, 2, 4]])


def test_translation_does_not_fill_gaps_or_exceed_three_mm():
    values = np.array([[0., 0., 0.], [1., 2., 3.], [0., 0., 0.], [4., 5., 6.], [0., 0., 0.]])
    result = smooth_translation(values, np.array([True, False, True]))
    assert np.linalg.norm(result-values, axis=-1).max() <= .003 + 1e-12
    np.testing.assert_array_equal(result[[0, 2, 4]], values[[0, 2, 4]])
