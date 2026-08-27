from __future__ import annotations

from types import SimpleNamespace
import unittest

from run_stereo import _strictly_better_geometry


def person(indices: list[int], reprojection: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        keypoints_3d=[{"index": index, "valid": True} for index in indices],
        mean_reprojection_error_px=reprojection,
    )


class LocalGeometrySelectionTests(unittest.TestCase):
    def test_more_valid_3d_points_wins(self) -> None:
        self.assertTrue(
            _strictly_better_geometry(
                [person([0, 1, 2, 3, 4], 1.0)],
                [person([0, 1, 2, 3, 4, 5], 5.0)],
            )
        )

    def test_same_information_needs_meaningful_reprojection_gain(self) -> None:
        baseline = [person([0, 1, 2, 3, 4, 5], 3.0)]
        self.assertTrue(
            _strictly_better_geometry(
                baseline, [person([0, 1, 2, 3, 4, 5], 2.70)]
            )
        )
        self.assertFalse(
            _strictly_better_geometry(
                baseline, [person([0, 1, 2, 3, 4, 5], 2.80)]
            )
        )

    def test_zero_baseline_requires_minimum_evidence(self) -> None:
        self.assertFalse(_strictly_better_geometry([], [person([0, 1, 2], 1.0)]))
        self.assertTrue(
            _strictly_better_geometry([], [person([0, 1, 2, 3], 1.0)])
        )

    def test_cannot_exchange_one_gait_joint_for_another(self) -> None:
        baseline = [person([0, 1, 11, 13, 15], 3.0)]
        changed = [person([0, 1, 11, 14, 15, 16], 2.0)]
        self.assertFalse(_strictly_better_geometry(baseline, changed))


if __name__ == "__main__":
    unittest.main()
