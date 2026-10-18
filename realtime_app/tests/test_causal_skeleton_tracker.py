import sys
import unittest

import numpy as np

from pose_app.fisheye_camera import (
    StereoFisheyeCalibration,
    fisheye_project_numpy,
)


def _torch_env() -> bool:
    return "venv-smplx" in sys.executable


_TORCH_SKIP = unittest.skipUnless(_torch_env(), "torch runtime tests run in the SMPL-X venv")


def _calibration():
    K = np.asarray([[520.0, 0.0, 960.0], [0.0, 515.0, 540.0], [0.0, 0.0, 1.0]])
    D = np.asarray([-0.05, 0.008, -0.001, 0.0002])
    return StereoFisheyeCalibration(
        K0=K, D0=D, K1=K, D1=D,
        R_cam0_to_cam1=np.eye(3),
        T_cam0_to_cam1_mm=np.asarray([-100.0, 0.0, 0.0]),
        image_width=1920, image_height=1080,
    )


def _synthetic_sequence(frame_count: int):
    """A standing person translating along +X. World: +Z up, +X camera forward."""
    from pose_app.skeleton_factor_graph import BODY_JOINTS

    cal = _calibration()
    base = np.zeros((17, 3))
    base[[0, 1, 2, 3, 4]] = np.nan  # head/neck not used
    base[5] = [0.0, -130.0, 1400.0]   # Lsh
    base[6] = [0.0, 130.0, 1400.0]    # Rsh
    base[7] = [0.0, -130.0, 1150.0]   # Lel
    base[8] = [0.0, 130.0, 1150.0]    # Rel
    base[9] = [0.0, -130.0, 950.0]    # Lwr
    base[10] = [0.0, 130.0, 950.0]    # Rwr
    base[11] = [0.0, -100.0, 900.0]   # Lhi
    base[12] = [0.0, 100.0, 900.0]    # Rhi
    base[13] = [0.0, -100.0, 500.0]   # Lkn
    base[14] = [0.0, 100.0, 500.0]    # Rkn
    base[15] = [0.0, -100.0, 50.0]    # Lank
    base[16] = [0.0, 100.0, 50.0]     # Rank
    base[:, 0] += 1500.0  # in front of the camera

    motion_frames = []
    stereo_rows = []
    interactions = []
    for frame in range(frame_count):
        world = base.copy()
        world[:, 0] += frame * 10.0
        motion_frames.append({
            "pair": frame,
            "stage": "warming_up",
            "points_mm": [row.tolist() if np.all(np.isfinite(row)) else None for row in world],
            "errors_px": [2.0 if np.all(np.isfinite(row)) else None for row in world],
            "strict": [bool(np.all(np.isfinite(row))) for row in world],
            "rotation_world_from_left": np.eye(3).tolist(),
            "translation_world_from_left_mm": [0.0, 0.0, 0.0],
        })
        left = world[list(BODY_JOINTS)]
        right = (cal.R_cam0_to_cam1 @ left.T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
        kp_l = fisheye_project_numpy(left, cal.K0, cal.D0)
        kp_r = fisheye_project_numpy(right, cal.K1, cal.D1)
        keypoints_l = np.full((17, 3), np.nan)
        keypoints_r = np.full((17, 3), np.nan)
        for position, joint in enumerate(BODY_JOINTS):
            keypoints_l[joint] = [kp_l[position, 0], kp_l[position, 1], 0.9]
            keypoints_r[joint] = [kp_r[position, 0], kp_r[position, 1], 0.9]
        stereo_rows.append({
            "pair_id": frame,
            "left": {"persons": [{"keypoints": keypoints_l.tolist()}]},
            "right": {"persons": [{"keypoints": keypoints_r.tolist()}]},
            "persons_3d": [],
        })
        interactions.append({
            "hands": {
                "left": {"closest_proxy_point_world_mm": world[9].tolist(), "wrist_to_handle_axis_distance_mm": 0.0},
                "right": {"closest_proxy_point_world_mm": world[10].tolist(), "wrist_to_handle_axis_distance_mm": 0.0},
            }
        })
    prefit = np.tile(base[None, ...], (frame_count, 1, 1))
    prefit[:, :, 0] += np.arange(frame_count)[:, None] * 10.0
    return cal, motion_frames, stereo_rows, interactions, prefit


@_TORCH_SKIP
class NumpyTorchProjectionConsistencyTests(unittest.TestCase):
    def test_numpy_and_torch_fisheye_projection_agree(self):
        import torch
        from pose_app.fisheye_camera import fisheye_project_torch

        cal = _calibration()
        points_m = np.asarray([[1.5, -0.1, 2.0], [0.5, 0.3, 1.4], [-0.2, 0.1, 1.1]])
        numpy_out = fisheye_project_numpy(points_m * 1000.0, cal.K0, cal.D0)
        torch_out = fisheye_project_torch(
            torch.as_tensor(points_m, dtype=torch.float32), cal.K0, cal.D0
        )
        np.testing.assert_allclose(numpy_out, torch_out.detach().cpu().numpy(), atol=1e-5)


@_TORCH_SKIP
class CausalSkeletonTrackerTests(unittest.TestCase):
    def setUp(self):
        from pose_app.skeleton_factor_graph import BODY_JOINTS, estimate_bone_lengths_mm

        self.cal, self.motion_frames, self.stereo_rows, self.interactions, self.prefit = _synthetic_sequence(20)
        self.frozen = estimate_bone_lengths_mm(self.prefit[:6, list(BODY_JOINTS)])
        self.assertTrue(np.all(np.isfinite(self.frozen)) and np.all(self.frozen > 0))

    def _tracker(self, iterations=10, window_size=15):
        from pose_app.causal_skeleton_tracker import CausalSkeletonConfig, CausalSkeletonTracker

        config = CausalSkeletonConfig(window_size=window_size, iterations_per_update=iterations)
        return CausalSkeletonTracker(config, self.cal, self.frozen, self.prefit, device="cpu")

    def test_causal_window_never_reads_future_frames(self):
        class RecordingArray:
            def __init__(self, data):
                self._data = data
                self.accessed = []

            def __getitem__(self, key):
                index = int(key)
                self.accessed.append(index)
                return self._data[index]

        tracker = self._tracker()
        tracker.prefit_points_mm = RecordingArray(self.prefit)
        for frame in range(0, 6):
            tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
        # Prefit is only touched for cold-start initialisation (frame 0); the
        # constant-velocity warm start must never read the future or the prefit.
        self.assertLessEqual(max(tracker.prefit_points_mm.accessed), 0)

    def test_warm_start_frame_order_and_history_limits(self):
        tracker = self._tracker(window_size=15)
        for frame in range(0, 20):
            tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
        self.assertEqual(sorted(tracker.output_history.keys()), list(range(20)))
        for frame, row in tracker.output_history.items():
            self.assertEqual(row["algorithmic_lookahead_frames"], 0)
            expected_history = min(frame, 14)
            self.assertEqual(row["window_history_frames"], expected_history)
        self.assertEqual(set(tracker._param_by_frame.keys()), set(range(20)))

    def test_frozen_bones_never_change_during_update(self):
        tracker = self._tracker()
        before = tracker.frozen_bone_lengths_mm.copy()
        created = []

        import pose_app.causal_skeleton_tracker as tracker_module
        original = tracker_module.FixedBoneSkeleton

        class RecordingSkeleton(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        tracker_module.FixedBoneSkeleton = RecordingSkeleton
        try:
            for frame in range(0, 5):
                tracker.update(
                    frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
                )
        finally:
            tracker_module.FixedBoneSkeleton = original
        np.testing.assert_array_equal(tracker.frozen_bone_lengths_mm, before)
        self.assertTrue(created)
        for skeleton in created:
            self.assertFalse(skeleton.bone_lengths.requires_grad)
            np.testing.assert_allclose(
                skeleton.bone_lengths.detach().cpu().numpy() * 1000.0, self.frozen, atol=1e-6
            )

    def test_iteration_counts_are_exact(self):
        import torch

        original_adam = torch.optim.Adam
        created = []

        class CountingAdam:
            def __init__(self, params, **kwargs):
                self._inner = original_adam(params, **kwargs)
                self.step_count = 0
                created.append(self)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def step(self, *args, **kwargs):
                self.step_count += 1
                return self._inner.step(*args, **kwargs)

            def zero_grad(self):
                return self._inner.zero_grad()

        try:
            for iterations in (5, 10, 20):
                created.clear()
                torch.optim.Adam = CountingAdam
                tracker = self._tracker(iterations=iterations)
                tracker.update(0, self.motion_frames[0], self.stereo_rows[0], None, self.interactions[0])
                self.assertEqual(len(created), 1)
                self.assertEqual(created[0].step_count, iterations)
        finally:
            torch.optim.Adam = original_adam

    def test_missing_observations_do_not_break_update(self):
        tracker = self._tracker()
        for frame in range(0, 3):
            tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
        # A fully missing frame (all keypoints NaN in both views).
        missing_row = {
            "pair_id": 99,
            "left": {"persons": [{"keypoints": np.full((17, 3), np.nan).tolist()}]},
            "right": {"persons": [{"keypoints": np.full((17, 3), np.nan).tolist()}]},
            "persons_3d": [],
        }
        motion_frame = dict(self.motion_frames[0])
        motion_frame["pair"] = 99
        result = tracker.update(99, motion_frame, missing_row, None, self.interactions[0])
        output = np.asarray(result["joints_ground_mm"])
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(result["nan_count"], 0)

    def test_nan_is_reported_not_silently_written(self):
        tracker = self._tracker()
        rows = []
        for frame in range(0, 4):
            result = tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
            rows.append(result)
        for result in rows:
            self.assertIn("nan_count", result)
            output = np.asarray(result["joints_ground_mm"])
            self.assertEqual(result["nan_count"], int(np.count_nonzero(~np.isfinite(output))))

    def test_old_frame_outputs_are_never_rewritten(self):
        tracker = self._tracker()
        for frame in range(0, 4):
            tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
        captured = dict(tracker.output_history[0])
        for frame in range(4, 8):
            tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
        self.assertEqual(tracker.output_history[0], captured)

    def test_output_is_reasonable_for_exact_cold_start(self):
        from pose_app.skeleton_factor_graph import BODY_JOINTS

        tracker = self._tracker(iterations=20)
        for frame in range(0, 6):
            tracker.update(
                frame, self.motion_frames[frame], self.stereo_rows[frame], None, self.interactions[frame]
            )
        # With an exact cold start the fitted joints stay close to the truth.
        for frame in range(0, 6):
            truth = self.prefit[frame, list(BODY_JOINTS)]
            fitted = np.asarray(tracker.output_history[frame]["joints_ground_mm"])
            error = np.linalg.norm(fitted - truth, axis=1)
            self.assertLess(np.median(error), 30.0)


if __name__ == "__main__":
    unittest.main()
