"""Coordinate direction, metric units and missing-pose regressions for scene comparison."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from walker_tools.scene.compare_visual_rig_poses import (
    export_world_geometry, measured_pose, projected_errors, read_rows, support_summary,
    observation_mask_diagnostics,
    compare,
)
from walker_tools.scene.prepare_visual_rig_comparison import raw_points, scaled_K
from walker_tools.scene.cuvslam_backend import quaternion_xyzw, matrix_from_pose, build_rig
from pose_app.calibration import StereoCalibration


def calibration():
    K = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
    return StereoCalibration('fisheye', (200, 200), (200, 200), K, np.zeros(4),
                             K, np.zeros(4), np.eye(3), np.array([-100., 0, 0]), 'mm')


class VisualRigComparisonTests(unittest.TestCase):
    def test_raw_mesh_comparator_rejects_rectified_coordinate_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder/'manifest.json'
            source.write_text(json.dumps({'frames': [], 'calibration': {'model': 'opencv_pinhole'}}))
            with self.assertRaisesRegex(ValueError, 'cannot consume rectified poses'):
                compare(SimpleNamespace(output=folder/'output', manifest=source))

    def test_raw_inverse_rotation_and_pixel_center_resize(self):
        upright = np.array([[120., 700., .8]])
        np.testing.assert_array_equal(raw_points(upright, 'left', (1920, 1080)), [[1219, 120, .8]])
        np.testing.assert_array_equal(raw_points(upright, 'right', (1920, 1080)), [[700, 959, .8]])
        K = scaled_K(np.array([[100., 0, 90], [0, 100, 80], [0, 0, 1]]), .5)
        self.assertEqual(K[0, 2], 44.75)

    def test_prediction_projects_previous_points_with_meter_translation(self):
        cal = calibration()
        xyz = np.array([[0., 0., 1000], [100., 100., 1500]])
        delta = np.eye(4)
        delta[0, 3] = .1
        expected = xyz + [100., 0, 0]
        errors, invalid = projected_errors(delta, xyz, cal.project_left(expected), cal.project_right(expected), cal)
        np.testing.assert_allclose(errors, 0, atol=1e-10)
        self.assertEqual(invalid, 0)
        delta[2, 3] = -2
        errors, invalid = projected_errors(delta, xyz, np.zeros((2, 2)), np.zeros((2, 2)), cal)
        self.assertEqual((errors, invalid), ([], 2))

    def test_failed_native_proposal_is_not_measured(self):
        self.assertIsNone(measured_pose({'status': 'initializing', 'world_from_left_camera_m': np.eye(4).tolist()}))
        with self.assertRaises(ValueError):
            measured_pose({'status': 'tracked', 'world_from_left_camera_m': np.zeros((4, 4)).tolist()})

    def test_new_rig_right_extrinsic_direction(self):
        class Pose:
            def __init__(self, rotation, translation):
                self.rotation, self.translation = rotation, translation
        class Dummy:
            pass
        Dummy.Pose = Pose
        Dummy.Camera = lambda **kwargs: type('Camera', (), kwargs)()
        Dummy.Rig = lambda **kwargs: type('Rig', (), kwargs)()
        Dummy.Distortion = type('Distortion', (), {'Model': type('Model', (), {'Fisheye': 1}),
                              '__init__': lambda self, model, coefficients: None})
        R = np.diag([-1., -1., 1.])
        value = {'R_right_from_left': R.tolist(), 'translation_right_from_left_m': [.1, .2, .3],
                 'image_size': [200, 200]}
        for side in ('left', 'right'):
            value[f'{side}_K'] = calibration().left_K.tolist()
            value[f'{side}_D'] = [0.] * 4
        rig = build_rig(Dummy, value)
        right = matrix_from_pose(rig.cameras[1].rig_from_camera)
        np.testing.assert_allclose(right[:3, :3], R.T, atol=1e-12)
        np.testing.assert_allclose(right[:3, 3], -R.T @ [.1, .2, .3], atol=1e-12)
        np.testing.assert_allclose(matrix_from_pose(Pose(quaternion_xyzw(R), [0, 0, 0]))[:3, :3], R, atol=1e-12)

    def test_frozen_body_walker_cameras_share_transform_and_failures_stay_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder / 'body.npz', pair_id=[0, 1], predicted_coco=np.zeros((2, 17, 3)),
                     vertices=np.zeros((2, 3, 3)), faces=[[0, 1, 2]])
            model = {'nodes_walker_mm': {'foot': [1000, 0, 0]},
                     'rotation_left_camera_from_walker': np.eye(3).tolist(),
                     'translation_left_camera_from_walker_mm': [0, 1000, 0]}
            (folder / 'walker.json').write_text(json.dumps(model))
            pose = np.eye(4)
            pose[:3, 3] = [1, 2, 3]
            rows = [{'pair_id': 0, 'status': 'tracked', 'world_from_left_camera_m': pose.tolist()},
                    {'pair_id': 1, 'status': 'lost', 'world_from_left_camera_m': None}]
            export_world_geometry(rows, folder / 'body.npz', folder / 'walker.json', calibration(), folder / 'world.npz')
            with np.load(folder / 'world.npz') as result:
                np.testing.assert_array_equal(result['world_body_coco_m'][0, 0], [1, 2, 3])
                np.testing.assert_array_equal(result['world_body_vertices_m'][0, 0], [1, 2, 3])
                np.testing.assert_array_equal(result['faces'], [[0, 1, 2]])
                np.testing.assert_array_equal(result['world_walker_nodes_m'][0, 0], [2, 3, 3])
                np.testing.assert_allclose(result['world_camera_centers_m'][0], [[1, 2, 3], [1.1, 2, 3]])
                self.assertTrue(np.isnan(result['world_body_coco_m'][1]).all())
                self.assertTrue(np.isnan(result['world_body_vertices_m'][1]).all())

    def test_pair_reorder_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'poses.jsonl'
            path.write_text(json.dumps({'frame_id': 0, 'pair_id': 8}) + '\n')
            with self.assertRaises(ValueError):
                read_rows(path, [{'frame_id': 0, 'pair_id': 9}])

    def test_support_counts_do_not_double_count_or_include_outliers(self):
        pose = np.eye(4).tolist()
        orb = {'status': 'tracked', 'world_from_left_camera_m': pose,
               'observations': [{'camera_index': 0, 'map_point_id': 1},
                                {'camera_index': 1, 'map_point_id': 1},
                                {'camera_index': 1, 'map_point_id': 2, 'outlier': True}]}
        cu = {'status': 'tracked', 'world_from_left_camera_m': pose,
              'finite_landmark_count': 1, 'observations': [[{'id': 1}, {'id': 2}], [{'id': 1}]]}
        for row in (orb, cu):
            result = support_summary([row])
            self.assertEqual(result['exported_3d_support_per_pose_frame']['median'], 1)
            self.assertEqual(result['same_frame_left_right_shared_observation_ids']['median'], 1)

    def test_deep_mask_overlap_is_not_ignored_as_a_rounding_boundary(self):
        depth = np.zeros((10, 10), dtype=float)
        depth[4, 4] = 5.0
        row = {'observations': [[{'id': 1, 'u': 1., 'v': 1.}],
                                [{'id': 1, 'u': 4., 'v': 4.}]]}
        result = observation_mask_diagnostics(row, [depth, depth])['per_camera']
        self.assertEqual(result[0]['deep_excluded_center_count_gt_3px'], 0)
        self.assertEqual(result[1]['deep_excluded_center_count_gt_3px'], 1)


if __name__ == '__main__':
    unittest.main()
