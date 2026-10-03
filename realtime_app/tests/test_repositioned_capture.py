"""Protect cross-dataset controls without requiring model weights or videos."""
from pathlib import Path
import copy
import tempfile
import unittest

import numpy as np

from experiment_repositioned_capture import (fit_commands, ROOT, read, checked_sapiens_rows,
                                             validate_saved_fit_controls, write)
from walker_tools.body.fit_smplh_wilor_sequence import build_parser
from evaluate_repositioned_capture import host_time_diagnostics, strict_first_instances
from pose_app.calibration import StereoCalibration
from pose_app.schema import PersonPose
from pose_app.triangulation import triangulate_matches


class RepositionedCaptureTests(unittest.TestCase):
    def test_changed_parser_defaults_cannot_reinterpret_recorded_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            commands, controls = fit_commands(out)
            write(out/'fit_controls.json', {'controls': controls})
            options = build_parser().parse_args(commands['pmpose'][4:])
            validate_saved_fit_controls(out, 'pmpose', options)
            options.beta_prior_weight *= 10
            with self.assertRaisesRegex(ValueError, 'resolved fit controls changed'):
                validate_saved_fit_controls(out, 'pmpose', options)

    def test_reordered_sapiens_source_is_rejected_before_renumbering(self):
        prompts, rows = [], []
        for i in range(2):
            prompts.append({'pair_id': 1157+i, 'file_name': f'pair_{i:04d}.png',
                            'detections': [{'bbox_xyxy': [0, 0, 100, 200], 'score': .8}]})
            rows.append({'pair_id': 1157+i, 'file_name': f'pair_{i:04d}.png', 'instances': [{
                'bbox': [0, 0, 100, 200], 'bbox_score': float(np.float32(.8)),
                'keypoints308': np.zeros((308, 2)).tolist(), 'keypoint_scores': np.ones(308).tolist()}]})
        self.assertEqual(len(checked_sapiens_rows(rows, prompts)), 2)
        with self.assertRaisesRegex(ValueError, 'source frame mismatch'):
            checked_sapiens_rows(rows[::-1], prompts)
        changed = copy.deepcopy(rows)
        changed[0]['instances'][0]['bbox'][2] = 99
        with self.assertRaisesRegex(ValueError, 'prompts differ'):
            checked_sapiens_rows(changed, prompts)

    def test_strict_diagnostic_does_not_replace_failed_first_instance(self):
        k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
        calibration = StereoCalibration('pinhole', (640, 480), (640, 480), k,
            np.zeros(5), k.copy(), np.zeros(5), np.eye(3), np.array([-.2, 0, 0]))
        def person(index, shift, score):
            points = [[320+shift+j, 230+j, score] for j in range(17)]
            return PersonPose(index, [250, 150, 450, 350], .9, score, points)
        left = [person(0, 0, .01), person(1, 0, .9)]
        right = [person(0, -50, .01), person(1, -50, .9)]
        left[0].bbox = [20, 10, 120, 80]
        right[0].bbox = [20, 400, 120, 470]
        all_instances = triangulate_matches(left, right, calibration, max_matches=1)
        self.assertEqual([(p.left_person_id, p.right_person_id) for p in all_instances], [(1, 1)])
        self.assertEqual(all_instances[0].valid_keypoints, 17)
        self.assertEqual(strict_first_instances(left, right, calibration), [])

    def test_irregular_host_intervals_do_not_create_acceleration(self):
        times = np.array([0., .04, .11, .20])
        joints = np.zeros((4, 22, 3))
        joints[:, [4, 5, 7, 8], 0] = times[:, None] * 2
        result = host_time_diagnostics(joints, times)
        self.assertAlmostEqual(result['leg_relative_speed_m_s']['rms'], 2)
        self.assertLess(result['leg_relative_acceleration_m_s2']['max'], 1e-12)
        with self.assertRaises(ValueError):
            host_time_diagnostics(joints, [0, .04, .04, .2])

    def test_same_fitter_and_controls_except_body_inputs(self):
        commands, controls = fit_commands(Path('isolated_new_capture'))
        self.assertEqual(controls['pmpose'], controls['sapiens2'])
        self.assertEqual(commands['pmpose'][3], commands['sapiens2'][3])
        self.assertEqual(controls['pmpose']['joint_steps'], 300)
        self.assertTrue(controls['pmpose']['shared_hand_pose'])
        self.assertTrue(controls['pmpose']['mano_pose_init'])

    def test_only_dataset_resources_and_wrist_reference_change(self):
        commands, _ = fit_commands(Path('new_capture'))
        old = vars(build_parser().parse_args(read(ROOT/'realtime_app/configs/cold_smplh_joint300.json')))
        new = vars(build_parser().parse_args(commands['sapiens2'][4:]))
        differences = {k for k in old if old[k] != new[k]}
        self.assertEqual(differences, {'left_raw', 'right_raw', 'wilor_left', 'wilor_right',
            'calibration_dir', 'output_dir', 'wrist_reference', 'wrist_reference_weight',
            'allow_diagnostic_wrist_reference'})
        self.assertIsNone(new['wrist_reference'])
        self.assertEqual(new['wrist_reference_weight'], 0)


if __name__ == '__main__':
    unittest.main()
