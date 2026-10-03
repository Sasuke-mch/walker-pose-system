"""Protect cross-dataset controls without requiring model weights or videos."""
from pathlib import Path
import unittest

import numpy as np

from experiment_repositioned_capture import fit_commands, ROOT, read
from walker_tools.body.fit_smplh_wilor_sequence import build_parser
from evaluate_repositioned_capture import host_time_diagnostics


class RepositionedCaptureTests(unittest.TestCase):
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
