import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_app.independent_wrist_reference import load_reference, wrist_position_loss


class WristReferenceTests(unittest.TestCase):
    def test_internal_wrists_units_and_gradient(self):
        joints = torch.zeros(2, 52, 3, requires_grad=True)
        targets = torch.tensor([[0., 0., .01], [0., 0., .02]])
        loss = wrist_position_loss(joints, targets)
        self.assertAlmostEqual(loss.item(), .00025, places=8)
        loss.backward()
        self.assertGreater(joints.grad[:, [20, 21]].abs().sum().item(), 0)
        self.assertEqual(joints.grad[:, [9, 10]].abs().sum().item(), 0)

    def test_shape_gradient_is_not_detached(self):
        beta = torch.tensor(0., requires_grad=True)
        joints = beta * torch.ones(3, 52, 3)
        wrist_position_loss(joints, torch.ones(2, 3)).backward()
        self.assertAlmostEqual(beta.grad.item(), -6., places=5)

    def test_rejected_input_explicit_and_no_constructed_grasp(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'target.json'
            data = dict(schema='annotated_wrist_targets_v1', finger_parameters_used=False,
                assumption='wrist_fixed_relative_to_walker_with_rigid_camera_mount',
                calibration_dir=d, source_annotations=d, status='incomplete_or_rejected',
                hands={s: dict(state='rejected', reasons=['shared_reprojection_over_10px'],
                              diagnostic_wrist_left_camera_m=[.1, .2, .3]) for s in ('left','right')})
            p.write_text(json.dumps(data), encoding='utf8')
            with self.assertRaises(ValueError): load_reference(p, d)
            pts, audit = load_reference(p, d, True)
            np.testing.assert_allclose(pts, [[.1,.2,.3]] * 2)
            self.assertEqual(audit['hands']['left']['state'], 'rejected')
            with self.assertRaises(ValueError): load_reference(p, Path(d)/'other', True)
            data['schema'] = 'constructed_bilateral_grasp_v1'
            p.write_text(json.dumps(data), encoding='utf8')
            with self.assertRaises(ValueError): load_reference(p, d, True)


if __name__ == '__main__':
    unittest.main()
