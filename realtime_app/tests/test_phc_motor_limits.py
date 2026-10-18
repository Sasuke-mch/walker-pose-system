import io
import unittest

import numpy as np

from learning_phc.contract import motor_effort_limits


class MotorLimitTests(unittest.TestCase):
    def xml(self, motors, default='ctrlrange="-1 1" ctrllimited="true"'):
        return io.StringIO('<mujoco><default><motor '+default+'/></default><actuator>'
                           +motors+'</actuator></mujoco>')

    def test_inherited_bounds_override_and_requested_order(self):
        xml = self.xml('<motor joint="a" gear="500"/><motor joint="b" gear="20" ctrlrange="-.5 .5"/>')
        np.testing.assert_array_equal(motor_effort_limits(xml, ["b", "a"]), [10, 500])

    def test_missing_duplicate_or_extra_actuator_fails(self):
        for motors in ('', '<motor joint="a"/><motor joint="a"/>',
                       '<motor joint="a"/><motor joint="b"/>'):
            with self.assertRaises(ValueError):
                motor_effort_limits(self.xml(motors), ["a"])

    def test_unsupported_bounds_are_not_invented(self):
        for attrs in ('ctrllimited="false"', 'gear="0"', 'gear="1 0 0 0 0 0"',
                      'ctrlrange="-1 2"', 'gear="nan"', 'class="other"'):
            with self.assertRaises(ValueError):
                motor_effort_limits(self.xml('<motor joint="a" '+attrs+'/>'), ["a"])

    def test_missing_default_bounds_fails(self):
        with self.assertRaises(ValueError):
            motor_effort_limits(self.xml('<motor joint="a"/>', default=''), ["a"])


if __name__ == "__main__":
    unittest.main()
