import unittest

import numpy as np

from learning_phc.grasp import free_velocity_predictor, grip_impulse, point_grasp, point_jacobian


class PointGraspTests(unittest.TestCase):
    def setUp(self):
        self.a = dict(p=np.array([.1, .2, .4]), R=np.eye(3),
                      com=np.array([.12, .2, .3]), v=np.array([.2, -.1, .3]),
                      w=np.array([.4, .3, -.2]))
        self.b = dict(p=np.array([.14, .19, .38]), R=np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]),
                      com=np.array([.12, .16, .35]), v=np.array([-.2, .1, .2]),
                      w=np.array([.1, -.2, .3]))

    def grip(self, **kwargs):
        return point_grasp(self.a, [.01, 0, -.02], self.b, [0, .02, .01], 400., 15., **kwargs)

    def test_pair_has_zero_total_force_and_moment(self):
        g = self.grip()
        np.testing.assert_allclose(g["force_a"]+g["force_b"], 0, atol=1e-14)
        np.testing.assert_allclose(np.cross(g["point"], g["force_a"])+
                                   np.cross(g["point"], g["force_b"]), 0, atol=1e-14)
        # Forces at the two separate anchors would leave a spurious couple.
        bad = np.cross(g["anchor_a"], g["force_a"])+np.cross(g["anchor_b"], g["force_b"])
        self.assertGreater(np.linalg.norm(bad), 1e-3)

    def test_power_matches_potential_and_dissipation(self):
        g = self.grip()
        self.assertAlmostEqual(g["mechanical_power_W"]+g["potential_rate_W"]+g["dissipation_W"], 0, places=12)
        self.assertGreaterEqual(g["dissipation_W"], 0)

    def test_common_rigid_motion_has_no_damping(self):
        omega = np.array([.5, -.2, .7])
        for state in (self.a, self.b):
            state["w"] = omega
            state["v"] = np.array([.1, .2, -.1])+np.cross(omega, state["com"])
        self.assertAlmostEqual(self.grip()["dissipation_W"], 0, places=12)

    def test_release_removes_all_connection_forces(self):
        g = self.grip(active=False)
        np.testing.assert_array_equal(g["force_a"], np.zeros(3))
        self.assertEqual(g["potential_J"], 0)
        self.assertEqual(g["mechanical_power_W"], 0)

    def test_parameters_must_be_physical(self):
        for k, d in ((0, 10), (400, -1), (np.nan, 2)):
            with self.assertRaises(ValueError):
                point_grasp(self.a, [0, 0, 0], self.b, [0, 0, 0], k, d)


class GripImpulseTests(unittest.TestCase):
    def test_free_predictor_removes_previous_constraint_response(self):
        old = np.array([.2, -.1])
        free_increment = np.array([.03, -.02])
        response = np.array([.11, .04])
        now = old+free_increment-response
        np.testing.assert_allclose(free_velocity_predictor(now, old, response), now+free_increment)
        np.testing.assert_array_equal(free_velocity_predictor(now), now)

    def test_constant_load_has_no_persistent_gap_with_free_prediction(self):
        dt, mobility = 1/240, np.eye(1)
        e, u, history = np.array([.0]), np.array([0.]), (None, None)
        for _ in range(240):
            free = free_velocity_predictor(u, *history)
            impulse, _ = grip_impulse(mobility, e, free, dt)
            history = (u.copy(), mobility@impulse)
            u = u+.1-mobility@impulse
            e = e+dt*u
        self.assertLess(abs(e[0]), 1e-12)
        self.assertLess(abs(u[0]), 1e-12)

    def test_zero_error_projection_removes_relative_motion_without_energy_gain(self):
        mobility = np.array([[2., .4, .1], [.4, 1., .2], [.1, .2, 3.]])
        velocity = np.array([.3, -.2, .1])
        impulse, work = grip_impulse(mobility, np.zeros(3), velocity, 1/240)
        np.testing.assert_allclose(velocity-mobility@impulse, 0, atol=1e-14)
        energy_change = -impulse@velocity+.5*impulse@mobility@impulse
        self.assertLess(energy_change, 0)
        self.assertAlmostEqual(energy_change+work['velocity_projection_dissipation_J'], 0)
        self.assertEqual(work['stabilization_work_J'], 0)

    def test_position_recovery_work_is_explicit(self):
        mobility = np.array([[2., .3], [.3, 1.]])
        velocity = np.array([-.1, .2])
        impulse, work = grip_impulse(mobility, [.01, -.02], velocity, 1/240)
        energy_change = -impulse@velocity+.5*impulse@mobility@impulse
        self.assertAlmostEqual(energy_change, work['stabilization_work_J']-work['velocity_projection_dissipation_J'])

    def test_recovery_decay_is_independent_of_timestep(self):
        def decay(dt):
            impulse, _ = grip_impulse(np.eye(1), [.01], [0.], dt)
            return 1-dt*impulse[0]/.01
        self.assertAlmostEqual(decay(1/240), decay(1/480)**2)

    def test_offset_point_velocity_includes_rotation(self):
        np.testing.assert_allclose(point_jacobian([.2, 0, 0], np.zeros(3))@np.array([0, 0, 0, 0, 0, 1]), [0, .2, 0])

    def test_unobservable_direction_and_bad_time_fail(self):
        with self.assertRaises(np.linalg.LinAlgError):
            grip_impulse(np.zeros((2, 2)), [0, 0], [1, 2], 1/240)
        with self.assertRaises(ValueError):
            grip_impulse(np.eye(2), [0, 0], [1, 2], 0)


if __name__ == "__main__":
    unittest.main()
