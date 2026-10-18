import unittest

from learning_phc.foundation_motion import ReferenceClock


class ReferenceClockTests(unittest.TestCase):
    def test_nonuniform_times_use_latest_arrived_frame(self):
        clock = ReferenceClock([0., .057, .108, .115], [True]*4, [0]*4)
        self.assertEqual(clock.at(.056), (0, .056))
        self.assertEqual(clock.at(.057), (1, 0.))
        self.assertEqual(clock.at(.114)[0], 2)
        self.assertAlmostEqual(clock.at(.114)[1], .006)

    def test_missing_arrived_frame_is_not_silently_held(self):
        clock = ReferenceClock([0, .05], [True, False], [0, 0])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            clock.at(.06)

    def test_reinitialization_is_not_a_continuous_target_jump(self):
        clock = ReferenceClock([0, .05], [True, True], [0, 1])
        with self.assertRaisesRegex(ValueError, 'reinitialization'):
            clock.at(.05)

    def test_no_future_or_invalid_clock(self):
        clock = ReferenceClock([.1], [True], [0])
        for t in (.09, float('nan')):
            with self.assertRaises(ValueError):
                clock.at(t)
        with self.assertRaises(ValueError):
            ReferenceClock([0, 0], [True]*2, [0]*2)


if __name__ == '__main__':
    unittest.main()
