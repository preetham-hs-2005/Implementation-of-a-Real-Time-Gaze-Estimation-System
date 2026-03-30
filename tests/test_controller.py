import unittest

import numpy as np

from gaze.controller import CursorSmoother, HeadPoseSmoother, map_to_screen, apply_sensitivity


class ControllerTests(unittest.TestCase):
    def test_map_to_screen_bounds(self):
        x, y = map_to_screen(0.5, 0.5, 1920, 1080, 0.1)
        self.assertTrue(0 <= x <= 1920)
        self.assertTrue(0 <= y <= 1080)

    def test_smoother_moves_towards_target(self):
        smoother = CursorSmoother(min_cutoff=1.0, beta=0.0, d_cutoff=1.0, dead_zone=0.0)
        p0 = smoother.update((0, 0), 0.0)
        p1 = smoother.update((100, 0), 0.016)
        self.assertEqual(p0, (0, 0))
        self.assertTrue(0 < p1[0] < 100)

    def test_one_euro_filter_has_no_overshoot_for_step(self):
        smoother = CursorSmoother(min_cutoff=0.8, beta=0.005, d_cutoff=1.0, dead_zone=0.0)
        smoother.update((0.0, 0.0), 0.0)
        outputs = [smoother.update((100.0, 0.0), 0.016 * (i + 1))[0] for i in range(10)]
        self.assertTrue(all(0.0 <= value <= 100.0 for value in outputs))

    def test_apply_sensitivity_controls_extent(self):
        # Low sensitivity should push points toward center
        x, y = apply_sensitivity(0.9, 0.1, 0.5)
        self.assertTrue(0.5 < x < 0.9)
        self.assertTrue(0.1 < y < 0.5)

        # High sensitivity should stretch away from center
        x2, y2 = apply_sensitivity(0.9, 0.1, 1.5)
        self.assertTrue(x2 >= x)
        self.assertTrue(y2 <= y)

    def test_head_pose_smoother_reduces_jump(self):
        smoother = HeadPoseSmoother(alpha=0.25)
        identity = np.eye(3)
        p0, r0 = smoother.update(0.0, 0.0, 0.0, identity)
        rotation = np.array([[0.99, 0.0, 0.1], [0.0, 1.0, 0.0], [-0.1, 0.0, 0.99]])
        p1, r1 = smoother.update(1.0, -1.0, 0.5, rotation)
        self.assertEqual(p0, (0.0, 0.0, 0.0))
        self.assertTrue(0.0 < p1[0] < 1.0)
        self.assertTrue(-1.0 < p1[1] < 0.0)
        self.assertEqual(len(r0), 3)
        self.assertAlmostEqual(np.linalg.det(np.array(r1)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
