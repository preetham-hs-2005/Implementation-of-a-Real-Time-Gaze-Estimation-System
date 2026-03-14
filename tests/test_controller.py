import unittest

from gaze.controller import CursorSmoother, map_to_screen


class ControllerTests(unittest.TestCase):
    def test_map_to_screen_bounds(self):
        x, y = map_to_screen(0.5, 0.5, 1920, 1080, 0.1)
        self.assertTrue(0 <= x <= 1920)
        self.assertTrue(0 <= y <= 1080)

    def test_smoother_moves_towards_target(self):
        smoother = CursorSmoother(alpha=0.5, velocity_damping=0.5, max_step=1000)
        p0 = smoother.update((0, 0))
        p1 = smoother.update((100, 0))
        self.assertEqual(p0, (0, 0))
        self.assertTrue(0 < p1[0] < 100)


if __name__ == "__main__":
    unittest.main()
