import unittest

from gaze.controller import CursorSmoother, apply_precision_curve, map_to_screen, apply_sensitivity


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

    def test_apply_sensitivity_controls_extent(self):
        # Low sensitivity should push points toward center
        x, y = apply_sensitivity(0.9, 0.1, 0.5)
        self.assertTrue(0.5 < x < 0.9)
        self.assertTrue(0.1 < y < 0.5)

        # High sensitivity should stretch away from center
        x2, y2 = apply_sensitivity(0.9, 0.1, 1.5)
        self.assertTrue(x2 >= x)
        self.assertTrue(y2 <= y)

    def test_precision_curve_holds_small_center_motion(self):
        x, y = apply_precision_curve(0.52, 0.48, deadzone=0.04, curve_power=1.8)
        self.assertEqual((x, y), (0.5, 0.5))

    def test_precision_curve_preserves_large_movement(self):
        x, y = apply_precision_curve(0.9, 0.1, deadzone=0.04, curve_power=1.8)
        self.assertTrue(0.5 < x <= 1.0)
        self.assertTrue(0.0 <= y < 0.5)


if __name__ == "__main__":
    unittest.main()
