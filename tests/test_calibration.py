import unittest

from gaze.calibration import CalibrationModel, apply_head_pose_compensation, average_points, is_stable_gaze


class CalibrationTests(unittest.TestCase):
    def test_quadratic_calibration_fit(self):
        model = CalibrationModel()
        samples = [
            ((0.1, 0.1), (0.12, 0.11)),
            ((0.9, 0.1), (0.88, 0.12)),
            ((0.5, 0.5), (0.5, 0.5)),
            ((0.1, 0.9), (0.12, 0.88)),
            ((0.9, 0.9), (0.87, 0.89)),
            ((0.5, 0.1), (0.5, 0.12)),
        ]
        for gaze, target in samples:
            model.add_sample(gaze, target)
        model.fit()
        mapped = model.map((0.5, 0.5))
        self.assertTrue(0.4 <= mapped[0] <= 0.6)
        self.assertTrue(0.4 <= mapped[1] <= 0.6)

    def test_pose_compensation_stable_without_matrix(self):
        self.assertEqual(apply_head_pose_compensation((0.4, 0.6), None), (0.4, 0.6))

    def test_average_points(self):
        self.assertEqual(average_points([(0.4, 0.6), (0.6, 0.4)]), (0.5, 0.5))

    def test_is_stable_gaze_detects_small_motion(self):
        self.assertTrue(is_stable_gaze([(0.5, 0.5), (0.51, 0.49), (0.495, 0.505)], 0.02))
        self.assertFalse(is_stable_gaze([(0.5, 0.5), (0.55, 0.49), (0.495, 0.505)], 0.02))

    def test_interaction_refinement_keeps_recent_window(self):
        model = CalibrationModel(max_interaction_points=2)
        for gaze, target in [
            ((0.1, 0.1), (0.1, 0.1)),
            ((0.9, 0.1), (0.9, 0.1)),
            ((0.5, 0.5), (0.5, 0.5)),
            ((0.1, 0.9), (0.1, 0.9)),
            ((0.9, 0.9), (0.9, 0.9)),
        ]:
            model.add_sample(gaze, target)
        self.assertTrue(model.refine_from_interaction((0.2, 0.2), (0.2, 0.2)))
        self.assertTrue(model.refine_from_interaction((0.3, 0.3), (0.3, 0.3)))
        self.assertTrue(model.refine_from_interaction((0.4, 0.4), (0.4, 0.4)))
        self.assertEqual(len(model.interaction_points), 2)


if __name__ == "__main__":
    unittest.main()
