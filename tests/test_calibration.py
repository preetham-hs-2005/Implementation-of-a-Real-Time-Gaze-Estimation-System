import unittest

from gaze.calibration import CalibrationModel, apply_head_pose_compensation


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


if __name__ == "__main__":
    unittest.main()
