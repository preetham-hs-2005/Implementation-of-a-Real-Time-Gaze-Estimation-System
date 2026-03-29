import unittest

from gaze.calibration import (
    HEAD_POSE_SEQUENCE,
    CalibrationModel,
    HeadPoseCalibrator,
    HeadPoseSequenceCalibrator,
    HeadPoseSample,
    apply_head_pose_compensation,
    build_gaze_feature_vector,
    default_calibration_targets,
    normalize_gaze_by_head_pose,
)


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

    def test_extended_calibration_grid_has_dense_targets(self):
        targets = default_calibration_targets(16)
        self.assertEqual(len(targets), 16)
        self.assertIn((0.08, 0.08), targets)
        self.assertIn((0.92, 0.92), targets)

    def test_head_reference_calibration_and_alignment(self):
        calibrator = HeadPoseCalibrator(required_samples=3)
        samples = [
            HeadPoseSample((0.50, 0.49), (0.34, 0.48), (0.50, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.01, 0.0, 0.0),
            HeadPoseSample((0.51, 0.50), (0.35, 0.49), (0.51, 0.54), (0.44, 0.43), (0.58, 0.43), 0.14, 0.02, -0.01, 0.01),
            HeadPoseSample((0.49, 0.50), (0.34, 0.47), (0.50, 0.55), (0.43, 0.44), (0.57, 0.44), 0.14, 0.0, 0.01, -0.01),
        ]
        for sample in samples:
            calibrator.add_sample(sample)
        reference = calibrator.build_reference()
        self.assertTrue(reference.is_aligned(samples[0]))

    def test_head_normalization_offsets_small_head_shift(self):
        reference_calibrator = HeadPoseCalibrator(required_samples=1)
        sample = HeadPoseSample((0.50, 0.50), (0.34, 0.48), (0.50, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.0, 0.0, 0.0)
        reference_calibrator.add_sample(sample)
        reference = reference_calibrator.build_reference()

        shifted = HeadPoseSample((0.55, 0.50), (0.34, 0.48), (0.55, 0.54), (0.48, 0.43), (0.62, 0.43), 0.14, 0.12, 0.0, 0.0)
        normalized = normalize_gaze_by_head_pose((0.60, 0.50), shifted, reference)
        self.assertLess(normalized[0], 0.60)

    def test_head_pose_sequence_builds_extremes(self):
        sequence = HeadPoseSequenceCalibrator(required_samples=1)
        samples = {
            "straight": HeadPoseSample((0.5, 0.5), (0.34, 0.48), (0.5, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.0, 0.0, 0.0),
            "left": HeadPoseSample((0.5, 0.5), (0.34, 0.48), (0.5, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, -0.18, 0.0, 0.0),
            "right": HeadPoseSample((0.5, 0.5), (0.34, 0.48), (0.5, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.20, 0.0, 0.0),
            "up": HeadPoseSample((0.5, 0.5), (0.34, 0.48), (0.5, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.0, -0.12, 0.0),
            "down": HeadPoseSample((0.5, 0.5), (0.34, 0.48), (0.5, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.0, 0.14, 0.0),
        }
        for label in HEAD_POSE_SEQUENCE:
            self.assertEqual(sequence.current_label, label)
            sequence.add_sample(samples[label])
        extremes = sequence.build_extremes()
        yaw_norm, pitch_norm, _ = extremes.normalize_angles(0.20, 0.14, 0.0)
        self.assertGreater(yaw_norm, 0.9)
        self.assertGreater(pitch_norm, 0.9)

    def test_feature_vector_contains_pose_and_eye_features(self):
        class Obs:
            gaze_norm = (0.5, 0.4)
            eye_gaze_norm = (0.1, -0.1)
            left_iris_relative = (0.08, -0.02)
            right_iris_relative = (0.09, -0.01)
            yaw = 0.05
            pitch = -0.03
            roll = 0.01

        reference_calibrator = HeadPoseCalibrator(required_samples=1)
        sample = HeadPoseSample((0.50, 0.50), (0.34, 0.48), (0.50, 0.54), (0.43, 0.43), (0.57, 0.43), 0.14, 0.0, 0.0, 0.0)
        reference_calibrator.add_sample(sample)
        reference = reference_calibrator.build_reference()
        features = build_gaze_feature_vector(Obs(), sample, reference, None)
        self.assertEqual(len(features), 11)
        self.assertAlmostEqual(features[2], 0.1)


if __name__ == "__main__":
    unittest.main()
