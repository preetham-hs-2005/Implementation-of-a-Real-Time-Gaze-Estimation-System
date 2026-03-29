from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
FeatureVector = Tuple[float, ...]


def _poly_features(features: Sequence[float]) -> np.ndarray:
    arr = np.array(features, dtype=np.float64)
    feats = [1.0, *arr.tolist()]
    feats.extend((arr * arr).tolist())
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            feats.append(float(arr[i] * arr[j]))
    return np.array(feats, dtype=np.float64)


@dataclass
class CalibrationModel:
    """Regularized quadratic mapper from gaze/head features to screen point."""

    coef_x: Optional[np.ndarray] = None
    coef_y: Optional[np.ndarray] = None
    points: List[Tuple[FeatureVector, Point]] = field(default_factory=list)

    def add_sample(self, gaze_features: Sequence[float], screen_norm: Point) -> None:
        self.points.append((tuple(float(v) for v in gaze_features), screen_norm))

    def clear(self) -> None:
        self.points.clear()
        self.coef_x = None
        self.coef_y = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    def fit(self, regularization: float = 1e-5) -> None:
        if len(self.points) < 5:
            raise ValueError("Need at least 5 samples for calibration fit.")
        features = np.stack([_poly_features(gaze_features) for gaze_features, _ in self.points], axis=0)
        target_x = np.array([sx for _, (sx, _) in self.points], dtype=np.float64)
        target_y = np.array([sy for _, (_, sy) in self.points], dtype=np.float64)

        # Regularized least squares for more stable calibration, reducing over-sensitive overfitting.
        xtx = features.T @ features
        reg_mat = np.eye(xtx.shape[0], dtype=np.float64) * regularization
        self.coef_x = np.linalg.solve(xtx + reg_mat, features.T @ target_x)
        self.coef_y = np.linalg.solve(xtx + reg_mat, features.T @ target_y)

    def map(self, gaze_features: Sequence[float]) -> Point:
        if not self.is_fitted:
            return max(0.0, min(1.0, float(gaze_features[0]))), max(0.0, min(1.0, float(gaze_features[1])))
        feats = _poly_features(gaze_features)
        out_x = float(feats @ self.coef_x)
        out_y = float(feats @ self.coef_y)
        return max(0.0, min(1.0, out_x)), max(0.0, min(1.0, out_y))


@dataclass
class HeadPoseSample:
    face_center_norm: Point
    face_size_norm: Point
    nose_norm: Point
    left_eye_center_norm: Point
    right_eye_center_norm: Point
    interocular_distance_norm: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class HeadPoseReference:
    face_center_norm: Point
    face_size_norm: Point
    nose_norm: Point
    left_eye_center_norm: Point
    right_eye_center_norm: Point
    interocular_distance_norm: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    center_tolerance: Point = (0.08, 0.08)
    size_tolerance: Point = (0.15, 0.18)
    eye_distance_tolerance: float = 0.12
    angle_tolerance: Tuple[float, float, float] = (0.30, 0.24, 0.25)

    def alignment_score(self, sample: HeadPoseSample) -> float:
        center_dx = abs(sample.face_center_norm[0] - self.face_center_norm[0]) / max(self.center_tolerance[0], 1e-6)
        center_dy = abs(sample.face_center_norm[1] - self.face_center_norm[1]) / max(self.center_tolerance[1], 1e-6)
        size_dx = abs(sample.face_size_norm[0] - self.face_size_norm[0]) / max(self.size_tolerance[0], 1e-6)
        size_dy = abs(sample.face_size_norm[1] - self.face_size_norm[1]) / max(self.size_tolerance[1], 1e-6)
        eye_dist = abs(sample.interocular_distance_norm - self.interocular_distance_norm) / max(self.eye_distance_tolerance, 1e-6)
        yaw = abs(sample.yaw - self.yaw) / max(self.angle_tolerance[0], 1e-6)
        pitch = abs(sample.pitch - self.pitch) / max(self.angle_tolerance[1], 1e-6)
        roll = abs(sample.roll - self.roll) / max(self.angle_tolerance[2], 1e-6)
        return float(max(center_dx, center_dy, size_dx, size_dy, eye_dist, yaw, pitch, roll))

    def is_aligned(self, sample: HeadPoseSample) -> bool:
        return self.alignment_score(sample) <= 1.0


@dataclass
class HeadPoseCalibrator:
    required_samples: int = 30
    samples: List[HeadPoseSample] = field(default_factory=list)

    def reset(self) -> None:
        self.samples.clear()

    def add_sample(self, sample: HeadPoseSample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.required_samples:
            self.samples.pop(0)

    @property
    def progress(self) -> float:
        return min(1.0, len(self.samples) / max(1, self.required_samples))

    @property
    def is_ready(self) -> bool:
        return len(self.samples) >= self.required_samples

    def build_reference(self) -> HeadPoseReference:
        if not self.is_ready:
            raise ValueError("Head pose calibration requires more samples.")

        centers = np.array([sample.face_center_norm for sample in self.samples], dtype=np.float64)
        sizes = np.array([sample.face_size_norm for sample in self.samples], dtype=np.float64)
        noses = np.array([sample.nose_norm for sample in self.samples], dtype=np.float64)
        left_eyes = np.array([sample.left_eye_center_norm for sample in self.samples], dtype=np.float64)
        right_eyes = np.array([sample.right_eye_center_norm for sample in self.samples], dtype=np.float64)
        eye_dists = np.array([sample.interocular_distance_norm for sample in self.samples], dtype=np.float64)
        yaws = np.array([sample.yaw for sample in self.samples], dtype=np.float64)
        pitches = np.array([sample.pitch for sample in self.samples], dtype=np.float64)
        rolls = np.array([sample.roll for sample in self.samples], dtype=np.float64)

        return HeadPoseReference(
            face_center_norm=(float(np.median(centers[:, 0])), float(np.median(centers[:, 1]))),
            face_size_norm=(float(np.median(sizes[:, 0])), float(np.median(sizes[:, 1]))),
            nose_norm=(float(np.median(noses[:, 0])), float(np.median(noses[:, 1]))),
            left_eye_center_norm=(float(np.median(left_eyes[:, 0])), float(np.median(left_eyes[:, 1]))),
            right_eye_center_norm=(float(np.median(right_eyes[:, 0])), float(np.median(right_eyes[:, 1]))),
            interocular_distance_norm=float(np.median(eye_dists)),
            yaw=float(np.median(yaws)),
            pitch=float(np.median(pitches)),
            roll=float(np.median(rolls)),
        )


@dataclass
class HeadPoseExtremes:
    center: HeadPoseReference
    left_yaw: float
    right_yaw: float
    up_pitch: float
    down_pitch: float

    def normalize_angles(self, yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
        yaw_left_span = max(1e-4, self.center.yaw - self.left_yaw)
        yaw_right_span = max(1e-4, self.right_yaw - self.center.yaw)
        pitch_up_span = max(1e-4, self.center.pitch - self.up_pitch)
        pitch_down_span = max(1e-4, self.down_pitch - self.center.pitch)

        yaw_norm = (yaw - self.center.yaw) / (yaw_right_span if yaw >= self.center.yaw else yaw_left_span)
        pitch_norm = (pitch - self.center.pitch) / (pitch_down_span if pitch >= self.center.pitch else pitch_up_span)
        roll_norm = (roll - self.center.roll) / max(self.center.angle_tolerance[2], 1e-4)
        return float(np.clip(yaw_norm, -1.5, 1.5)), float(np.clip(pitch_norm, -1.5, 1.5)), float(np.clip(roll_norm, -1.5, 1.5))

    def exceeds_threshold(self, yaw: float, pitch: float, roll: float, threshold: float = 1.15) -> bool:
        yaw_norm, pitch_norm, roll_norm = self.normalize_angles(yaw, pitch, roll)
        return max(abs(yaw_norm), abs(pitch_norm), abs(roll_norm)) > threshold


HEAD_POSE_SEQUENCE: tuple[str, ...] = ("straight", "left", "right", "up", "down")


@dataclass
class HeadPoseSequenceCalibrator:
    required_samples: int = 18
    current_index: int = 0
    samples_by_label: dict[str, list[HeadPoseSample]] = field(default_factory=lambda: {label: [] for label in HEAD_POSE_SEQUENCE})

    def reset(self) -> None:
        self.current_index = 0
        self.samples_by_label = {label: [] for label in HEAD_POSE_SEQUENCE}

    @property
    def current_label(self) -> str:
        return HEAD_POSE_SEQUENCE[min(self.current_index, len(HEAD_POSE_SEQUENCE) - 1)]

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(HEAD_POSE_SEQUENCE)

    def add_sample(self, sample: HeadPoseSample) -> None:
        if self.is_complete:
            return
        label = self.current_label
        bucket = self.samples_by_label[label]
        bucket.append(sample)
        if len(bucket) >= self.required_samples:
            self.current_index += 1

    def progress_text(self) -> str:
        if self.is_complete:
            return "done"
        return f"{self.current_label} {len(self.samples_by_label[self.current_label])}/{self.required_samples}"

    def build_extremes(self) -> HeadPoseExtremes:
        if not self.is_complete:
            raise ValueError("Head pose sequence calibration is incomplete.")
        refs = {}
        for label in HEAD_POSE_SEQUENCE:
            calibrator = HeadPoseCalibrator(required_samples=len(self.samples_by_label[label]))
            calibrator.samples = list(self.samples_by_label[label])
            refs[label] = calibrator.build_reference()
        return HeadPoseExtremes(
            center=refs["straight"],
            left_yaw=refs["left"].yaw,
            right_yaw=refs["right"].yaw,
            up_pitch=refs["up"].pitch,
            down_pitch=refs["down"].pitch,
        )


def default_calibration_targets(grid_size: int = 9) -> Sequence[Point]:
    if grid_size not in (5, 9, 16):
        raise ValueError("grid_size must be 5, 9, or 16")
    if grid_size == 5:
        return [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
    if grid_size == 16:
        coords = [0.08, 0.35, 0.65, 0.92]
        return [(x, y) for y in coords for x in coords]
    return [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    ]


def apply_head_pose_compensation(gaze_norm: Point, transform_matrix: Optional[Iterable[Iterable[float]]], gain: float = 0.08) -> Point:
    """Compensate gaze by coarse yaw/pitch from face transform matrix."""
    if transform_matrix is None:
        return gaze_norm
    mat = np.array(transform_matrix, dtype=np.float64)
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return gaze_norm
    r = mat[:3, :3]
    yaw = float(np.arctan2(r[0, 2], r[2, 2]))
    pitch = float(np.arctan2(-r[1, 2], np.sqrt(r[0, 2] ** 2 + r[2, 2] ** 2)))
    compensated_x = gaze_norm[0] - yaw * gain
    compensated_y = gaze_norm[1] + pitch * gain
    return max(0.0, min(1.0, compensated_x)), max(0.0, min(1.0, compensated_y))


def make_head_pose_sample(obs) -> HeadPoseSample:
    return HeadPoseSample(
        face_center_norm=obs.face_center_norm,
        face_size_norm=obs.face_size_norm,
        nose_norm=obs.nose_norm,
        left_eye_center_norm=obs.left_eye_center_norm,
        right_eye_center_norm=obs.right_eye_center_norm,
        interocular_distance_norm=obs.interocular_distance_norm,
        yaw=obs.yaw,
        pitch=obs.pitch,
        roll=obs.roll,
    )


def normalize_gaze_by_head_pose(
    gaze_norm: Point,
    head_sample: HeadPoseSample,
    head_reference: Optional[HeadPoseReference],
    center_gain: float = 0.45,
    size_gain: float = 0.18,
    nose_gain: float = 0.25,
) -> Point:
    if head_reference is None:
        return gaze_norm

    center_dx = head_sample.face_center_norm[0] - head_reference.face_center_norm[0]
    center_dy = head_sample.face_center_norm[1] - head_reference.face_center_norm[1]
    size_dx = head_sample.face_size_norm[0] - head_reference.face_size_norm[0]
    size_dy = head_sample.face_size_norm[1] - head_reference.face_size_norm[1]
    nose_dx = head_sample.nose_norm[0] - head_reference.nose_norm[0]
    nose_dy = head_sample.nose_norm[1] - head_reference.nose_norm[1]

    normalized_x = gaze_norm[0] - center_dx * center_gain - size_dx * size_gain - nose_dx * nose_gain
    normalized_y = gaze_norm[1] - center_dy * center_gain - size_dy * size_gain - nose_dy * nose_gain
    return max(0.0, min(1.0, normalized_x)), max(0.0, min(1.0, normalized_y))


def build_gaze_feature_vector(
    obs,
    head_sample: HeadPoseSample,
    head_reference: Optional[HeadPoseReference],
    head_extremes: Optional[HeadPoseExtremes],
) -> FeatureVector:
    base_x, base_y = normalize_gaze_by_head_pose(obs.gaze_norm, head_sample, head_reference)
    yaw_norm, pitch_norm, roll_norm = (0.0, 0.0, 0.0)
    if head_extremes is not None:
        yaw_norm, pitch_norm, roll_norm = head_extremes.normalize_angles(obs.yaw, obs.pitch, obs.roll)
    return (
        base_x,
        base_y,
        obs.eye_gaze_norm[0],
        obs.eye_gaze_norm[1],
        obs.left_iris_relative[0],
        obs.left_iris_relative[1],
        obs.right_iris_relative[0],
        obs.right_iris_relative[1],
        yaw_norm,
        pitch_norm,
        roll_norm,
    )
