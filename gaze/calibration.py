from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import cv2 as _cv2
except ModuleNotFoundError:  # pragma: no cover - optional during unit tests
    _cv2 = None
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

    def fit(self, regularization: float = 1e-3) -> None:
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
class HomographyCalibrationModel:
    """
    Maps 2D gaze angle to normalised screen via perspective homography.
    Falls back to polynomial fit if homography is unavailable.
    """

    H: Optional[np.ndarray] = None
    H_inv: Optional[np.ndarray] = None
    points: list = field(default_factory=list)  # [(angle_x, angle_y, sx, sy), ...]
    fallback: Optional[CalibrationModel] = field(default_factory=CalibrationModel)

    def add_sample(self, angle_x: float, angle_y: float, sx: float, sy: float) -> None:
        self.points.append((float(angle_x), float(angle_y), float(sx), float(sy)))
        if self.fallback is not None:
            self.fallback.add_sample((angle_x, angle_y), (sx, sy))

    def clear(self) -> None:
        self.points.clear()
        self.H = None
        self.H_inv = None
        if self.fallback is not None:
            self.fallback.clear()

    @property
    def is_fitted(self) -> bool:
        return self.H is not None or (self.fallback is not None and self.fallback.is_fitted)

    def fit(self) -> None:
<<<<<<< HEAD
        self.H = None
        self.H_inv = None
=======
        if _cv2 is None:
            if self.fallback is not None:
                print("[CALIB] OpenCV unavailable. Falling back to polynomial fit.")
                self.fallback.fit()
                return
            raise RuntimeError("OpenCV is required for homography calibration.")
>>>>>>> b66aadb8d542aa5d936cd64a2bf6983ce88d561e
        if len(self.points) < 8:
            if self.fallback is not None:
                print("[CALIB] Homography unavailable (<8 points). Falling back to polynomial fit.")
                self.fallback.fit()
                return
            raise ValueError("Homography needs at least 8 point correspondences.")
        src = np.array([[p[0], p[1]] for p in self.points], dtype=np.float32)
        dst = np.array([[p[2], p[3]] for p in self.points], dtype=np.float32)
        H, mask = _cv2.findHomography(src, dst, _cv2.RANSAC, ransacReprojThreshold=0.05)
        if H is None:
            if self.fallback is not None:
                print("[CALIB] Homography fit failed. Falling back to polynomial fit.")
                self.fallback.fit()
                return
            raise ValueError("Homography fit failed - not enough inliers.")
        inlier_count = int(mask.sum()) if mask is not None else len(self.points)
        min_inlier_ratio = 0.5
        if inlier_count / max(len(self.points), 1) < min_inlier_ratio:
            print(
                f"[CALIB] Homography inlier ratio too low ({inlier_count}/{len(self.points)} < 50%). "
                "Falling back to polynomial fit for better accuracy."
            )
            if self.fallback is not None:
                self.fallback.fit()
                return
        self.H = H
        self.H_inv = np.linalg.inv(H)
        print(f"[CALIB] Homography fitted. Inliers: {inlier_count}/{len(self.points)}")

    def map(self, angle_x: float, angle_y: float) -> tuple[float, float]:
        if self.H is None:
            if self.fallback is not None and self.fallback.is_fitted:
                return self.fallback.map((angle_x, angle_y))
            return 0.5, 0.5
        src = np.array([[[angle_x, angle_y]]], dtype=np.float32)
        dst = _cv2.perspectiveTransform(src, self.H)
        sx = float(np.clip(dst[0, 0, 0], -0.1, 1.1))
        sy = float(np.clip(dst[0, 0, 1], -0.1, 1.1))
        return sx, sy

    def reprojection_errors(self) -> list[float]:
        if self.H is None and (self.fallback is None or not self.fallback.is_fitted):
            return []
        errors = []
        for ax, ay, sx, sy in self.points:
            px, py = self.map(ax, ay)
            errors.append(math.sqrt((px - sx) ** 2 + (py - sy) ** 2))
        return errors


@dataclass
class NeutralPoseCapture:
    """
    Captures the head rotation matrix when the user is looking straight ahead.
    This is the only head reference the system needs.
    Auto-captures after 25 stable frames. Can also be triggered with j.
    """
    required_frames: int = 25
    _rotation_matrices: list = field(default_factory=list)
    R_neutral: Optional[np.ndarray] = None

    def add_frame(self, R: np.ndarray, iris_valid: bool, is_fixating: bool) -> bool:
        """Returns True when capture is complete."""
        if self.R_neutral is not None:
            return True
        if not iris_valid or not is_fixating:
            return False
        self._rotation_matrices.append(R.copy())
        if len(self._rotation_matrices) >= self.required_frames:
            # Average rotation matrices and re-orthogonalise with SVD
            mean_R = np.mean(self._rotation_matrices, axis=0)
            U, _, Vt = np.linalg.svd(mean_R)
            self.R_neutral = U @ Vt
            print(f"[HEAD] Neutral rotation captured from {len(self._rotation_matrices)} frames.")
            return True
        return False

    def reset(self) -> None:
        self._rotation_matrices.clear()
        self.R_neutral = None

    @property
    def progress(self) -> float:
        if self.R_neutral is not None:
            return 1.0
        return len(self._rotation_matrices) / self.required_frames

    @property
    def is_ready(self) -> bool:
        return self.R_neutral is not None


def default_calibration_targets(grid_size: int = 9) -> Sequence[Point]:
    if grid_size not in (5, 9, 16):
        raise ValueError("grid_size must be 5, 9, or 16")
    if grid_size == 5:
        return [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
    if grid_size == 16:
        coords = [0.05, 0.32, 0.68, 0.95]
        targets = [(x, y) for y in coords for x in coords]
        targets += [(0.02, 0.02), (0.98, 0.02), (0.02, 0.98), (0.98, 0.98)]
        return targets
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


def project_gaze_to_screen(gaze_dir_world: Sequence[float], screen_z: float = 1.0) -> Point:
    del screen_z
    angle_x, angle_y = gaze_to_screen_angles(gaze_dir_world)
    return angle_x / 0.5236, angle_y / 0.5236


def gaze_to_screen_angles(gaze_world: Sequence[float]) -> Point:
    gx, gy, gz = (float(gaze_world[0]), float(gaze_world[1]), float(gaze_world[2]))
    if gz < 0.0:
        gx, gy, gz = -gx, -gy, -gz
    gz = max(gz, 0.05)
    angle_x = math.atan2(gx, gz)
    angle_y = math.atan2(gy, gz)
    return angle_x, angle_y


@dataclass
class GazeDriftCorrector:
    correction: Point = (0.0, 0.0)
    alpha: float = 0.002
    max_correction: float = 0.08

    def update(self, predicted_norm: Point, stable_anchor_norm: Point) -> Point:
        err_x = stable_anchor_norm[0] - predicted_norm[0]
        err_y = stable_anchor_norm[1] - predicted_norm[1]
        cx = self.correction[0] + self.alpha * err_x
        cy = self.correction[1] + self.alpha * err_y
        self.correction = (
            max(-self.max_correction, min(self.max_correction, cx)),
            max(-self.max_correction, min(self.max_correction, cy)),
        )
        return self.correction

    def apply(self, norm: Point) -> Point:
        return norm[0] + self.correction[0], norm[1] + self.correction[1]





@dataclass
class CalibrationQualityMap:
    point_errors: dict[tuple[float, float], float] = field(default_factory=dict)
    point_sample_counts: dict[tuple[float, float], int] = field(default_factory=dict)
    error_alpha: float = 0.3

    def record_error(self, target: tuple[float, float], predicted) -> None:
        if isinstance(predicted, tuple):
            err = math.sqrt((target[0] - predicted[0]) ** 2 + (target[1] - predicted[1]) ** 2)
        else:
            err = float(predicted)
        if target not in self.point_errors:
            self.point_errors[target] = err
            self.point_sample_counts[target] = 1
        else:
            self.point_errors[target] = (1.0 - self.error_alpha) * self.point_errors[target] + self.error_alpha * err
            self.point_sample_counts[target] += 1

    def worst_points(self, n: int = 5) -> list[tuple[float, float]]:
        return sorted(self.point_errors, key=lambda p: self.point_errors[p], reverse=True)[:n]

    def score(self, point: tuple[float, float]) -> float:
        if not self.point_errors:
            return 0.0
        max_err = max(self.point_errors.values()) or 1.0
        return self.point_errors.get(point, 0.0) / max_err

    def needs_improvement(self, threshold: float = 0.06) -> bool:
        return any(error > threshold for error in self.point_errors.values())

    def clear(self) -> None:
        self.point_errors.clear()
        self.point_sample_counts.clear()


@dataclass
class AdaptiveCalibrationSequencer:
    base_targets: list[tuple[float, float]] = field(default_factory=list)
    quality_map: Optional[CalibrationQualityMap] = None
    refinement_points: int = 8

    def first_pass_sequence(self) -> list[tuple[float, float]]:
        targets = list(self.base_targets)
        center = min(targets, key=lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2)
        targets.remove(center)
        random.shuffle(targets)
        return [center] + targets

    def refinement_sequence(self) -> list[tuple[float, float]]:
        if self.quality_map is None or not self.quality_map.point_errors:
            return self.first_pass_sequence()

        worst = self.quality_map.worst_points(n=self.refinement_points)
        sequence: list[tuple[float, float]] = []
        for point in worst:
            sequence.append(point)
            sequence.append(point)
        for i in range(len(worst) - 1):
            mid = ((worst[i][0] + worst[i + 1][0]) / 2.0, (worst[i][1] + worst[i + 1][1]) / 2.0)
            sequence.append(mid)

        all_sorted = sorted(self.quality_map.point_errors, key=lambda p: self.quality_map.point_errors[p], reverse=True)
        medium = all_sorted[self.refinement_points:]
        sequence.extend(medium[:self.refinement_points])

        if len(sequence) > 1:
            fixed = sequence[0]
            rest = sequence[1:]
            random.shuffle(rest)
            return [fixed] + rest
        return sequence
