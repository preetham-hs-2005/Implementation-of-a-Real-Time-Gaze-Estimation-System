from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _poly_features(x: float, y: float) -> np.ndarray:
    return np.array([1.0, x, y, x * x, y * y, x * y], dtype=np.float64)


def average_points(points: Sequence[Point]) -> Point:
    if not points:
        raise ValueError("Need at least one point to average.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def is_stable_gaze(points: Sequence[Point], max_range: float) -> bool:
    if len(points) < 2:
        return False
    max_range = max(0.0, max_range)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (max(xs) - min(xs) <= max_range) and (max(ys) - min(ys) <= max_range)


@dataclass
class CalibrationModel:
    """Quadratic regression mapper from normalized gaze to normalized screen point."""

    coef_x: Optional[np.ndarray] = None
    coef_y: Optional[np.ndarray] = None
    points: List[Tuple[Point, Point]] = field(default_factory=list)
    interaction_points: List[Tuple[Point, Point]] = field(default_factory=list)
    max_interaction_points: int = 60

    def add_sample(self, gaze_norm: Point, screen_norm: Point, interaction: bool = False) -> None:
        if interaction:
            self.interaction_points.append((gaze_norm, screen_norm))
            self.interaction_points = self.interaction_points[-self.max_interaction_points:]
            return
        self.points.append((gaze_norm, screen_norm))

    def clear(self) -> None:
        self.points.clear()
        self.interaction_points.clear()
        self.coef_x = None
        self.coef_y = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    @property
    def sample_count(self) -> int:
        return len(self.points) + len(self.interaction_points)

    def fit(self, regularization: float = 1e-5) -> None:
        all_points = self.points + self.interaction_points
        if len(all_points) < 5:
            raise ValueError("Need at least 5 samples for calibration fit.")
        features = np.stack([_poly_features(gx, gy) for (gx, gy), _ in all_points], axis=0)
        target_x = np.array([sx for _, (sx, _) in all_points], dtype=np.float64)
        target_y = np.array([sy for _, (_, sy) in all_points], dtype=np.float64)

        # Regularized least squares for more stable calibration, reducing over-sensitive overfitting.
        xtx = features.T @ features
        reg_mat = np.eye(xtx.shape[0], dtype=np.float64) * regularization
        self.coef_x = np.linalg.solve(xtx + reg_mat, features.T @ target_x)
        self.coef_y = np.linalg.solve(xtx + reg_mat, features.T @ target_y)

    def refine_from_interaction(self, gaze_norm: Point, screen_norm: Point, regularization: float = 5e-4) -> bool:
        self.add_sample(gaze_norm, screen_norm, interaction=True)
        if self.sample_count < 5:
            return False
        self.fit(regularization=regularization)
        return True

    def map(self, gaze_norm: Point) -> Point:
        if not self.is_fitted:
            return gaze_norm
        feats = _poly_features(gaze_norm[0], gaze_norm[1])
        out_x = float(feats @ self.coef_x)
        out_y = float(feats @ self.coef_y)
        return max(0.0, min(1.0, out_x)), max(0.0, min(1.0, out_y))


def default_calibration_targets(grid_size: int = 9) -> Sequence[Point]:
    if grid_size not in (5, 9):
        raise ValueError("grid_size must be 5 or 9")
    if grid_size == 5:
        return [(0.08, 0.08), (0.92, 0.08), (0.5, 0.5), (0.08, 0.92), (0.92, 0.92)]
    return [
        (0.08, 0.08), (0.5, 0.08), (0.92, 0.08),
        (0.08, 0.5), (0.5, 0.5), (0.92, 0.5),
        (0.08, 0.92), (0.5, 0.92), (0.92, 0.92),
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
