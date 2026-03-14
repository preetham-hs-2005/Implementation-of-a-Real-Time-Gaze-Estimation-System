from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _poly_features(x: float, y: float) -> np.ndarray:
    return np.array([1.0, x, y, x * x, y * y, x * y], dtype=np.float64)


@dataclass
class CalibrationModel:
    """Quadratic regression mapper from normalized gaze to normalized screen point."""

    coef_x: Optional[np.ndarray] = None
    coef_y: Optional[np.ndarray] = None
    points: List[Tuple[Point, Point]] = field(default_factory=list)

    def add_sample(self, gaze_norm: Point, screen_norm: Point) -> None:
        self.points.append((gaze_norm, screen_norm))

    def clear(self) -> None:
        self.points.clear()
        self.coef_x = None
        self.coef_y = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    def fit(self) -> None:
        if len(self.points) < 5:
            raise ValueError("Need at least 5 samples for calibration fit.")
        features = np.stack([_poly_features(gx, gy) for (gx, gy), _ in self.points], axis=0)
        target_x = np.array([sx for _, (sx, _) in self.points], dtype=np.float64)
        target_y = np.array([sy for _, (_, sy) in self.points], dtype=np.float64)
        self.coef_x, *_ = np.linalg.lstsq(features, target_x, rcond=None)
        self.coef_y, *_ = np.linalg.lstsq(features, target_y, rcond=None)

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
        return [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
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
