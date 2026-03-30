from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class CursorSmoother:
    """One-Euro filter for low-latency, non-overshooting cursor motion."""

    min_cutoff: float = 0.8
    beta: float = 0.01
    d_cutoff: float = 1.0
    speed_scale: float = 1.0
    dead_zone: float = 0.005
    max_cursor_speed: float = 0.08
    _x: Optional[Point] = None
    _dx: Point = (0.0, 0.0)
    _last_time: float = 0.0
    _prev_output: Optional[Point] = None

    def reset(self) -> None:
        self._x = None
        self._dx = (0.0, 0.0)
        self._last_time = 0.0
        self._prev_output = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def _clamp_speed(
        self,
        prev: Point,
        current: Point,
        max_norm_speed: float,
        screen_w: int,
        screen_h: int,
    ) -> Point:
        dx = (current[0] - prev[0]) / max(float(screen_w), 1.0)
        dy = (current[1] - prev[1]) / max(float(screen_h), 1.0)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > max_norm_speed:
            scale = max_norm_speed / dist
            return (
                prev[0] + (current[0] - prev[0]) * scale,
                prev[1] + (current[1] - prev[1]) * scale,
            )
        return current

    def update(self, target: Point, timestamp: float, screen_w: int = 1920, screen_h: int = 1080) -> Point:
        if self._x is None:
            self._x = target
            self._last_time = timestamp
            self._prev_output = target
            return target

        dt = max(timestamp - self._last_time, 1e-6)
        self._last_time = timestamp
        
        raw_dx = ((target[0] - self._x[0]) / dt, (target[1] - self._x[1]) / dt)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (
            a_d * raw_dx[0] + (1.0 - a_d) * self._dx[0],
            a_d * raw_dx[1] + (1.0 - a_d) * self._dx[1],
        )
        self._dx = dx

        speed = math.sqrt(dx[0] ** 2 + dx[1] ** 2)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        filtered = (
            a * target[0] + (1.0 - a) * self._x[0],
            a * target[1] + (1.0 - a) * self._x[1],
        )
        self._x = filtered
        self._prev_output = filtered
        return filtered


@dataclass
class RotationSmoother:
    alpha: float = 0.25
    _R_smooth: Optional[np.ndarray] = None

    def update(self, R: np.ndarray) -> np.ndarray:
        if self._R_smooth is None:
            self._R_smooth = R.copy()
            return R
        # Element-wise EMA
        R_blended = (1 - self.alpha) * self._R_smooth + self.alpha * R
        # Re-orthogonalise to keep it a valid rotation matrix
        U, _, Vt = np.linalg.svd(R_blended)
        self._R_smooth = U @ Vt
        return self._R_smooth

    def reset(self) -> None:
        self._R_smooth = None


def apply_sensitivity(norm_x: float, norm_y: float, sensitivity: float) -> Point:
    """Scale gaze movement around screen center to control sensitivity."""
    sensitivity = max(0.05, min(2.0, sensitivity))
    cx, cy = 0.5, 0.5
    x = (norm_x - cx) * sensitivity + cx
    y = (norm_y - cy) * sensitivity + cy
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def apply_precision_curve(norm_x: float, norm_y: float, deadzone: float, curve_power: float) -> Point:
    """Suppress tiny gaze jitter near center while preserving edge reach."""
    deadzone = max(0.0, min(0.3, deadzone))
    curve_power = max(1.0, min(4.0, curve_power))
    cx, cy = 0.5, 0.5

    def transform(value: float, center: float) -> float:
        offset = value - center
        distance = abs(offset)
        if distance <= deadzone:
            return center

        usable = max(1e-6, 0.5 - deadzone)
        normalized = min(1.0, (distance - deadzone) / usable)
        curved = normalized ** curve_power
        return center + (1.0 if offset >= 0.0 else -1.0) * curved * 0.5

    x = transform(norm_x, cx)
    y = transform(norm_y, cy)
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def map_to_screen(norm_x: float, norm_y: float, screen_w: int, screen_h: int, margin: float) -> Point:
    margin = max(0.0, min(0.3, margin))

    def stretch(v: float) -> float:
        v = (v - margin) / max(1.0 - 2.0 * margin, 0.01)
        if v < 0.0:
            return max(-0.05, v * 0.3)
        if v > 1.0:
            return min(1.05, 1.0 + (v - 1.0) * 0.3)
        return v

    x = max(0.0, min(1.0, stretch(norm_x)))
    y = max(0.0, min(1.0, stretch(norm_y)))
    return x * screen_w, y * screen_h
