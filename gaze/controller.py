from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[float, float]


@dataclass
class CursorSmoother:
    alpha: float = 0.35
    velocity_damping: float = 0.70
    max_step: float = 120.0
    _position: Optional[Point] = None
    _velocity: Point = (0.0, 0.0)

    def reset(self) -> None:
        self._position = None
        self._velocity = (0.0, 0.0)

    def update(self, target: Point) -> Point:
        if self._position is None:
            self._position = target
            self._velocity = (0.0, 0.0)
            return target

        dx = target[0] - self._position[0]
        dy = target[1] - self._position[1]
        vx = self._velocity[0] * self.velocity_damping + self.alpha * dx
        vy = self._velocity[1] * self.velocity_damping + self.alpha * dy

        step_mag = (vx * vx + vy * vy) ** 0.5
        if step_mag > self.max_step and step_mag > 0:
            scale = self.max_step / step_mag
            vx *= scale
            vy *= scale

        self._position = (self._position[0] + vx, self._position[1] + vy)
        self._velocity = (vx, vy)
        return self._position


def apply_sensitivity(norm_x: float, norm_y: float, sensitivity: float) -> Point:
    """Scale gaze movement around screen center to control sensitivity."""
    sensitivity = max(0.05, min(2.0, sensitivity))
    cx, cy = 0.5, 0.5
    x = (norm_x - cx) * sensitivity + cx
    y = (norm_y - cy) * sensitivity + cy
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def map_to_screen(norm_x: float, norm_y: float, screen_w: int, screen_h: int, margin: float) -> Point:
    margin = max(0.0, min(0.4, margin))
    usable = max(0.2, 1.0 - 2.0 * margin)
    x = max(0.0, min(1.0, (norm_x - margin) / usable))
    y = max(0.0, min(1.0, (norm_y - margin) / usable))
    return x * screen_w, y * screen_h
