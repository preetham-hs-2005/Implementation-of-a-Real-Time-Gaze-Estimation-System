from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class AdaptiveBlinkDetector:
    baseline_alpha: float = 0.02
    threshold_ratio: float = 0.70
    blink_frames: int = 2
    cooldown_s: float = 0.6
    baseline_ear: float = 0.28
    _low_frames: int = 0
    _last_click_ts: float = 0.0

    def update(self, ear: float, now_s: float) -> bool:
        if ear > 0:
            self.baseline_ear = (1 - self.baseline_alpha) * self.baseline_ear + self.baseline_alpha * ear
        threshold = self.baseline_ear * self.threshold_ratio
        if ear < threshold:
            self._low_frames += 1
        else:
            self._low_frames = 0

        if self._low_frames >= self.blink_frames and (now_s - self._last_click_ts) >= self.cooldown_s:
            self._last_click_ts = now_s
            self._low_frames = 0
            return True
        return False


@dataclass
class DwellClickDetector:
    radius_px: float = 45.0
    dwell_s: float = 1.0
    cooldown_s: float = 0.6
    _anchor: tuple[float, float] | None = None
    _anchor_ts: float = 0.0
    _last_click_ts: float = 0.0

    def update(self, pos: tuple[float, float], now_s: float) -> bool:
        if self._anchor is None:
            self._anchor = pos
            self._anchor_ts = now_s
            return False

        dx = pos[0] - self._anchor[0]
        dy = pos[1] - self._anchor[1]
        if (dx * dx + dy * dy) ** 0.5 > self.radius_px:
            self._anchor = pos
            self._anchor_ts = now_s
            return False

        if now_s - self._anchor_ts >= self.dwell_s and now_s - self._last_click_ts >= self.cooldown_s:
            self._last_click_ts = now_s
            self._anchor = pos
            self._anchor_ts = now_s
            return True
        return False


@dataclass
class DragState:
    enabled: bool = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled


@dataclass
class FixationDetector:
    """
    Classifies gaze as fixation / saccade / stabilising from angular velocity.
    """

    saccade_threshold_rad_s: float = 0.35
    stabilisation_frames: int = 3
    history_size: int = 3
    _angle_history: list = field(default_factory=list)
    _saccade_active: bool = False
    _stabilisation_counter: int = 0
    _last_velocity: float = 0.0

    def update(self, angle_x: float, angle_y: float, timestamp: float) -> str:
        self._angle_history.append((angle_x, angle_y, timestamp))
        if len(self._angle_history) > self.history_size:
            self._angle_history.pop(0)

        if len(self._angle_history) < 2:
            return "fixation"

        old_x, old_y, old_t = self._angle_history[0]
        new_x, new_y, new_t = self._angle_history[-1]
        dt = max(new_t - old_t, 1e-6)
        dangle = math.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
        velocity = dangle / dt
        self._last_velocity = velocity

        if velocity > self.saccade_threshold_rad_s:
            self._saccade_active = True
            self._stabilisation_counter = self.stabilisation_frames
            return "saccade"

        if self._stabilisation_counter > 0:
            self._stabilisation_counter -= 1
            return "stabilising"

        self._saccade_active = False
        return "fixation"

    @property
    def is_fixating(self) -> bool:
        return (not self._saccade_active) and self._stabilisation_counter == 0

    @property
    def velocity(self) -> float:
        return self._last_velocity
