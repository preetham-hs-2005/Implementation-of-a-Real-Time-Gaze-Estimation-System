from __future__ import annotations

from dataclasses import dataclass


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
