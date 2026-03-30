"""Persistent storage for personal gaze calibration samples."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

DEFAULT_PATH = Path.home() / ".gaze_mouse_profile.json"
MAX_SAMPLES = 600


class PersonalGazeStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self.samples: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.samples = data.get("samples", [])
                print(f"[PROFILE] Loaded {len(self.samples)} personal gaze samples from {self.path}")
            except Exception:
                self.samples = []

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps({"samples": self.samples[-MAX_SAMPLES:], "updated": time.time()}, indent=2))
        except Exception as exc:
            print(f"[PROFILE] Save failed: {exc}")

    def clear(self) -> None:
        self.samples = []
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as exc:
            print(f"[PROFILE] Clear failed: {exc}")

    def add_session_samples(self, points: list[tuple[float, float, float, float]], weight: float = 1.0) -> None:
        ts = time.time()
        for ax, ay, sx, sy in points:
            self.samples.append({"ax": ax, "ay": ay, "sx": sx, "sy": sy, "ts": ts, "w": weight})
        self.samples = self.samples[-MAX_SAMPLES:]

    def get_weighted_points(self) -> list[tuple[float, float, float, float]]:
        if not self.samples:
            return []
        now = time.time()
        weighted = []
        for sample in self.samples:
            age_hours = (now - sample.get("ts", now)) / 3600.0
            decay = math.exp(-0.14 * age_hours)
            effective_weight = sample.get("w", 1.0) * decay
            if effective_weight > 0.05:
                weighted.append((sample["ax"], sample["ay"], sample["sx"], sample["sy"]))
        return weighted

    def sample_count(self) -> int:
        return len(self.samples)
