import unittest

from gaze.interactions import AdaptiveBlinkDetector, DwellClickDetector


class InteractionTests(unittest.TestCase):
    def test_adaptive_blink_click(self):
        detector = AdaptiveBlinkDetector(blink_frames=2, cooldown_s=0.0)
        t = 0.0
        detector.update(0.30, t)
        t += 0.03
        self.assertFalse(detector.update(0.10, t))
        t += 0.03
        self.assertTrue(detector.update(0.10, t))

    def test_dwell_click(self):
        dwell = DwellClickDetector(radius_px=10.0, dwell_s=0.2, cooldown_s=0.0)
        self.assertFalse(dwell.update((100, 100), 0.0))
        self.assertFalse(dwell.update((104, 105), 0.1))
        self.assertTrue(dwell.update((103, 104), 0.25))


if __name__ == "__main__":
    unittest.main()
