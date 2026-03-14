# Eye-Tracker (Gaze + Cursor Control)

A real-time IPCV project for gaze-based mouse control with calibration, head-pose compensation, adaptive blink clicks, dwell clicks, and drag toggle.

## Major upgrades implemented
- **Calibration system (5-point or 9-point):** quadratic regression from gaze features to screen coordinates.
- **Head-pose compensation:** uses MediaPipe facial transform matrix to reduce drift.
- **Adaptive blink detection:** per-user EAR baseline + ratio threshold instead of fixed static threshold.
- **Smoother cursor controller:** exponential smoothing + velocity damping + capped motion step.
- **Interaction layer:**
  - blink click (left/right click mode),
  - dwell click,
  - drag toggle,
  - on-screen keyboard launcher.
- **Packaging/reliability:**
  - modular code split (`gaze/tracker.py`, `gaze/controller.py`, `gaze/calibration.py`, `gaze/interactions.py`),
  - startup checks with actionable errors for OpenCV/libGL/MediaPipe/pyautogui,
  - unit tests for non-camera math and interaction logic.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py --camera 0 --flip --show-debug
```

Safe first run (no real mouse movement):
```bash
python main.py --dry-run --flip --show-debug
```

## Runtime controls
- `q`: quit
- `c`: start calibration
- `SPACE`: capture curren calibration point
- `m`: toggle click mode (left/right)
- `v`: toggle dwell click
- `g`: toggle drag hold
- `k`: launch on-screen keyboard (if available)
m
## Useful flags
- `--calibration-points {5,9}`
- `--pose-comp-gain 0.08`
- `--smoothing-alpha 0.35`
- `--velocity-damping 0.70`
- `--max-step 120`
- `--blink-threshold-ratio 0.70`
- `--baseline-alpha 0.02`
- `--blink-frames 2`
- `--dwell-seconds 1.0`
- `--dwell-radius 45`
- `--model-path models/face_landmarker.task`
- `--model-url <url>`

## Tests
```bash
python -m unittest discover -s tests -v
```
