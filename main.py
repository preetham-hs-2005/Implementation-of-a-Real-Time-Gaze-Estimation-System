#!/usr/bin/env python3
"""Real-time gaze-controlled cursor with calibration and advanced interactions."""
from __future__ import annotations

import argparse
from collections import deque
import math
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from gaze.calibration import (
    AdaptiveCalibrationSequencer,
    HEAD_POSE_SEQUENCE,
    CalibrationModel,
    CalibrationQualityMap,
    GazeDriftCorrector,
    HeadPoseCalibrator,
    HeadPoseExtremes,
    HeadPoseReference,
    HeadPoseSequenceCalibrator,
    HomographyCalibrationModel,
    default_calibration_targets,
    fallback_head_pose_extremes,
    gaze_to_screen_angles,
    make_head_pose_sample,
)
from gaze.controller import CursorSmoother, HeadPoseSmoother, map_to_screen, apply_sensitivity
from gaze.gaze_store import PersonalGazeStore
from gaze.interactions import AdaptiveBlinkDetector, DragState, DwellClickDetector, FixationDetector
from gaze.tracker import combine_head_and_eye_gaze, extract_observation


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye-controlled cursor using latest MediaPipe Face Landmarker.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "default", "dshow", "msmf"],
        default="auto",
        help="Camera backend on Windows. 'auto' probes multiple backends and prefers a non-black stream.",
    )
    parser.add_argument("--flip", action="store_true", help="Mirror the frame horizontally.")
    parser.add_argument("--margin", type=float, default=0.05, help="Dead margin from each edge (0-0.4).")
    parser.add_argument("--show-debug", action="store_true", help="Show landmarks and tracking overlays.")
    parser.add_argument("--debug-gaze", action="store_true", help="Print raw gaze-pipeline values every 30 frames.")
    parser.add_argument("--dry-run", action="store_true", help="Do not move/click cursor, only visualize.")

    parser.add_argument("--min-cutoff", type=float, default=0.8, help="One-Euro filter minimum cutoff frequency in Hz.")
    parser.add_argument("--beta", type=float, default=0.01, help="One-Euro filter speed coefficient.")
    parser.add_argument("--d-cutoff", type=float, default=1.0, help="One-Euro derivative cutoff frequency in Hz.")
    parser.add_argument("--max-cursor-speed", type=float, default=0.08, help="Maximum cursor travel per frame as normalized screen fraction.")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Global cursor speed multiplier after filtering.")
    parser.add_argument("--smoothing-alpha", type=float, default=0.18, help="Legacy cursor smoothing arg kept for compatibility.")
    parser.add_argument("--velocity-damping", type=float, default=0.70, help="Legacy cursor smoothing arg kept for compatibility.")
    parser.add_argument("--max-step", type=float, default=120.0, help="Legacy cursor step arg kept for compatibility.")
    parser.add_argument("--dead-zone", type=float, default=0.005, help="Legacy dead-zone arg kept for compatibility.")

    parser.add_argument("--blink-frames", type=int, default=2, help="Consecutive low-EAR frames for click.")
    parser.add_argument("--click-cooldown", type=float, default=0.6, help="Seconds between clicks.")
    parser.add_argument("--blink-threshold-ratio", type=float, default=0.70, help="Adaptive EAR threshold ratio.")
    parser.add_argument("--baseline-alpha", type=float, default=0.02, help="Adaptive baseline update rate.")

    parser.add_argument("--dwell-seconds", type=float, default=1.0, help="Seconds for dwell click trigger.")
    parser.add_argument("--dwell-radius", type=float, default=45.0, help="Movement radius for dwell trigger.")

    parser.add_argument("--calibration-points", type=int, default=16, choices=[5, 9, 16], help="Calibration grid size.")
    parser.add_argument("--pose-comp-gain", type=float, default=0.0, help="Head-pose compensation gain (0.0 to disable, higher for more head-motion effect).")
    parser.add_argument("--disable-pose-comp", action="store_true", help="Disable head-pose compensation entirely.")
    parser.add_argument("--gaze-smoothing-alpha", type=float, default=0.35, help="Gaze low-pass alpha (0.0-1.0), smaller is smoother.")
    parser.add_argument("--sensitivity", type=float, default=1.3, help="Gaze-to-cursor sensitivity scale (0.05-2.0, default 1.3).")
    parser.add_argument("--saccade-threshold", type=float, default=0.35, help="Angular velocity threshold in rad/s for saccade detection.")
    parser.add_argument("--ear-valid-min", type=float, default=0.20, help="Minimum EAR at which iris observations are considered valid.")
    parser.add_argument("--head-calibration-frames", type=int, default=30, help="Stable frames required for head-position calibration.")
    parser.add_argument("--head-pose-calibration-frames", type=int, default=16, help="Frames required for each optional head pose direction.")
    parser.add_argument("--disable-head-pose-calibration", action="store_true", help="Skip directional head pose calibration and use only straight-ahead reference.")
    parser.add_argument("--head-center-tolerance", type=float, default=0.08, help="Allowed normalized head-center drift from calibrated pose.")
    parser.add_argument("--head-size-tolerance", type=float, default=0.16, help="Allowed normalized face-size drift from calibrated pose.")
    parser.add_argument("--recalibration-angle-threshold", type=float, default=1.2, help="Normalized yaw/pitch/roll threshold before suggesting recalibration.")
    parser.add_argument("--calib-frames", type=int, default=30, help="Accepted stable frames required per gaze calibration point.")
    parser.add_argument("--no-personal-model", action="store_true", help="Disable persistent personal gaze model.")
    parser.add_argument("--clear-personal-model", action="store_true", help="Delete saved personal gaze profile and exit.")

    parser.add_argument("--model-path", default="models/face_landmarker.task", help="Path to face_landmarker.task.")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL, help="Model URL for first-run download.")
    return parser.parse_args()


def ensure_model(model_path: Path, model_url: str) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


def import_runtime_dependencies():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV failed to import. If you see libGL errors, install system package 'libgl1' (Debian/Ubuntu) "
            "or run in a desktop environment with OpenCV GUI dependencies."
        ) from exc

    try:
        import mediapipe as mp
    except Exception as exc:
        print(f"MediaPipe import error: {exc}")
        raise RuntimeError("MediaPipe import failed. Run: pip install -r requirements.txt") from exc

    try:
        import pyautogui
    except Exception as exc:
        raise RuntimeError("pyautogui import failed. Run: pip install -r requirements.txt") from exc

    return cv2, mp, pyautogui


def draw_status(cv2, frame, text: str, line: int) -> None:
    cv2.putText(frame, text, (10, 24 + line * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def normalize_key(key: int) -> int:
    if key < 0:
        return key
    key &= 0xFF
    if 65 <= key <= 90:
        key += 32
    return key


def default_head_reference() -> HeadPoseReference:
    return HeadPoseReference(
        face_center_norm=(0.5, 0.5),
        face_size_norm=(0.34, 0.48),
        nose_norm=(0.5, 0.54),
        left_eye_center_norm=(0.43, 0.43),
        right_eye_center_norm=(0.57, 0.43),
        interocular_distance_norm=0.14,
        center_tolerance=(0.22, 0.22),
        size_tolerance=(0.28, 0.32),
        eye_distance_tolerance=0.20,
    )


def draw_head_guidance_overlay(cv2, frame, obs, reference: HeadPoseReference, aligned: bool, calibrating_head: bool) -> None:
    height, width = frame.shape[:2]
    ref_w = int(reference.face_size_norm[0] * width)
    ref_h = int(reference.face_size_norm[1] * height)
    ref_cx = int(reference.face_center_norm[0] * width)
    ref_cy = int(reference.face_center_norm[1] * height)

    x1 = max(0, ref_cx - ref_w // 2)
    y1 = max(0, ref_cy - ref_h // 2)
    x2 = min(width - 1, ref_cx + ref_w // 2)
    y2 = min(height - 1, ref_cy + ref_h // 2)

    color = (0, 200, 255) if calibrating_head else ((0, 220, 0) if aligned else (0, 0, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), color, 1)

    if obs is not None:
        cur_w = int(obs.face_size_norm[0] * width)
        cur_h = int(obs.face_size_norm[1] * height)
        cur_cx = int(obs.face_center_norm[0] * width)
        cur_cy = int(obs.face_center_norm[1] * height)
        cx1 = max(0, cur_cx - cur_w // 2)
        cy1 = max(0, cur_cy - cur_h // 2)
        cx2 = min(width - 1, cur_cx + cur_w // 2)
        cy2 = min(height - 1, cur_cy + cur_h // 2)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1)
        cv2.circle(frame, (cur_cx, cur_cy), 4, color, -1)


def head_alignment_hint(obs, reference: HeadPoseReference) -> str:
    dx = obs.face_center_norm[0] - reference.face_center_norm[0]
    dy = obs.face_center_norm[1] - reference.face_center_norm[1]
    size_dy = obs.face_size_norm[1] - reference.face_size_norm[1]
    hints = []
    if dx < -reference.center_tolerance[0]:
        hints.append("move right")
    elif dx > reference.center_tolerance[0]:
        hints.append("move left")
    if dy < -reference.center_tolerance[1]:
        hints.append("move down")
    elif dy > reference.center_tolerance[1]:
        hints.append("move up")
    if size_dy < -reference.size_tolerance[1]:
        hints.append("move closer")
    elif size_dy > reference.size_tolerance[1]:
        hints.append("move back")
    return ", ".join(hints) if hints else "hold still"


def head_pose_instruction(label: str) -> str:
    mapping = {
        "straight": "Look straight at the screen",
        "left": "Turn your head slightly left",
        "right": "Turn your head slightly right",
        "up": "Tilt your head slightly up",
        "down": "Tilt your head slightly down",
    }
    return mapping.get(label, label)


def head_stability_score(head_sample, reference: HeadPoseReference) -> float:
    center_dx = abs(head_sample.face_center_norm[0] - reference.face_center_norm[0])
    center_dy = abs(head_sample.face_center_norm[1] - reference.face_center_norm[1])
    size_dx = abs(head_sample.face_size_norm[0] - reference.face_size_norm[0])
    size_dy = abs(head_sample.face_size_norm[1] - reference.face_size_norm[1])
    return max(center_dx, center_dy, size_dx, size_dy)


def draw_gaze_indicator(cv2, frame, obs) -> None:
    height, width = frame.shape[:2]
    origin = (int(obs.nose_norm[0] * width), int(obs.nose_norm[1] * height))
    scale = min(width, height) * 0.18
    end = (
        int(origin[0] + obs.gaze_vector[0] * scale),
        int(origin[1] + obs.gaze_vector[1] * scale),
    )
    cv2.arrowedLine(frame, origin, end, (0, 255, 255), 2, tipLength=0.2)


def draw_gaze_debug_overlay(cv2, frame, angle_x: float, angle_y: float, gaze_state: str) -> None:
    h, w = frame.shape[:2]
    raw_x = int(w * (0.5 + angle_x / 0.6))
    raw_y = int(h * (0.5 + angle_y / 0.6))
    raw_x = max(0, min(w - 1, raw_x))
    raw_y = max(0, min(h - 1, raw_y))
    cv2.drawMarker(frame, (raw_x, raw_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"raw gaze ({angle_x:+.2f}, {angle_y:+.2f})",
        (raw_x + 12, raw_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"state: {gaze_state}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 200),
        1,
        cv2.LINE_AA,
    )


def compute_gaze_angles(obs, neutral_R, width: int, height: int) -> tuple[float, float] | None:
    if obs is None or not obs.iris_valid:
        return None
    if neutral_R is None:
        if not getattr(compute_gaze_angles, "_warned_missing_neutral", False):
            print("[WARN] Cannot compute gaze angles yet: neutral head reference is not ready.")
            compute_gaze_angles._warned_missing_neutral = True
        return None
    compute_gaze_angles._warned_missing_neutral = False
    gaze_vec = combine_head_and_eye_gaze(
        obs.left_iris_px,
        obs.right_iris_px,
        obs.head_rotation_matrix,
        neutral_R,
        width,
        height,
    )
    return gaze_to_screen_angles(gaze_vec)


def draw_calibration_target(
    cv2,
    frame_or_calib,
    tx: float,
    ty: float,
    screen_w: int,
    screen_h: int,
    collecting: bool,
    samples_collected: int,
    frames_to_collect: int,
    quality_score: float = 0.0,
    is_refinement: bool = False,
) -> None:
    x, y = int(tx * screen_w), int(ty * screen_h)
    outer_r = 28 if not collecting else max(12, 28 - int(samples_collected / max(frames_to_collect, 1) * 16))

    if is_refinement:
        if quality_score > 0.7:
            ring_color = (0, 60, 220)
        elif quality_score > 0.35:
            ring_color = (0, 180, 220)
        else:
            ring_color = (0, 220, 80)
    else:
        ring_color = (0, 220, 220)

    cv2.circle(frame_or_calib, (x, y), outer_r, ring_color, 2)
    if collecting and samples_collected > 0:
        progress = samples_collected / max(frames_to_collect, 1)
        axes = (outer_r - 3, outer_r - 3)
        end_angle = int(-90 + 360 * progress)
        cv2.ellipse(frame_or_calib, (x, y), axes, 0, -90, end_angle, ring_color, 3)

    cv2.circle(frame_or_calib, (x, y), 6, (255, 255, 255), -1)
    cv2.circle(frame_or_calib, (x, y), 3, ring_color, -1)

    if is_refinement and quality_score > 0.0:
        label = f"err {quality_score:.2f}"
        cv2.putText(frame_or_calib, label, (x + 16, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ring_color, 1, cv2.LINE_AA)


def draw_quality_heatmap(cv2, screen_w: int, screen_h: int, quality_map: CalibrationQualityMap):
    overlay = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if not quality_map.point_errors:
        return overlay

    for point in quality_map.point_errors:
        score = quality_map.score(point)
        x, y = int(point[0] * screen_w), int(point[1] * screen_h)
        r = int(10 + score * 40)
        if score > 0.6:
            color = (0, 60, 220)
        elif score > 0.3:
            color = (0, 180, 220)
        else:
            color = (0, 200, 80)
        cv2.circle(overlay, (x, y), r, color, -1)
        cv2.circle(overlay, (x, y), r + 2, (200, 200, 200), 1)
        err_val = quality_map.point_errors[point]
        cv2.putText(overlay, f"{err_val:.3f}", (x - 18, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        overlay,
        "Calibration quality map - red = needs improvement",
        (screen_w // 2 - 320, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    if quality_map.needs_improvement():
        cv2.putText(
            overlay,
            "Press c or r to recalibrate weak areas",
            (screen_w // 2 - 230, screen_h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 220),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            overlay,
            "Calibration looks good",
            (screen_w // 2 - 160, screen_h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 80),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_control_buttons(cv2, frame, gaze_control_enabled: bool) -> dict[str, tuple[int, int, int, int]]:
    buttons = [
        ("toggle", f"Gaze {'ON' if gaze_control_enabled else 'OFF'}"),
        ("calibrate", "Start Calib"),
        ("reset_head", "Reset Head"),
        ("capture_head", "Capture Head"),
    ]
    height, width = frame.shape[:2]
    x = width - 170
    y = 16
    button_h = 34
    rects: dict[str, tuple[int, int, int, int]] = {}
    for action, label in buttons:
        color = (0, 180, 0) if action == "toggle" and gaze_control_enabled else (70, 70, 70)
        if action == "toggle" and not gaze_control_enabled:
            color = (0, 0, 180)
        cv2.rectangle(frame, (x, y), (x + 150, y + button_h), color, -1)
        cv2.rectangle(frame, (x, y), (x + 150, y + button_h), (255, 255, 255), 1)
        cv2.putText(frame, label, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        rects[action] = (x, y, x + 150, y + button_h)
        y += button_h + 8
    return rects


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _backend_candidates(cv2, backend_name: str) -> list[tuple[str, Optional[int]]]:
    if not platform.system().lower().startswith("win"):
        return [("default", None)]

    backend_map = {
        "default": [("default", None)],
        "dshow": [("dshow", cv2.CAP_DSHOW)],
        "msmf": [("msmf", cv2.CAP_MSMF)],
        # Windows cameras can behave differently across backends.
        # Prefer generic auto-selection first, then common explicit backends.
        "auto": [("default", None), ("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)],
    }
    return backend_map[backend_name]


def _probe_capture_stream(cv2, cap, num_frames: int = 3) -> tuple[bool, float]:
    max_mean = 0.0
    any_frame = False
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        any_frame = True
        max_mean = max(max_mean, float(frame.mean()))
    return any_frame, max_mean


def _try_open_camera(cv2, camera_index: int, backend_name: str, accept_dark_stream: bool = False):
    attempts: list[str] = []
    best_dark = None
    for backend_label, backend_flag in _backend_candidates(cv2, backend_name):
        cap = cv2.VideoCapture(camera_index) if backend_flag is None else cv2.VideoCapture(camera_index, backend_flag)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            attempts.append(f"{backend_label}: could not open")
            cap.release()
            continue

        any_frame, max_mean = _probe_capture_stream(cv2, cap)
        if any_frame and (accept_dark_stream or max_mean > 3.0):
            print(f"[INFO] Camera opened with backend '{backend_label}' on index {camera_index} (brightness {max_mean:.1f}).")
            return cap, attempts

        attempts.append(
            f"{backend_label}: {'no frames' if not any_frame else f'frames too dark (brightness {max_mean:.1f})'}"
        )
        if any_frame and (best_dark is None or max_mean > best_dark[0]):
            best_dark = (max_mean, backend_label, cap)
        else:
            cap.release()

    if best_dark is not None:
        brightness, backend_label, cap = best_dark
        print(
            f"[WARN] Using dark camera stream from index {camera_index} with backend '{backend_label}' "
            f"(brightness {brightness:.1f}) because no brighter stream was found."
        )
        return cap, attempts

    return None, attempts


def open_camera(cv2, camera_index: int, backend_name: str):
    if platform.system().lower().startswith("win") and backend_name == "default":
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.isOpened():
            print(f"[INFO] Camera opened with backend 'default' on index {camera_index}.")
            return cap
        cap.release()

    cap, attempts = _try_open_camera(cv2, camera_index, backend_name, accept_dark_stream=True)
    if cap is not None:
        return cap

    if platform.system().lower().startswith("win") and backend_name in {"auto", "default"}:
        print(f"[WARN] Requested camera index {camera_index} was unusable. Scanning nearby indices...")
        for fallback_index in range(6):
            if fallback_index == camera_index:
                continue
            cap, fallback_attempts = _try_open_camera(cv2, fallback_index, "auto", accept_dark_stream=False)
            attempts.extend([f"index {fallback_index} {item}" for item in fallback_attempts])
            if cap is not None:
                print(f"[INFO] Falling back to camera index {fallback_index}.")
                return cap

    attempted = "; ".join(attempts) if attempts else "no backends attempted"
    raise RuntimeError(
        f"Unable to get a usable camera stream from index {camera_index}. Attempts: {attempted}. "
        "Try --camera-backend default or a different --camera index."
    )


def launch_on_screen_keyboard() -> str:
    system = platform.system().lower()
    try:
        if "windows" in system:
            subprocess.Popen(["osk"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "On-screen keyboard launched"
        if "linux" in system:
            for cmd in (["onboard"], ["florence"], ["matchbox-keyboard"]):
                try:
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return f"Launched {' '.join(cmd)}"
                except FileNotFoundError:
                    continue
            return "No on-screen keyboard app found (tried onboard/florence/matchbox-keyboard)"
        if "darwin" in system:
            return "Enable macOS Accessibility Keyboard manually (System Settings > Accessibility > Keyboard)."
    except Exception:
        return "Failed to launch on-screen keyboard"
    return "Unsupported platform for automatic on-screen keyboard launch"


def main() -> None:
    args = parse_args()
    store = PersonalGazeStore()
    if args.clear_personal_model:
        store.clear()
        print("[PROFILE] Personal gaze model cleared.")
        return
    model_path = ensure_model(Path(args.model_path), args.model_url)
    cv2, mp, pyautogui = import_runtime_dependencies()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    print("[INFO] Opening camera, this may take 1-3 seconds...")
    cap = open_camera(cv2, args.camera, args.camera_backend)

    print("[INFO] Camera opened, initializing model...")

    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()

    calibration = HomographyCalibrationModel()
    quality_map = CalibrationQualityMap()
    sequencer = AdaptiveCalibrationSequencer(base_targets=list(default_calibration_targets(args.calibration_points)))
    calibration_targets = sequencer.first_pass_sequence()
    is_first_calibration = True
    calibrating = False
    calibration_index = 0
    collecting = False
    samples_collected = 0
    FRAMES_TO_COLLECT = args.calib_frames
    head_calibrator = HeadPoseCalibrator(required_samples=args.head_calibration_frames)
    head_pose_sequence = HeadPoseSequenceCalibrator(required_samples=args.head_pose_calibration_frames)
    head_reference: Optional[HeadPoseReference] = None
    head_extremes: Optional[HeadPoseExtremes] = None
    head_guidance_reference = default_head_reference()
    drift = GazeDriftCorrector()

    cursor = CursorSmoother(
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        d_cutoff=args.d_cutoff,
        speed_scale=args.speed_scale,
        max_cursor_speed=args.max_cursor_speed,
        dead_zone=args.dead_zone,
    )
    head_pose_smoother = HeadPoseSmoother(alpha=0.25)
    prev_gaze_angles = None
    calibration_status_message = ""
    calibration_anchor_feature: Optional[tuple[float, float]] = None
    calibration_stability_buffer: deque[float] = deque(maxlen=6)
    fixation = FixationDetector(saccade_threshold_rad_s=args.saccade_threshold, stabilisation_frames=3)
    blink = AdaptiveBlinkDetector(
        baseline_alpha=args.baseline_alpha,
        threshold_ratio=args.blink_threshold_ratio,
        blink_frames=args.blink_frames,
        cooldown_s=args.click_cooldown,
    )
    dwell = DwellClickDetector(radius_px=args.dwell_radius, dwell_s=args.dwell_seconds, cooldown_s=args.click_cooldown)
    drag = DragState(False)

    click_mode = "left"
    dwell_enabled = False
    gaze_control_enabled = True
    osk_message = ""

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=False,
    )

    fps_timer = time.time()
    frame_counter = 0
    fps = 0.0

    print("[INFO] Creating FaceLandmarker model (this may take several seconds)...")
    try:
        with FaceLandmarker.create_from_options(options) as face_landmarker:
            print("[INFO] FaceLandmarker model is ready. Starting capture loop.")
            window_name = "Gaze Cursor Control"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 720)
            if platform.system().lower().startswith("win"):
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            pending_action: Optional[str] = None

            def on_mouse(_event, x, y, _flags, _param):
                nonlocal pending_action
                if _event == cv2.EVENT_LBUTTONDOWN:
                    pending_action = f"click:{x}:{y}"

            cv2.setMouseCallback(window_name, on_mouse)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera")
                    break
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"[INFO] Processing frame {frame_count}")
                if args.flip:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    result = face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                except Exception as e:
                    print(f"[ERROR] Face detection failed: {e}")
                    result = None

                status = 0
                obs = None
                head_sample = None
                smoothed_angles: Optional[tuple[float, float]] = None
                mapped_norm: Optional[tuple[float, float]] = None
                gaze_state = "fixation"
                active_head_reference = head_reference or head_guidance_reference

                if result and result.face_landmarks:
                    transform_matrix: Optional[list[list[float]]] = None
                    if result.facial_transformation_matrixes:
                        transform_matrix = result.facial_transformation_matrixes[0]

                    neutral_rotation = head_reference.neutral_rotation_matrix if head_reference is not None else None
                    obs = extract_observation(
                        result.face_landmarks[0],
                        transform_matrix,
                        width,
                        height,
                        neutral_rotation=neutral_rotation,
                    )
                    (smoothed_yaw, smoothed_pitch, smoothed_roll), smoothed_rotation_matrix = head_pose_smoother.update(
                        obs.yaw,
                        obs.pitch,
                        obs.roll,
                        obs.head_rotation_matrix,
                    )
                    obs.yaw = smoothed_yaw
                    obs.pitch = smoothed_pitch
                    obs.roll = smoothed_roll
                    obs.head_rotation_matrix = smoothed_rotation_matrix
                    obs.gaze_vector = combine_head_and_eye_gaze(
                        obs.left_iris_px,
                        obs.right_iris_px,
                        smoothed_rotation_matrix,
                        neutral_rotation=neutral_rotation,
                        width=width,
                        height=height,
                        left_iris_relative=obs.left_iris_relative,
                        right_iris_relative=obs.right_iris_relative,
                    )
                    head_sample = make_head_pose_sample(obs)
                    raw_angles = compute_gaze_angles(obs, neutral_rotation, width, height)
                    if raw_angles is not None:
                        if prev_gaze_angles is None:
                            smoothed_angles = raw_angles
                        else:
                            a = max(0.0, min(1.0, args.gaze_smoothing_alpha))
                            smoothed_angles = (
                                prev_gaze_angles[0] * (1.0 - a) + raw_angles[0] * a,
                                prev_gaze_angles[1] * (1.0 - a) + raw_angles[1] * a,
                            )
                        prev_gaze_angles = smoothed_angles
                    draw_head_guidance_overlay(
                        cv2,
                        frame,
                        obs,
                        active_head_reference,
                        aligned=True,
                        calibrating_head=head_reference is None,
                    )

                    if head_reference is None:
                        draw_status(cv2, frame, "Head reference: hold still for auto-capture or press j", status)
                        status += 1
                        if obs.iris_visibility >= 0.5:
                            head_calibrator.add_sample(head_sample)
                            draw_status(
                                cv2,
                                frame,
                                f"Capturing head reference... {int(head_calibrator.progress * 100)}%",
                                status,
                            )
                            status += 1
                            if head_calibrator.is_ready:
                                head_reference = head_calibrator.build_reference()
                                head_reference.center_tolerance = (
                                    args.head_center_tolerance,
                                    args.head_center_tolerance,
                                )
                                head_reference.size_tolerance = (
                                    args.head_size_tolerance,
                                    args.head_size_tolerance * 1.2,
                                )
                                head_guidance_reference = head_reference
                                active_head_reference = head_reference
                                head_extremes = fallback_head_pose_extremes(head_reference) if args.disable_head_pose_calibration else head_extremes
                                if args.disable_head_pose_calibration:
                                    osk_message = "Head calibration complete. Press c to calibrate gaze."
                                else:
                                    osk_message = "Straight pose captured. Follow pose prompts to improve tilt/turn robustness."
                        else:
                            draw_status(cv2, frame, "Keep both eyes visible for auto head capture", status)
                            status += 1
                    else:
                        if head_extremes is None and not args.disable_head_pose_calibration:
                            draw_status(
                                cv2,
                                frame,
                                f"Head pose calibration: {head_pose_instruction(head_pose_sequence.current_label)}",
                                status,
                            )
                            status += 1
                            if obs.iris_visibility >= 0.5:
                                head_pose_sequence.add_sample(head_sample)
                                draw_status(
                                    cv2,
                                    frame,
                                    f"Pose samples: {head_pose_sequence.progress_text()}",
                                    status,
                                )
                                status += 1
                                if head_pose_sequence.is_complete:
                                    head_extremes = head_pose_sequence.build_extremes()
                                    head_reference = head_extremes.center
                                    head_reference.center_tolerance = (
                                        args.head_center_tolerance,
                                        args.head_center_tolerance,
                                    )
                                    head_reference.size_tolerance = (
                                        args.head_size_tolerance,
                                        args.head_size_tolerance * 1.2,
                                    )
                                    head_guidance_reference = head_reference
                                    osk_message = "Head pose calibration complete. Press c to calibrate gaze."
                            else:
                                draw_status(cv2, frame, "Hold pose and keep eyes visible", status)
                                status += 1
                        else:
                            draw_status(cv2, frame, "Head reference ready", status)
                            status += 1

                        if head_extremes is not None:
                            draw_status(
                                cv2,
                                frame,
                                f"Pose yaw/pitch/roll: {obs.yaw:+.2f} {obs.pitch:+.2f} {obs.roll:+.2f}",
                                status,
                            )
                            status += 1
                            if head_extremes.exceeds_threshold(obs.yaw, obs.pitch, obs.roll, args.recalibration_angle_threshold):
                                draw_status(cv2, frame, "Large head rotation detected - consider recalibration", status)
                                status += 1

                    if calibrating:
                        import numpy as np

                        calib_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                        tx, ty = calibration_targets[calibration_index]
                        target_x, target_y = int(tx * screen_w), int(ty * screen_h)
                        draw_calibration_target(
                            cv2,
                            calib_frame,
                            tx,
                            ty,
                            screen_w,
                            screen_h,
                            collecting,
                            samples_collected,
                            FRAMES_TO_COLLECT,
                            quality_score=quality_map.score(calibration_targets[calibration_index]),
                            is_refinement=not is_first_calibration,
                        )
                        msg = (
                            f"Capturing... {samples_collected}/{FRAMES_TO_COLLECT}"
                            if collecting
                            else "Look at the dot, hold gaze steady, then press SPACE"
                        )
                        cv2.putText(
                            calib_frame,
                            msg,
                            (screen_w // 2 - 320, screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if collecting and calibration_status_message:
                            cv2.putText(
                                calib_frame,
                                calibration_status_message,
                                (screen_w // 2 - 320, (screen_h // 2) + 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (180, 255, 180),
                                2,
                                cv2.LINE_AA,
                            )
                        if collecting and calibration_anchor_feature is not None and prev_gaze_angles is not None:
                            dx = prev_gaze_angles[0] - calibration_anchor_feature[0]
                            dy = prev_gaze_angles[1] - calibration_anchor_feature[1]
                            helper_scale = min(screen_w, screen_h) * 3.5
                            helper_x = int(np.clip(target_x + dx * helper_scale, 0, screen_w - 1))
                            helper_y = int(np.clip(target_y + dy * helper_scale, 0, screen_h - 1))
                            cv2.circle(calib_frame, (helper_x, helper_y), 12, (255, 220, 0), -1)
                            cv2.circle(calib_frame, (helper_x, helper_y), 18, (255, 255, 255), 2)

                            hint = []
                            if helper_x < target_x - 25:
                                hint.append("look slightly right")
                            elif helper_x > target_x + 25:
                                hint.append("look slightly left")
                            if helper_y < target_y - 25:
                                hint.append("look slightly down")
                            elif helper_y > target_y + 25:
                                hint.append("look slightly up")
                            if not hint:
                                hint_text = "steady"
                            else:
                                hint_text = ", ".join(hint)
                            cv2.putText(
                                calib_frame,
                                f"Helper: {hint_text}",
                                (screen_w // 2 - 180, (screen_h // 2) + 95),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (200, 255, 200),
                                2,
                                cv2.LINE_AA,
                            )
                        cv2.imshow("Full Screen Calibration", calib_frame)
                        draw_status(cv2, frame, f"Gaze calibration point {calibration_index + 1}/{len(calibration_targets)}", status)
                        status += 1

                    if obs.iris_visibility < 0.5:
                        draw_status(cv2, frame, "Iris not visible clearly - adjust lighting/position", status)
                        status += 1
                        cursor.reset()
                    elif not obs.iris_valid or obs.iris_weight < 0.3:
                        draw_status(cv2, frame, f"Iris unreliable ({obs.iris_weight:.2f}) - open eyes wider", status)
                        status += 1
                        cursor.reset()
                    elif smoothed_angles is None:
                        draw_status(cv2, frame, "Head reference not ready for gaze angles", status)
                        status += 1
                        cursor.reset()
                    elif calibration.is_fitted and not calibrating:
                        angle_x_smoothed, angle_y_smoothed = smoothed_angles
                        gaze_state = fixation.update(angle_x_smoothed, angle_y_smoothed, time.time())
                        mapped_norm = calibration.map(angle_x_smoothed, angle_y_smoothed)
                        mapped_norm = drift.apply(mapped_norm)
                        adjusted_norm = apply_sensitivity(mapped_norm[0], mapped_norm[1], args.sensitivity)
                        target_cursor = map_to_screen(adjusted_norm[0], adjusted_norm[1], screen_w, screen_h, args.margin)
                        # The FixationDetector tracks the state for logging,
                        # but we let the One-Euro filter handle all the smoothing.
                        # It naturally allows fast jump during saccade and heavy filtering during fixation.
                        smooth_cursor = cursor.update(target_cursor, time.time(), screen_w=screen_w, screen_h=screen_h)
                        if gaze_control_enabled and not args.dry_run:
                            pyautogui.moveTo(smooth_cursor[0], smooth_cursor[1], _pause=False)

                        if gaze_control_enabled and blink.update(obs.ear, time.time()):
                            if not args.dry_run:
                                if click_mode == "left":
                                    pyautogui.click(button="left")
                                else:
                                    pyautogui.click(button="right")
                            draw_status(cv2, frame, f"Blink {click_mode} click", status)
                            status += 1

                        if gaze_control_enabled and dwell_enabled and dwell.update(smooth_cursor, time.time()):
                            stable_anchor_norm = (
                                smooth_cursor[0] / max(float(screen_w), 1.0),
                                smooth_cursor[1] / max(float(screen_h), 1.0),
                            )
                            drift.update(mapped_norm, stable_anchor_norm)
                            nearest = min(
                                calibration_targets,
                                key=lambda p: (p[0] - stable_anchor_norm[0]) ** 2 + (p[1] - stable_anchor_norm[1]) ** 2,
                            )
                            quality_map.record_error(nearest, mapped_norm)
                            if not args.dry_run:
                                pyautogui.click(button=click_mode)
                            draw_status(cv2, frame, f"Dwell {click_mode} click", status)
                            status += 1

                        draw_status(cv2, frame, f"Cursor: ({int(smooth_cursor[0])}, {int(smooth_cursor[1])})", status)
                        status += 1
                        draw_status(cv2, frame, f"EAR: {obs.ear:.3f} baseline: {blink.baseline_ear:.3f}", status)
                        status += 1
                        draw_status(cv2, frame, f"Gaze: {gaze_state} vel {fixation.velocity:.2f}", status)
                        status += 1

                    draw_status(cv2, frame, f"Iris visibility: {obs.iris_visibility:.2f}", status)
                    status += 1

                    if args.show_debug:
                        cv2.circle(frame, (int(obs.left_iris_px[0]), int(obs.left_iris_px[1])), 4, (255, 0, 0), -1)
                        cv2.circle(frame, (int(obs.right_iris_px[0]), int(obs.right_iris_px[1])), 4, (0, 0, 255), -1)
                        draw_gaze_indicator(cv2, frame, obs)
                        if not calibration.is_fitted and smoothed_angles is not None:
                            draw_gaze_debug_overlay(cv2, frame, smoothed_angles[0], smoothed_angles[1], gaze_state)
                    if args.debug_gaze and obs is not None and frame_count % 30 == 0:
                        print(f"[DEBUG] left_iris_rel:  {obs.left_iris_relative}")
                        print(f"[DEBUG] right_iris_rel: {obs.right_iris_relative}")
                        print(f"[DEBUG] eye_gaze_norm:  {obs.eye_gaze_norm}")
                        print(f"[DEBUG] gaze_vector:    {obs.gaze_vector}")
                        print(f"[DEBUG] yaw/pitch/roll: {obs.yaw:.3f} {obs.pitch:.3f} {obs.roll:.3f}")
                        if smoothed_angles is not None:
                            print(f"[DEBUG] angle_x/y:      {smoothed_angles[0]:.3f} {smoothed_angles[1]:.3f}")
                        else:
                            print("[DEBUG] angle_x/y:      None")
                        print(f"[DEBUG] mapped_norm:    {mapped_norm}")
                        print("[DEBUG] ---")
                else:
                    cursor.reset()
                    draw_head_guidance_overlay(
                        cv2,
                        frame,
                        None,
                        active_head_reference,
                        aligned=False,
                        calibrating_head=head_reference is None,
                    )
                    draw_status(cv2, frame, "Face not detected", status)
                    status += 1

                frame_counter += 1
                if frame_counter >= 10:
                    now = time.time()
                    fps = frame_counter / max(now - fps_timer, 1e-6)
                    fps_timer = now
                    frame_counter = 0

                draw_status(cv2, frame, f"FPS: {fps:.1f}", status)
                status += 1
                draw_status(cv2, frame, f"Mode: {click_mode} | Dwell: {dwell_enabled} | Drag: {drag.enabled}", status)
                status += 1
                draw_status(cv2, frame, f"Gaze Control: {'ON' if gaze_control_enabled else 'OFF'}", status)
                status += 1
                calibration_status = "ready" if calibration.is_fitted else ("capturing" if calibrating else "pending")
                head_status = "ready" if head_reference is not None else "capturing"
                draw_status(cv2, frame, f"Head: {head_status} | Gaze calibration: {calibration_status}", status)
                status += 1
                draw_status(cv2, frame, f"Profile: {store.sample_count()} samples", status)
                status += 1
                if osk_message:
                    draw_status(cv2, frame, osk_message, status)
                    status += 1
                draw_status(cv2, frame, "q quit | t on/off | j capture-head | h reset-head | c gaze-calib | r refine", status)
                button_rects = draw_control_buttons(cv2, frame, gaze_control_enabled)

                cv2.imshow(window_name, frame)
                if frame_count == 1:
                    print("[INFO] Window should be visible now. Press 'q' to quit.")
                key = normalize_key(cv2.waitKeyEx(10))
                if pending_action and pending_action.startswith("click:"):
                    _, sx, sy = pending_action.split(":")
                    click_point = (int(sx), int(sy))
                    pending_action = None
                    for action_name, rect in button_rects.items():
                        if point_in_rect(click_point, rect):
                            if action_name == "toggle":
                                key = ord("t")
                            elif action_name == "calibrate":
                                key = ord("c")
                            elif action_name == "reset_head":
                                key = ord("h")
                            elif action_name == "capture_head":
                                key = ord("j")
                            break

                if key == ord("q"):
                    break
                if key == ord("t"):
                    gaze_control_enabled = not gaze_control_enabled
                    cursor.reset()
                    prev_gaze_angles = None
                    if drag.enabled and not args.dry_run:
                        pyautogui.mouseUp(button="left")
                        drag.enabled = False
                    osk_message = f"Gaze control {'enabled' if gaze_control_enabled else 'disabled'}"
                if key == ord("h"):
                    head_reference = None
                    head_extremes = None
                    head_calibrator.reset()
                    head_pose_sequence.reset()
                    calibration.clear()
                    calibrating = False
                    collecting = False
                    samples_collected = 0
                    prev_gaze_angles = None
                    calibration_status_message = ""
                    calibration_anchor_feature = None
                    calibration_stability_buffer.clear()
                    drift = GazeDriftCorrector()
                    quality_map.clear()
                    calibration_targets = sequencer.first_pass_sequence()
                    is_first_calibration = True
                    head_pose_smoother.reset()
                    fixation = FixationDetector(saccade_threshold_rad_s=args.saccade_threshold, stabilisation_frames=3)
                    try:
                        cv2.destroyWindow("Full Screen Calibration")
                    except Exception:
                        pass
                    osk_message = "Head reference reset. Align with guide to recalibrate."
                if key == ord("j") and head_sample is not None and head_reference is None:
                    head_calibrator.add_sample(head_sample)
                    if head_calibrator.is_ready:
                        head_reference = head_calibrator.build_reference()
                        head_reference.center_tolerance = (
                            args.head_center_tolerance,
                            args.head_center_tolerance,
                        )
                        head_reference.size_tolerance = (
                            args.head_size_tolerance,
                            args.head_size_tolerance * 1.2,
                        )
                        head_guidance_reference = head_reference
                        head_extremes = fallback_head_pose_extremes(head_reference)
                        osk_message = "Straight head captured. Press c to start gaze calibration."
                    else:
                        osk_message = f"Capturing straight head... {int(head_calibrator.progress * 100)}%"
                if key == ord("m"):
                    click_mode = "right" if click_mode == "left" else "left"
                if key == ord("v"):
                    dwell_enabled = not dwell_enabled
                if key == ord("g"):
                    if drag.toggle() and not args.dry_run:
                        pyautogui.mouseDown(button="left")
                    elif not drag.enabled and not args.dry_run:
                        pyautogui.mouseUp(button="left")
                if key == ord("k"):
                    osk_message = launch_on_screen_keyboard()

                if key == ord("c"):
                    neutral_rotation = head_reference.neutral_rotation_matrix if head_reference is not None else None
                    if head_reference is None:
                        osk_message = "Waiting for auto head reference capture. Hold still or press j."
                    elif neutral_rotation is None:
                        osk_message = "Cannot calibrate: head reference not ready. Wait or press j."
                    else:
                        if head_extremes is None:
                            head_extremes = fallback_head_pose_extremes(head_reference)
                            if not args.disable_head_pose_calibration and not head_pose_sequence.is_complete:
                                osk_message = "Using straight head reference only. You can press h later to redo full pose calibration."
                        calibration_targets = sequencer.first_pass_sequence() if is_first_calibration else sequencer.refinement_sequence()
                        calibration.clear()
                        calibrating = True
                        calibration_index = 0
                        collecting = False
                        samples_collected = 0
                        prev_gaze_angles = None
                        calibration_status_message = ""
                        calibration_anchor_feature = None
                        calibration_stability_buffer.clear()
                        cv2.namedWindow("Full Screen Calibration", cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_TOPMOST, 1)
                if key == ord("r") and calibration.is_fitted:
                    sequencer.quality_map = quality_map
                    calibration_targets = sequencer.refinement_sequence()
                    calibration.clear()
                    calibrating = True
                    is_first_calibration = False
                    calibration_index = 0
                    collecting = False
                    samples_collected = 0
                    prev_gaze_angles = None
                    calibration_status_message = ""
                    calibration_anchor_feature = None
                    calibration_stability_buffer.clear()
                    osk_message = f"Refinement calibration: {len(calibration_targets)} points targeting weak areas"
                    cv2.namedWindow("Full Screen Calibration", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_TOPMOST, 1)

                if calibrating:
                    if key == ord(" "):
                        collecting = True
                        samples_collected = 0
                        calibration_status_message = "Capturing samples..."
                        calibration_anchor_feature = None
                        calibration_stability_buffer.clear()
                    
                    if collecting and obs is not None and head_sample is not None and head_reference is not None:
                        if smoothed_angles is None:
                            calibration_status_message = "Head reference not ready for calibration"
                            continue
                        if calibration_anchor_feature is None:
                            calibration_anchor_feature = smoothed_angles

                        gaze_state = fixation.update(smoothed_angles[0], smoothed_angles[1], time.time())
                        if gaze_state != "fixation":
                            calibration_status_message = f"Wait for fixation... ({gaze_state})"
                            continue
                        if not obs.iris_valid or obs.iris_weight < 0.3:
                            calibration_status_message = "Open eyes wider - iris not reliable"
                            continue

                        calibration_stability_buffer.append(smoothed_angles[0])
                        if len(calibration_stability_buffer) == calibration_stability_buffer.maxlen:
                            std_x = float(np.std(list(calibration_stability_buffer)))
                            if std_x < 0.015:
                                calibration.add_sample(
                                    smoothed_angles[0],
                                    smoothed_angles[1],
                                    calibration_targets[calibration_index][0],
                                    calibration_targets[calibration_index][1],
                                )
                                samples_collected += 1
                                calibration_status_message = f"Captured: {samples_collected}/{FRAMES_TO_COLLECT}"
                            else:
                                calibration_status_message = f"Hold still: projected jitter {std_x:.3f}"
                        else:
                            calibration_status_message = "Locking onto stable gaze..."

                        if samples_collected >= FRAMES_TO_COLLECT:
                            collecting = False
                            calibration_status_message = ""
                            calibration_anchor_feature = None
                            calibration_stability_buffer.clear()
                            calibration_index += 1
                            if calibration_index >= len(calibration_targets):
                                calibration.fit()
                                quality_map.clear()
                                for (ax, ay, sx, sy), err in zip(calibration.points, calibration.reprojection_errors()):
                                    quality_map.record_error((sx, sy), err)
                                if not args.no_personal_model:
                                    session_pts = [(ax, ay, sx, sy) for ax, ay, sx, sy in calibration.points]
                                    store.add_session_samples(session_pts)
                                    store.save()
                                    all_pts = store.get_weighted_points()
                                    if len(all_pts) >= 20:
                                        enriched = HomographyCalibrationModel()
                                        for ax, ay, sx, sy in all_pts:
                                            enriched.add_sample(ax, ay, sx, sy)
                                        try:
                                            enriched.fit()
                                            calibration = enriched
                                            osk_message = f"Personal model fitted on {len(all_pts)} lifetime samples"
                                        except Exception:
                                            pass
                                heatmap = draw_quality_heatmap(cv2, screen_w, screen_h, quality_map)
                                cv2.imshow("Full Screen Calibration", heatmap)
                                cv2.waitKey(3000)
                                calibrating = False
                                cv2.destroyWindow("Full Screen Calibration")
                                sequencer.quality_map = quality_map
                                is_first_calibration = False
                                osk_message = "Gaze calibration fitted successfully"
    finally:
        if drag.enabled and not args.dry_run:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
