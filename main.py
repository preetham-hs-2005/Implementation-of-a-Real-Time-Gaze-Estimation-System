#!/usr/bin/env python3
"""Real-time gaze-controlled cursor with calibration and advanced interactions."""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from gaze.calibration import (
    HEAD_POSE_SEQUENCE,
    CalibrationModel,
    HeadPoseCalibrator,
    HeadPoseExtremes,
    HeadPoseReference,
    HeadPoseSequenceCalibrator,
    apply_head_pose_compensation,
    build_gaze_feature_vector,
    default_calibration_targets,
    make_head_pose_sample,
    normalize_gaze_by_head_pose,
)
from gaze.controller import CursorSmoother, map_to_screen, apply_sensitivity
from gaze.interactions import AdaptiveBlinkDetector, DragState, DwellClickDetector
from gaze.tracker import extract_observation


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
    parser.add_argument("--margin", type=float, default=0.15, help="Dead margin from each edge (0-0.4).")
    parser.add_argument("--show-debug", action="store_true", help="Show landmarks and tracking overlays.")
    parser.add_argument("--dry-run", action="store_true", help="Do not move/click cursor, only visualize.")

    parser.add_argument("--smoothing-alpha", type=float, default=0.35, help="EMA smoothing alpha for cursor.")
    parser.add_argument("--velocity-damping", type=float, default=0.70, help="Velocity damping for smooth cursor.")
    parser.add_argument("--max-step", type=float, default=120.0, help="Max cursor step per frame.")

    parser.add_argument("--blink-frames", type=int, default=2, help="Consecutive low-EAR frames for click.")
    parser.add_argument("--click-cooldown", type=float, default=0.6, help="Seconds between clicks.")
    parser.add_argument("--blink-threshold-ratio", type=float, default=0.70, help="Adaptive EAR threshold ratio.")
    parser.add_argument("--baseline-alpha", type=float, default=0.02, help="Adaptive baseline update rate.")

    parser.add_argument("--dwell-seconds", type=float, default=1.0, help="Seconds for dwell click trigger.")
    parser.add_argument("--dwell-radius", type=float, default=45.0, help="Movement radius for dwell trigger.")

    parser.add_argument("--calibration-points", type=int, default=16, choices=[5, 9, 16], help="Calibration grid size.")
    parser.add_argument("--pose-comp-gain", type=float, default=0.0, help="Head-pose compensation gain (0.0 to disable, higher for more head-motion effect).")
    parser.add_argument("--disable-pose-comp", action="store_true", help="Disable head-pose compensation entirely.")
    parser.add_argument("--gaze-smoothing-alpha", type=float, default=0.20, help="Gaze low-pass alpha (0.0-1.0), smaller is smoother.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Gaze-to-cursor sensitivity scale (0.05-2.0, default 1.0).")
    parser.add_argument("--head-calibration-frames", type=int, default=30, help="Stable frames required for head-position calibration.")
    parser.add_argument("--head-pose-calibration-frames", type=int, default=16, help="Frames required for each optional head pose direction.")
    parser.add_argument("--disable-head-pose-calibration", action="store_true", help="Skip directional head pose calibration and use only straight-ahead reference.")
    parser.add_argument("--head-center-tolerance", type=float, default=0.08, help="Allowed normalized head-center drift from calibrated pose.")
    parser.add_argument("--head-size-tolerance", type=float, default=0.16, help="Allowed normalized face-size drift from calibrated pose.")
    parser.add_argument("--recalibration-angle-threshold", type=float, default=1.2, help="Normalized yaw/pitch/roll threshold before suggesting recalibration.")

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
        center_tolerance=(0.12, 0.12),
        size_tolerance=(0.16, 0.20),
        eye_distance_tolerance=0.10,
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

    calibration = CalibrationModel()
    calibration_targets = default_calibration_targets(args.calibration_points)
    calibrating = False
    calibration_index = 0
    collecting = False
    samples_collected = 0
    FRAMES_TO_COLLECT = 15
    head_calibrator = HeadPoseCalibrator(required_samples=args.head_calibration_frames)
    head_pose_sequence = HeadPoseSequenceCalibrator(required_samples=args.head_pose_calibration_frames)
    head_reference: Optional[HeadPoseReference] = None
    head_extremes: Optional[HeadPoseExtremes] = None
    head_guidance_reference = default_head_reference()

    cursor = CursorSmoother(alpha=args.smoothing_alpha, velocity_damping=args.velocity_damping, max_step=args.max_step)
    prev_gaze_norm = None
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
                head_aligned = False
                active_head_reference = head_reference or head_guidance_reference

                if result and result.face_landmarks:
                    transform_matrix: Optional[list[list[float]]] = None
                    if result.facial_transformation_matrixes:
                        transform_matrix = result.facial_transformation_matrixes[0]

                    obs = extract_observation(result.face_landmarks[0], transform_matrix, width, height)
                    head_sample = make_head_pose_sample(obs)
                    head_aligned = active_head_reference.is_aligned(head_sample)
                    draw_head_guidance_overlay(
                        cv2,
                        frame,
                        obs,
                        active_head_reference,
                        aligned=head_aligned,
                        calibrating_head=head_reference is None,
                    )

                    if head_reference is None:
                        draw_status(cv2, frame, "Head calibration: align your face with the guide", status)
                        status += 1
                        if head_aligned:
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
                                if args.disable_head_pose_calibration:
                                    osk_message = "Head calibration complete. Press c to calibrate gaze."
                                else:
                                    osk_message = "Straight pose captured. Follow pose prompts to improve tilt/turn robustness."
                        else:
                            if head_calibrator.samples:
                                head_calibrator.reset()
                            draw_status(cv2, frame, f"Align head: {head_alignment_hint(obs, active_head_reference)}", status)
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
                            if head_aligned:
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
                                    osk_message = "Head pose calibration complete. Press c to calibrate gaze."
                            else:
                                draw_status(cv2, frame, f"Align head: {head_alignment_hint(obs, head_reference)}", status)
                                status += 1
                        elif not head_aligned:
                            cursor.reset()
                            draw_status(cv2, frame, f"Head out of range: {head_alignment_hint(obs, head_reference)}", status)
                            status += 1
                        else:
                            draw_status(cv2, frame, "Head alignment: good", status)
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
                        cv2.circle(calib_frame, (target_x, target_y), 30, (0, 255, 255), -1)
                        cv2.circle(calib_frame, (target_x, target_y), 10, (0, 0, 255), -1)
                        msg = (
                            f"Capturing... {samples_collected}/{FRAMES_TO_COLLECT}"
                            if collecting
                            else "Look at the dot, keep head aligned, then press SPACE"
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
                        cv2.imshow("Full Screen Calibration", calib_frame)
                        draw_status(cv2, frame, f"Gaze calibration point {calibration_index + 1}/{len(calibration_targets)}", status)
                        status += 1

                    if obs.iris_visibility < 0.5:
                        draw_status(cv2, frame, "Iris not visible clearly - adjust lighting/position", status)
                        status += 1
                        cursor.reset()
                    elif head_reference is not None and head_aligned and not calibrating:
                        gaze_features = build_gaze_feature_vector(obs, head_sample, head_reference, head_extremes)
                        gaze_norm = (gaze_features[0], gaze_features[1])
                        if not args.disable_pose_comp and args.pose_comp_gain > 0.0:
                            gaze_norm = apply_head_pose_compensation(gaze_norm, obs.transform_matrix, gain=args.pose_comp_gain)
                            gaze_features = (gaze_norm[0], gaze_norm[1], *gaze_features[2:])

                        if prev_gaze_norm is None:
                            smoothed_gaze = gaze_norm
                        else:
                            a = max(0.0, min(1.0, args.gaze_smoothing_alpha))
                            smoothed_gaze = (
                                prev_gaze_norm[0] * (1.0 - a) + gaze_norm[0] * a,
                                prev_gaze_norm[1] * (1.0 - a) + gaze_norm[1] * a,
                            )
                        prev_gaze_norm = smoothed_gaze

                        smooth_features = (smoothed_gaze[0], smoothed_gaze[1], *gaze_features[2:])
                        mapped_norm = calibration.map(smooth_features)
                        adjusted_norm = apply_sensitivity(mapped_norm[0], mapped_norm[1], args.sensitivity)
                        target_cursor = map_to_screen(adjusted_norm[0], adjusted_norm[1], screen_w, screen_h, args.margin)
                        smooth_cursor = cursor.update(target_cursor)
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
                            if not args.dry_run:
                                pyautogui.click(button=click_mode)
                            draw_status(cv2, frame, f"Dwell {click_mode} click", status)
                            status += 1

                        draw_status(cv2, frame, f"Cursor: ({int(smooth_cursor[0])}, {int(smooth_cursor[1])})", status)
                        status += 1
                        draw_status(cv2, frame, f"EAR: {obs.ear:.3f} baseline: {blink.baseline_ear:.3f}", status)
                        status += 1

                    draw_status(cv2, frame, f"Iris visibility: {obs.iris_visibility:.2f}", status)
                    status += 1

                    if args.show_debug:
                        cv2.circle(frame, (int(obs.left_iris_px[0]), int(obs.left_iris_px[1])), 4, (255, 0, 0), -1)
                        cv2.circle(frame, (int(obs.right_iris_px[0]), int(obs.right_iris_px[1])), 4, (0, 0, 255), -1)
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
                if osk_message:
                    draw_status(cv2, frame, osk_message, status)
                    status += 1
                draw_status(cv2, frame, "q quit | t gaze on/off | h reset-head | c gaze-calib | m/v/g/k controls", status)

                cv2.imshow(window_name, frame)
                if frame_count == 1:
                    print("[INFO] Window should be visible now. Press 'q' to quit.")
                key = normalize_key(cv2.waitKeyEx(10))
                if key == ord("q"):
                    break
                if key == ord("t"):
                    gaze_control_enabled = not gaze_control_enabled
                    cursor.reset()
                    prev_gaze_norm = None
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
                    prev_gaze_norm = None
                    osk_message = "Head reference reset. Align with guide to recalibrate."
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
                if key == ord("c") and head_reference is not None:
                    calibration.clear()
                    calibrating = True
                    calibration_index = 0
                    collecting = False
                    samples_collected = 0
                    prev_gaze_norm = None
                    # Create full screen window
                    cv2.namedWindow("Full Screen Calibration", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # Force it to the top
                    cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_TOPMOST, 1)

                if calibrating:
                    if key == ord(" "):
                        collecting = True
                        samples_collected = 0
                    
                    if collecting and obs is not None and head_sample is not None and head_reference is not None:
                        if not head_reference.is_aligned(head_sample):
                            collecting = False
                            osk_message = "Head moved outside guide. Re-align and press SPACE again."
                        else:
                            gaze_features = build_gaze_feature_vector(obs, head_sample, head_reference, head_extremes)
                            gaze_norm = (gaze_features[0], gaze_features[1])
                            if not args.disable_pose_comp and args.pose_comp_gain > 0.0:
                                gaze_norm = apply_head_pose_compensation(gaze_norm, obs.transform_matrix, gain=args.pose_comp_gain)
                                gaze_features = (gaze_norm[0], gaze_norm[1], *gaze_features[2:])

                            calibration.add_sample(gaze_features, calibration_targets[calibration_index])
                            samples_collected += 1

                        if samples_collected >= FRAMES_TO_COLLECT:
                            collecting = False
                            calibration_index += 1
                            if calibration_index >= len(calibration_targets):
                                calibration.fit()
                                calibrating = False
                                cv2.destroyWindow("Full Screen Calibration")
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
