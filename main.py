#!/usr/bin/env python3
"""Real-time gaze-controlled cursor with calibration and advanced interactions."""
from __future__ import annotations

import argparse
from collections import deque
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from gaze.calibration import CalibrationModel, apply_head_pose_compensation, average_points, default_calibration_targets, is_stable_gaze
from gaze.controller import CursorSmoother, apply_precision_curve, map_to_screen, apply_sensitivity
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
    parser.add_argument("--margin", type=float, default=0.02, help="Dead margin from each edge (0-0.4).")
    parser.add_argument("--show-debug", action="store_true", help="Show landmarks and tracking overlays.")
    parser.add_argument("--dry-run", action="store_true", help="Do not move/click cursor, only visualize.")

    parser.add_argument("--smoothing-alpha", type=float, default=0.28, help="EMA smoothing alpha for cursor.")
    parser.add_argument("--velocity-damping", type=float, default=0.78, help="Velocity damping for smooth cursor.")
    parser.add_argument("--max-step", type=float, default=90.0, help="Max cursor step per frame.")

    parser.add_argument("--blink-frames", type=int, default=2, help="Consecutive low-EAR frames for click.")
    parser.add_argument("--click-cooldown", type=float, default=0.6, help="Seconds between clicks.")
    parser.add_argument("--blink-threshold-ratio", type=float, default=0.70, help="Adaptive EAR threshold ratio.")
    parser.add_argument("--baseline-alpha", type=float, default=0.02, help="Adaptive baseline update rate.")

    parser.add_argument("--dwell-seconds", type=float, default=1.0, help="Seconds for dwell click trigger.")
    parser.add_argument("--dwell-radius", type=float, default=45.0, help="Movement radius for dwell trigger.")

    parser.add_argument("--calibration-points", type=int, default=9, choices=[5, 9], help="Calibration grid size.")
    parser.add_argument("--calibration-frames", type=int, default=15, help="Frames to average after pressing SPACE for each calibration dot.")
    parser.add_argument("--pose-comp-gain", type=float, default=0.0, help="Head-pose compensation gain (0.0 to disable, higher for more head-motion effect).")
    parser.add_argument("--disable-pose-comp", action="store_true", help="Disable head-pose compensation entirely.")
    parser.add_argument("--gaze-smoothing-alpha", type=float, default=0.16, help="Gaze low-pass alpha (0.0-1.0), smaller is smoother.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Gaze-to-cursor sensitivity scale (0.05-2.0, default 1.0).")
    parser.add_argument("--deadzone", type=float, default=0.025, help="Ignore tiny gaze motion near center (0.0-0.3).")
    parser.add_argument("--precision-curve", type=float, default=1.35, help="Non-linear cursor response outside deadzone (1.0-4.0).")
    parser.add_argument("--prediction-window", type=int, default=4, help="Average this many recent predictions for steadier cursor control.")
    parser.add_argument("--disable-interaction-learning", action="store_true", help="Disable continuous calibration updates from blink and dwell interactions.")

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
    calibration_recent_samples: list[tuple[float, float]] = []

    cursor = CursorSmoother(alpha=args.smoothing_alpha, velocity_damping=args.velocity_damping, max_step=args.max_step)
    prev_gaze_norm = None
    prediction_history: deque[tuple[float, float]] = deque(maxlen=max(1, args.prediction_window))
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
                if calibrating:
                    # Create black full-screen image for calibration
                    import numpy as np
                    calib_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    tx, ty = calibration_targets[calibration_index]
                    target_x, target_y = int(tx * screen_w), int(ty * screen_h)
                    
                    # Draw target on the full screen frame
                    cv2.circle(calib_frame, (target_x, target_y), 30, (0, 255, 255), -1)
                    cv2.circle(calib_frame, (target_x, target_y), 10, (0, 0, 255), -1)
                    
                    if collecting:
                        msg = f"Capturing... {samples_collected}/{args.calibration_frames}"
                    else:
                        msg = "Look at the dot and press SPACE"
                    
                    cv2.putText(calib_frame, msg, (screen_w // 2 - 200, screen_h // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow("Full Screen Calibration", calib_frame)
                    
                    # Also show the camera feed to ensure user knows it's doing something
                    draw_status(cv2, frame, "Calibration in progress...", status)
                else:
                    if result.face_landmarks:
                        transform_matrix: Optional[list[list[float]]] = None
                        if result.facial_transformation_matrixes:
                            transform_matrix = result.facial_transformation_matrixes[0]

                        obs = extract_observation(result.face_landmarks[0], transform_matrix, width, height)

                        if obs.iris_visibility < 0.5:
                            draw_status(cv2, frame, "Iris not visible clearly - adjust lighting/position", status)
                            status += 1
                            cursor.reset()
                            prediction_history.clear()
                        else:
                            gaze_norm = obs.gaze_norm
                            # Optional head-pose compensation can be disabled for less head-motion effect
                            if not args.disable_pose_comp and args.pose_comp_gain > 0.0:
                                gaze_norm = apply_head_pose_compensation(gaze_norm, obs.transform_matrix, gain=args.pose_comp_gain)

                            # Soft gaze smoothing to reduce frame jitter
                            if prev_gaze_norm is None:
                                smoothed_gaze = gaze_norm
                            else:
                                a = max(0.0, min(1.0, args.gaze_smoothing_alpha))
                                smoothed_gaze = (
                                    prev_gaze_norm[0] * (1.0 - a) + gaze_norm[0] * a,
                                    prev_gaze_norm[1] * (1.0 - a) + gaze_norm[1] * a,
                                )
                            prev_gaze_norm = smoothed_gaze

                            mapped_norm = calibration.map(smoothed_gaze)
                            prediction_history.append(mapped_norm)
                            mapped_norm = average_points(list(prediction_history))
                            adjusted_norm = apply_sensitivity(mapped_norm[0], mapped_norm[1], args.sensitivity)
                            adjusted_norm = apply_precision_curve(adjusted_norm[0], adjusted_norm[1], args.deadzone, args.precision_curve)
                            target_cursor = map_to_screen(adjusted_norm[0], adjusted_norm[1], screen_w, screen_h, args.margin)
                            smooth_cursor = cursor.update(target_cursor)
                            cursor_norm = (
                                max(0.0, min(1.0, smooth_cursor[0] / max(screen_w, 1))),
                                max(0.0, min(1.0, smooth_cursor[1] / max(screen_h, 1))),
                            )
                            if not args.dry_run:
                                pyautogui.moveTo(smooth_cursor[0], smooth_cursor[1], _pause=False)

                            if blink.update(obs.ear, time.time()):
                                if not args.dry_run:
                                    if click_mode == "left":
                                        pyautogui.click(button="left")
                                    else:
                                        pyautogui.click(button="right")
                                if not args.disable_interaction_learning:
                                    calibration.refine_from_interaction(smoothed_gaze, cursor_norm)
                                draw_status(cv2, frame, f"Blink {click_mode} click", status)
                                status += 1

                            if dwell_enabled and dwell.update(smooth_cursor, time.time()):
                                if not args.dry_run:
                                    pyautogui.click(button=click_mode)
                                if not args.disable_interaction_learning:
                                    calibration.refine_from_interaction(smoothed_gaze, cursor_norm)
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
                        prediction_history.clear()
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
                if osk_message:
                    draw_status(cv2, frame, osk_message, status)
                    status += 1
                draw_status(cv2, frame, "q quit | c calib | m click-mode | v dwell | g drag | k keyboard", status)

                cv2.imshow(window_name, frame)
                if frame_count == 1:
                    print("[INFO] Window should be visible now. Press 'q' to quit.")
                key = normalize_key(cv2.waitKeyEx(10))
                if key == ord("q"):
                    break
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
                    calibration.clear()
                    calibrating = True
                    calibration_index = 0
                    collecting = False
                    samples_collected = 0
                    calibration_recent_samples = []
                    prediction_history.clear()
                    # Create full screen window
                    cv2.namedWindow("Full Screen Calibration", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # Force it to the top
                    cv2.setWindowProperty("Full Screen Calibration", cv2.WND_PROP_TOPMOST, 1)

                if calibrating:
                    if key == ord(" "):
                        collecting = True
                        samples_collected = 0
                        calibration_recent_samples = []

                    if collecting and result.face_landmarks:
                        transform_matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
                        obs = extract_observation(result.face_landmarks[0], transform_matrix, width, height)
                        if obs.iris_visibility < 0.5:
                            calibration_recent_samples = []
                            samples_collected = 0
                        else:
                            gaze_norm = obs.gaze_norm
                            if not args.disable_pose_comp and args.pose_comp_gain > 0.0:
                                gaze_norm = apply_head_pose_compensation(gaze_norm, obs.transform_matrix, gain=args.pose_comp_gain)
                            calibration_recent_samples.append(gaze_norm)
                            calibration_recent_samples = calibration_recent_samples[-args.calibration_frames:]
                            samples_collected = len(calibration_recent_samples)

                            if samples_collected >= args.calibration_frames:
                                stable_samples = calibration_recent_samples
                                if is_stable_gaze(calibration_recent_samples, 0.03):
                                    stable_samples = calibration_recent_samples
                                sample = average_points(stable_samples or [gaze_norm])
                                calibration.add_sample(sample, calibration_targets[calibration_index])
                                calibration_index += 1
                                collecting = False
                                samples_collected = 0
                                calibration_recent_samples = []
                                if calibration_index >= len(calibration_targets):
                                    calibration.fit()
                                    calibrating = False
                                    cv2.destroyWindow("Full Screen Calibration")
                                    osk_message = "Calibration fitted successfully"
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
