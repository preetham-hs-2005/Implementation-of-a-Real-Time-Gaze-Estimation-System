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

from gaze.calibration import CalibrationModel, apply_head_pose_compensation, default_calibration_targets
from gaze.controller import CursorSmoother, map_to_screen
from gaze.interactions import AdaptiveBlinkDetector, DragState, DwellClickDetector
from gaze.tracker import extract_observation


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye-controlled cursor using latest MediaPipe Face Landmarker.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
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

    parser.add_argument("--calibration-points", type=int, default=9, choices=[5, 9], help="Calibration grid size.")
    parser.add_argument("--pose-comp-gain", type=float, default=0.08, help="Head-pose compensation gain.")

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
        raise RuntimeError("MediaPipe import failed. Run: pip install -r requirements.txt") from exc

    try:
        import pyautogui
    except Exception as exc:
        raise RuntimeError("pyautogui import failed. Run: pip install -r requirements.txt") from exc

    return cv2, mp, pyautogui


def draw_status(cv2, frame, text: str, line: int) -> None:
    cv2.putText(frame, text, (10, 24 + line * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Try a different --camera index.")

    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()

    calibration = CalibrationModel()
    calibration_targets = default_calibration_targets(args.calibration_points)
    calibrating = False
    calibration_index = 0

    cursor = CursorSmoother(alpha=args.smoothing_alpha, velocity_damping=args.velocity_damping, max_step=args.max_step)
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

    try:
        with FaceLandmarker.create_from_options(options) as face_landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if args.flip:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))

                status = 0
                if result.face_landmarks:
                    transform_matrix: Optional[list[list[float]]] = None
                    if result.facial_transformation_matrixes:
                        transform_matrix = result.facial_transformation_matrixes[0]

                    obs = extract_observation(result.face_landmarks[0], transform_matrix, width, height)
                    gaze_norm = apply_head_pose_compensation(obs.gaze_norm, obs.transform_matrix, gain=args.pose_comp_gain)

                    if calibrating:
                        tx, ty = calibration_targets[calibration_index]
                        cv2.circle(frame, (int(tx * width), int(ty * height)), 12, (0, 255, 255), -1)
                        draw_status(cv2, frame, "Calibration active: look at dot then press SPACE", status)
                        status += 1
                    else:
                        mapped_norm = calibration.map(gaze_norm)
                        target_cursor = map_to_screen(mapped_norm[0], mapped_norm[1], screen_w, screen_h, args.margin)
                        smooth_cursor = cursor.update(target_cursor)
                        if not args.dry_run:
                            pyautogui.moveTo(smooth_cursor[0], smooth_cursor[1], _pause=False)

                        if blink.update(obs.ear, time.time()):
                            if not args.dry_run:
                                if click_mode == "left":
                                    pyautogui.click(button="left")
                                else:
                                    pyautogui.click(button="right")
                            draw_status(cv2, frame, f"Blink {click_mode} click", status)
                            status += 1

                        if dwell_enabled and dwell.update(smooth_cursor, time.time()):
                            if not args.dry_run:
                                pyautogui.click(button=click_mode)
                            draw_status(cv2, frame, f"Dwell {click_mode} click", status)
                            status += 1

                        draw_status(cv2, frame, f"Cursor: ({int(smooth_cursor[0])}, {int(smooth_cursor[1])})", status)
                        status += 1

                    draw_status(cv2, frame, f"EAR: {obs.ear:.3f} baseline: {blink.baseline_ear:.3f}", status)
                    status += 1

                    if args.show_debug:
                        cv2.circle(frame, (int(obs.left_iris_px[0]), int(obs.left_iris_px[1])), 4, (255, 0, 0), -1)
                        cv2.circle(frame, (int(obs.right_iris_px[0]), int(obs.right_iris_px[1])), 4, (0, 0, 255), -1)
                else:
                    cursor.reset()
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

                cv2.imshow("Gaze Cursor Control", frame)
                key = cv2.waitKey(1) & 0xFF
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
                if calibrating and key == ord(" "):
                    if result.face_landmarks:
                        transform_matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
                        obs = extract_observation(result.face_landmarks[0], transform_matrix, width, height)
                        gaze_norm = apply_head_pose_compensation(obs.gaze_norm, obs.transform_matrix, gain=args.pose_comp_gain)
                        calibration.add_sample(gaze_norm, calibration_targets[calibration_index])
                        calibration_index += 1
                        if calibration_index >= len(calibration_targets):
                            calibration.fit()
                            calibrating = False
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
