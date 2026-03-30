import re
from pathlib import Path

content = Path("main.py").read_text(encoding="utf-8")

# Group 1: imports
content = re.sub(
    r"from gaze\.calibration import \([\s\S]+?make_head_pose_sample,\n\)",
    """from gaze.calibration import (
    AdaptiveCalibrationSequencer,
    CalibrationModel,
    CalibrationQualityMap,
    GazeDriftCorrector,
    NeutralPoseCapture,
    HomographyCalibrationModel,
    default_calibration_targets,
)""",
    content
)
content = re.sub(
    r"from gaze\.controller import CursorSmoother, HeadPoseSmoother, map_to_screen, apply_sensitivity",
    """from gaze.pipeline import iris_to_gaze_ray_head_space, head_gaze_to_world, gaze_ray_to_angles\nfrom gaze.controller import CursorSmoother, RotationSmoother, map_to_screen, apply_sensitivity""",
    content
)
content = re.sub(r"from gaze\.tracker import combine_head_and_eye_gaze, extract_observation", "from gaze.tracker import extract_observation", content)

# Remove unused functions from line 145 to 286 (default_head_reference to compute_gaze_angles)
content = re.sub(r"def default_head_reference\(\)[\s\S]*?def compute_gaze_angles\([\s\S]*?return gaze_to_screen_angles\(gaze_vec\)\n\n\n", "", content)

# Setup Variables
content = re.sub(
    r"    head_calibrator = HeadPoseCalibrator[\s\S]*?    head_guidance_reference = default_head_reference\(\)\n",
    "    neutral_capture = NeutralPoseCapture(required_frames=args.head_calibration_frames)\n",
    content
)
content = re.sub(r"head_pose_smoother = HeadPoseSmoother\(alpha=0\.25\)", "rotation_smoother = RotationSmoother(alpha=0.25)", content)

# Remove unused local variables in tracking loop
content = content.replace("head_sample = None\n", "")
content = content.replace("active_head_reference = head_reference or head_guidance_reference\n", "")

# The large observation extraction and processing block
replacement = """                if result and result.face_landmarks:
                    obs = extract_observation(
                        result.face_landmarks[0],
                        width,
                        height,
                    )
                    
                    R_current = np.array(obs.head_rotation_matrix, dtype=np.float64)
                    flip = np.diag([1, -1, -1]).astype(np.float64)
                    R_current = flip @ R_current
                    R_smooth = rotation_smoother.update(R_current)
                    
                    gaze_ray_head = iris_to_gaze_ray_head_space(obs.left_iris_relative, obs.right_iris_relative)
                    
                    if not neutral_capture.is_ready:
                        neutral_capture.add_frame(R_smooth, obs.iris_valid, fixation.is_fixating)
                        draw_status(cv2, frame, f"Auto-capturing head reference... {int(neutral_capture.progress*100)}%", status)
                        status += 1
                        raw_angles = None
                    else:
                        gaze_ray_world = head_gaze_to_world(gaze_ray_head, R_smooth, neutral_capture.R_neutral)
                        angle_x, angle_y = gaze_ray_to_angles(gaze_ray_world)
                        raw_angles = (angle_x, angle_y)

                    if raw_angles is not None:
                        if prev_gaze_angles is None:
                            smoothed_angles = raw_angles
                        else:
                            a = max(0.0, min(1.0, args.gaze_smoothing_alpha))
                            smoothed_angles = (
                                prev_gaze_angles[0] * (1.0 - a) + raw_angles[0] * a,
                                prev_gaze_angles[1] * (1.0 - a) + raw_angles[1] * a,
                            )
                        prev_gaze_angles = smoothed_angles"""

content = re.sub(
    r"                if result and result\.face_landmarks:.*?if prev_gaze_angles is None:\n\s*smoothed_angles = raw_angles\n\s*else:\n.+?prev_gaze_angles = smoothed_angles\n",
    replacement + "\n",
    content, count=1, flags=re.DOTALL
)

# Remove block: "draw_head_guidance_overlay(..." up to "if calibrating:"
content = re.sub(r"                    draw_head_guidance_overlay\([\s\S]*?if head_extremes is not None.*?status \+= 1\n", "", content)

# Handle UI drawing in calibrating mode
content = re.sub(
    r"                        msg = \([\s\S]*?cv2\.LINE_AA,\n                        \)",
    """                        status_color = (0, 220, 80) if (obs.iris_valid and fixation.is_fixating) else (0, 80, 255)
                        msg = "Ready - press SPACE" if (obs.iris_valid and fixation.is_fixating) else "Hold still, eyes open"
                        cv2.putText(
                            calib_frame,
                            msg,
                            (screen_w // 2 - 320, screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            status_color,
                            2,
                            cv2.LINE_AA,
                        )""",
    content
)

# Handle Debug Overlay
# "draw_gaze_indicator" missing error removal
content = re.sub(r"                        draw_gaze_indicator\(cv2, frame, obs\)\n", "", content)
content = re.sub(r"                        if not calibration\.is_fitted and smoothed_angles is not None:[\s\S]*?\(gaze_state\)\n", "", content)

# Debug outputs
debug_replace = """                    if args.debug_gaze and frame_count % 30 == 0:
                        print(f"[PIPELINE] iris_rel L={obs.left_iris_relative} R={obs.right_iris_relative}")
                        print(f"[PIPELINE] gaze_ray_head={gaze_ray_head.round(3)}")
                        if neutral_capture.is_ready:
                            print(f"[PIPELINE] gaze_ray_world={gaze_ray_world.round(3)}")
                            print(f"[PIPELINE] angle_x={angle_x:+.3f} rad  angle_y={angle_y:+.3f} rad")
                            if mapped_norm:
                                print(f"[PIPELINE] mapped=({mapped_norm[0]:.3f}, {mapped_norm[1]:.3f})")"""
content = re.sub(r"                    if args\.debug_gaze and obs is not None and frame_count % 30 == 0:[\s\S]*?print\(\"\[DEBUG\] ---\"\)\n", debug_replace + "\n", content)

# else (no face detected) block:
content = re.sub(
    r"                    draw_head_guidance_overlay\([\s\S]*?calibrating_head=head_reference is None,\n                    \)",
    "",
    content
)

# Head status and keyboard handles
content = re.sub(r"head_status = \"ready\" if head_reference is not None else \"capturing\"", "head_status = \"ready\" if neutral_capture.is_ready else \"capturing\"", content)

content = re.sub(r"                if key == ord\(\"h\"\):[\s\S]*?osk_message = \"Head reference reset\. Align with guide to recalibrate\.\"\n", 
"""                if key == ord("h"):
                    neutral_capture.reset()
                    osk_message = "Head reference reset."
""", content)

content = re.sub(r"                if key == ord\(\"j\"\) and head_sample is not None and head_reference is None:[\s\S]*?osk_message = f\"Capturing straight head\.\.\. \{int\(head_calibrator\.progress \* 100\)\}%\n",
"""                if key == ord("j"):
                    neutral_capture.reset()
                    osk_message = "Recapturing head reference — look straight at screen."
""", content)

content = re.sub(r"                if key == ord\(\"c\"\):\s*neutral_rotation = head_reference.*?\n[\s\S]*?if head_extremes is None:[\s\S]*?redo full pose calibration\.\"\n",
"""                if key == ord("c"):
                    if not neutral_capture.is_ready:
                        osk_message = "Waiting for auto head reference capture. Hold still or press j."
                    else:""", content)

content = content.replace("and head_sample is not None and head_reference is not None:", "")

Path("main2.py").write_text(content, encoding="utf-8")
