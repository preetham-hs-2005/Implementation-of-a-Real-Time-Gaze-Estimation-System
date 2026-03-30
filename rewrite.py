import re
from pathlib import Path

content = Path("main.py").read_text()

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
    """from gaze.pipeline import iris_to_gaze_ray_head_space, head_gaze_to_world, gaze_ray_to_angles
from gaze.controller import CursorSmoother, RotationSmoother, map_to_screen, apply_sensitivity""",
    content
)
content = re.sub(r"from gaze\.tracker import combine_head_and_eye_gaze, extract_observation", "from gaze.tracker import extract_observation", content)

# Remove unused functions
content = re.sub(r"def default_head_reference\(\).*?return np\.array\(feats, dtype=np\.float64\)\n\n", "", content, flags=re.DOTALL) # wait regex dotall is tricky here.

# Group 3 variables
content = re.sub(
    r"head_calibrator = HeadPoseCalibrator.*?drift = GazeDriftCorrector\(\)",
    """neutral_capture = NeutralPoseCapture(required_frames=args.head_calibration_frames)
    drift = GazeDriftCorrector()""",
    content, flags=re.DOTALL
)
content = re.sub(r"head_pose_smoother = HeadPoseSmoother\(alpha=0\.25\)", "rotation_smoother = RotationSmoother(alpha=0.25)", content)

# Group 4 extract blocks
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
                    obs.head_rotation_matrix = R_smooth
                    
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
                        if prev_gaze_angles is None:"""
content = re.sub(
    r"                if result and result\.face_landmarks:.*?if prev_gaze_angles is None:",
    replacement,
    content, count=1, flags=re.DOTALL
)

# Strip out drawing functions using sed style strings
content = content.replace("head_sample = None\n", "")
content = content.replace("active_head_reference = head_reference or head_guidance_reference\n", "")


Path("main2.py").write_text(content)
