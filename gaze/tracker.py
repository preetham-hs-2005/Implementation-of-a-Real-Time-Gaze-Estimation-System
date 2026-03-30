from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
LEFT_EYE = {"left": 33, "right": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"left": 362, "right": 263, "top": 386, "bottom": 374}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1
LEFT_EYE_CENTER = [33, 133]
RIGHT_EYE_CENTER = [362, 263]
CHIN = 152
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10
HEAD_POSE_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
        (-67.0, 10.0, -50.0),
        (67.0, 10.0, -50.0),
        (0.0, 80.0, -20.0),
    ],
    dtype=np.float64,
)
HEAD_POSE_INDICES = [NOSE_TIP, CHIN, 33, 263, LEFT_MOUTH, RIGHT_MOUTH, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]
EAR_VALID_MIN = 0.20
EAR_VALID_MAX = 0.45
EAR_BLINK_MAX = 0.15


@dataclass
class FaceObservation:
    gaze_norm: Point
    eye_gaze_norm: Point
    ear: float
    left_iris_px: Point
    right_iris_px: Point
    transform_matrix: Optional[Sequence[Sequence[float]]]
    iris_visibility: float
    face_center_norm: Point
    face_size_norm: Point
    nose_norm: Point
    left_eye_center_norm: Point
    right_eye_center_norm: Point
    interocular_distance_norm: float
    yaw: float
    pitch: float
    roll: float
    left_iris_relative: Point
    right_iris_relative: Point
    gaze_vector: Tuple[float, float, float]
    head_rotation_matrix: tuple[tuple[float, float, float], ...]
    iris_valid: bool
    iris_weight: float


def landmark_to_pixel(landmark, width: int, height: int) -> Point:
    return landmark.x * width, landmark.y * height


def eye_aspect_ratio(landmarks, eye_idx: dict[str, int], width: int, height: int) -> float:
    left = np.array(landmark_to_pixel(landmarks[eye_idx["left"]], width, height))
    right = np.array(landmark_to_pixel(landmarks[eye_idx["right"]], width, height))
    top = np.array(landmark_to_pixel(landmarks[eye_idx["top"]], width, height))
    bottom = np.array(landmark_to_pixel(landmarks[eye_idx["bottom"]], width, height))
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return float(vertical / horizontal)


def iris_center(landmarks, iris_idx: Iterable[int], width: int, height: int) -> Point:
    points = np.array([landmark_to_pixel(landmarks[i], width, height) for i in iris_idx], dtype=np.float32)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def landmark_mean_norm(landmarks, indices: Iterable[int]) -> Point:
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def iris_relative_position(landmarks, eye_idx: dict[str, int], iris_px: Point, width: int, height: int) -> Point:
    left_pt = np.array(landmark_to_pixel(landmarks[eye_idx["left"]], width, height), dtype=np.float32)
    right_pt = np.array(landmark_to_pixel(landmarks[eye_idx["right"]], width, height), dtype=np.float32)
    top_pt = np.array(landmark_to_pixel(landmarks[eye_idx["top"]], width, height), dtype=np.float32)
    bot_pt = np.array(landmark_to_pixel(landmarks[eye_idx["bottom"]], width, height), dtype=np.float32)

    h_center = (left_pt + right_pt) / 2.0
    v_center = (top_pt + bot_pt) / 2.0
    h_span = max(np.linalg.norm(right_pt - left_pt), 1e-6)
    v_span = max(np.linalg.norm(bot_pt - top_pt), 1e-6)

    iris = np.array(iris_px, dtype=np.float32)
    rel_x = float(np.clip((iris[0] - h_center[0]) / h_span, -0.6, 0.6))
    rel_y = float(np.clip((iris[1] - v_center[1]) / v_span, -0.6, 0.6))
    return rel_x, rel_y


def iris_validity_weight(ear: float) -> float:
    if ear < EAR_BLINK_MAX:
        return 0.0
    if ear < EAR_VALID_MIN:
        return 0.0
    if ear > EAR_VALID_MAX:
        return 1.0
    return (ear - EAR_VALID_MIN) / (EAR_VALID_MAX - EAR_VALID_MIN)


def estimate_head_pose(face_landmarks, width: int, height: int) -> tuple[float, float, float, tuple[tuple[float, float, float], ...]]:
    image_points = np.array(
        [[face_landmarks[idx].x * width, face_landmarks[idx].y * height] for idx in HEAD_POSE_INDICES],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.array(
        [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rotation_vec, _translation_vec = cv2.solvePnP(
            HEAD_POSE_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        ok, rotation_vec, _translation_vec = cv2.solvePnP(
            HEAD_POSE_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return 0.0, 0.0, 0.0, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    flip = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    rotation_matrix = flip @ rotation_matrix
    yaw = float(np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2]))
    pitch = float(np.arctan2(-rotation_matrix[1, 2], np.sqrt(rotation_matrix[0, 2] ** 2 + rotation_matrix[2, 2] ** 2)))
    roll = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1]))
    return yaw, pitch, roll, tuple(tuple(float(v) for v in row) for row in rotation_matrix)


def build_gaze_ray_camera_space(
    iris_px: Point,
    width: int,
    height: int,
    focal_length: Optional[float] = None,
) -> np.ndarray:
    fx = float(focal_length) if focal_length else float(width)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    nx = (iris_px[0] - cx) / fx
    ny = (iris_px[1] - cy) / fy
    ray = np.array([nx, ny, 1.0], dtype=np.float64)
    ray /= max(np.linalg.norm(ray), 1e-6)
    return ray


def combine_head_and_eye_gaze(
    left_iris_px: Point,
    right_iris_px: Point,
    rotation_matrix: Sequence[Sequence[float]],
    neutral_rotation: Optional[Sequence[Sequence[float]]] = None,
    width: int = 640,
    height: int = 480,
    left_iris_relative: Optional[Point] = None,
    right_iris_relative: Optional[Point] = None,
) -> tuple[float, float, float]:
    """Compute world-space gaze direction.

    When iris_relative positions are provided (the preferred path) we build the
    camera-space ray from iris position *within the eye socket* rather than the
    absolute iris pixel location.  Absolute pixel position is dominated by where
    the head sits in the camera frame (i.e. where the nose points), which causes
    the 'pointer follows nose' artefact.  Iris-relative position is a pure
    gaze-direction signal decoupled from head translation.
    """
    if left_iris_relative is not None and right_iris_relative is not None:
        # PURE IRIS TRACKING: Decoupled from head rotation.
        # This fixes "pointer follows my nose". We just map the position 
        # of the iris within the eye socket straight to a steering vector.
        # EYE_SCALE=2.0 massively boosts iris movement range.
        EYE_SCALE = 2.0
        rel_x = (left_iris_relative[0] + right_iris_relative[0]) / 2.0
        rel_y = (left_iris_relative[1] + right_iris_relative[1]) / 2.0
        world_gaze = np.array([rel_x * EYE_SCALE, rel_y * EYE_SCALE, 1.0], dtype=np.float64)
        world_gaze /= max(np.linalg.norm(world_gaze), 1e-6)
    else:
        # Fallback: use absolute iris position
        left_ray = build_gaze_ray_camera_space(left_iris_px, width, height)
        right_ray = build_gaze_ray_camera_space(right_iris_px, width, height)
        mean_ray = (left_ray + right_ray) / 2.0
        mean_ray /= max(np.linalg.norm(mean_ray), 1e-6)
        
        rotation = np.array(rotation_matrix, dtype=np.float64)
        world_gaze = rotation @ mean_ray
        world_gaze /= max(np.linalg.norm(world_gaze), 1e-6)
        
        if neutral_rotation is not None:
            neutral = np.array(neutral_rotation, dtype=np.float64)
            world_gaze = neutral.T @ world_gaze
            world_gaze /= max(np.linalg.norm(world_gaze), 1e-6)

    return float(world_gaze[0]), float(world_gaze[1]), float(world_gaze[2])


def extract_observation(
    face_landmarks,
    transform_matrix: Optional[Sequence[Sequence[float]]],
    width: int,
    height: int,
    neutral_rotation: Optional[Sequence[Sequence[float]]] = None,
) -> FaceObservation:
    left_iris = iris_center(face_landmarks, LEFT_IRIS, width, height)
    right_iris = iris_center(face_landmarks, RIGHT_IRIS, width, height)
    gaze_px = ((left_iris[0] + right_iris[0]) / 2.0, (left_iris[1] + right_iris[1]) / 2.0)
    gaze_norm = (gaze_px[0] / width, gaze_px[1] / height)
    left_iris_relative = iris_relative_position(face_landmarks, LEFT_EYE, left_iris, width, height)
    right_iris_relative = iris_relative_position(face_landmarks, RIGHT_EYE, right_iris, width, height)
    left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, width, height)
    right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, width, height)
    ear = (left_ear + right_ear) / 2.0
    left_weight = iris_validity_weight(left_ear)
    right_weight = iris_validity_weight(right_ear)
    total_weight = left_weight + right_weight
    if total_weight < 0.1:
        iris_valid = False
        iris_weight = 0.0
        eye_gaze_norm = (0.0, 0.0)
    else:
        iris_valid = True
        iris_weight = min(1.0, total_weight / 2.0)
        eye_gaze_norm = (
            (left_iris_relative[0] * left_weight + right_iris_relative[0] * right_weight) / total_weight,
            (left_iris_relative[1] * left_weight + right_iris_relative[1] * right_weight) / total_weight,
        )
    xs = np.array([landmark.x for landmark in face_landmarks], dtype=np.float32)
    ys = np.array([landmark.y for landmark in face_landmarks], dtype=np.float32)
    face_center_norm = (float(xs.mean()), float(ys.mean()))
    face_size_norm = (float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    nose_norm = (float(face_landmarks[NOSE_TIP].x), float(face_landmarks[NOSE_TIP].y))
    left_eye_center_norm = landmark_mean_norm(face_landmarks, LEFT_EYE_CENTER)
    right_eye_center_norm = landmark_mean_norm(face_landmarks, RIGHT_EYE_CENTER)
    interocular_distance_norm = float(
        np.hypot(
            right_eye_center_norm[0] - left_eye_center_norm[0],
            right_eye_center_norm[1] - left_eye_center_norm[1],
        )
    )
    yaw, pitch, roll, head_rotation_matrix = estimate_head_pose(face_landmarks, width, height)
    gaze_vector = combine_head_and_eye_gaze(
        left_iris,
        right_iris,
        head_rotation_matrix,
        neutral_rotation=neutral_rotation,
        width=width,
        height=height,
        left_iris_relative=left_iris_relative,
        right_iris_relative=right_iris_relative,
    )

    # MediaPipe FaceLandmarker often does not populate visibility (returns None).
    # If the face was detected and landmarks exist, we assume the iris is visible enough.
    iris_visibility = iris_weight if iris_valid else 0.0
    return FaceObservation(
        gaze_norm=gaze_norm,
        eye_gaze_norm=eye_gaze_norm,
        ear=ear,
        left_iris_px=left_iris,
        right_iris_px=right_iris,
        transform_matrix=transform_matrix,
        iris_visibility=iris_visibility,
        face_center_norm=face_center_norm,
        face_size_norm=face_size_norm,
        nose_norm=nose_norm,
        left_eye_center_norm=left_eye_center_norm,
        right_eye_center_norm=right_eye_center_norm,
        interocular_distance_norm=interocular_distance_norm,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        left_iris_relative=left_iris_relative,
        right_iris_relative=right_iris_relative,
        gaze_vector=gaze_vector,
        head_rotation_matrix=head_rotation_matrix,
        iris_valid=iris_valid,
        iris_weight=iris_weight,
    )
