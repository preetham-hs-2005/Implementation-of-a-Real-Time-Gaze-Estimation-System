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
    ear: float
    left_iris_px: Point
    right_iris_px: Point
    iris_visibility: float
    iris_weight: float
    left_iris_relative: Point
    right_iris_relative: Point
    yaw: float
    pitch: float
    roll: float
    head_rotation_matrix: tuple[tuple[float, float, float], ...]
    iris_valid: bool


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
    left_pt = np.array(landmark_to_pixel(landmarks[eye_idx["left"]], width, height), dtype=np.float64)
    right_pt = np.array(landmark_to_pixel(landmarks[eye_idx["right"]], width, height), dtype=np.float64)
    top_pt = np.array(landmark_to_pixel(landmarks[eye_idx["top"]], width, height), dtype=np.float64)
    bot_pt = np.array(landmark_to_pixel(landmarks[eye_idx["bottom"]], width, height), dtype=np.float64)

    eye_center_h = (left_pt + right_pt) / 2.0
    eye_center_v = (top_pt + bot_pt) / 2.0
    h_span = max(np.linalg.norm(right_pt - left_pt), 1e-6)
    v_span = max(np.linalg.norm(bot_pt - top_pt), 1e-6)

    iris = np.array(iris_px, dtype=np.float64)
    rel_x = (iris[0] - eye_center_h[0]) / h_span
    rel_y = (iris[1] - eye_center_v[1]) / v_span

    return float(np.clip(rel_x, -0.5, 0.5)), float(np.clip(rel_y, -0.5, 0.5))


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
    yaw = float(np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2]))
    pitch = float(np.arctan2(-rotation_matrix[1, 2], np.sqrt(rotation_matrix[0, 2] ** 2 + rotation_matrix[2, 2] ** 2)))
    roll = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1]))
    return yaw, pitch, roll, tuple(tuple(float(v) for v in row) for row in rotation_matrix)


def extract_observation(
    face_landmarks,
    width: int,
    height: int,
) -> FaceObservation:
    left_iris = iris_center(face_landmarks, LEFT_IRIS, width, height)
    right_iris = iris_center(face_landmarks, RIGHT_IRIS, width, height)
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
    else:
        iris_valid = True
        iris_weight = min(1.0, total_weight / 2.0)
    
    yaw, pitch, roll, head_rotation_matrix = estimate_head_pose(face_landmarks, width, height)
    
    iris_visibility = iris_weight if iris_valid else 0.0
    return FaceObservation(
        ear=ear,
        left_iris_px=left_iris,
        right_iris_px=right_iris,
        iris_visibility=iris_visibility,
        iris_weight=iris_weight,
        left_iris_relative=left_iris_relative,
        right_iris_relative=right_iris_relative,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        head_rotation_matrix=head_rotation_matrix,
        iris_valid=iris_valid,
    )
