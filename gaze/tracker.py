from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
LEFT_EYE = {"left": 33, "right": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"left": 362, "right": 263, "top": 386, "bottom": 374}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1
LEFT_EYE_CENTER = [33, 133]
RIGHT_EYE_CENTER = [362, 263]


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
    left = np.array(landmark_to_pixel(landmarks[eye_idx["left"]], width, height), dtype=np.float32)
    right = np.array(landmark_to_pixel(landmarks[eye_idx["right"]], width, height), dtype=np.float32)
    top = np.array(landmark_to_pixel(landmarks[eye_idx["top"]], width, height), dtype=np.float32)
    bottom = np.array(landmark_to_pixel(landmarks[eye_idx["bottom"]], width, height), dtype=np.float32)

    horizontal = max(np.linalg.norm(right - left), 1e-6)
    vertical = max(np.linalg.norm(bottom - top), 1e-6)
    center = (left + right + top + bottom) / 4.0
    iris = np.array(iris_px, dtype=np.float32)
    relative = np.array([(iris[0] - center[0]) / horizontal, (iris[1] - center[1]) / vertical], dtype=np.float32)
    return float(relative[0]), float(relative[1])


def extract_pose_angles(transform_matrix: Optional[Sequence[Sequence[float]]]) -> tuple[float, float, float]:
    if transform_matrix is None:
        return 0.0, 0.0, 0.0
    mat = np.array(transform_matrix, dtype=np.float64)
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return 0.0, 0.0, 0.0
    r = mat[:3, :3]
    yaw = float(np.arctan2(r[0, 2], r[2, 2]))
    pitch = float(np.arctan2(-r[1, 2], np.sqrt(r[0, 2] ** 2 + r[2, 2] ** 2)))
    roll = float(np.arctan2(r[1, 0], r[1, 1]))
    return yaw, pitch, roll


def extract_observation(
    face_landmarks,
    transform_matrix: Optional[Sequence[Sequence[float]]],
    width: int,
    height: int,
) -> FaceObservation:
    left_iris = iris_center(face_landmarks, LEFT_IRIS, width, height)
    right_iris = iris_center(face_landmarks, RIGHT_IRIS, width, height)
    gaze_px = ((left_iris[0] + right_iris[0]) / 2.0, (left_iris[1] + right_iris[1]) / 2.0)
    gaze_norm = (gaze_px[0] / width, gaze_px[1] / height)
    left_iris_relative = iris_relative_position(face_landmarks, LEFT_EYE, left_iris, width, height)
    right_iris_relative = iris_relative_position(face_landmarks, RIGHT_EYE, right_iris, width, height)
    eye_gaze_norm = (
        (left_iris_relative[0] + right_iris_relative[0]) / 2.0,
        (left_iris_relative[1] + right_iris_relative[1]) / 2.0,
    )
    left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, width, height)
    right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, width, height)
    ear = (left_ear + right_ear) / 2.0
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
    yaw, pitch, roll = extract_pose_angles(transform_matrix)

    # MediaPipe FaceLandmarker often does not populate visibility (returns None).
    # If the face was detected and landmarks exist, we assume the iris is visible enough.
    iris_visibility = 1.0
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
    )
