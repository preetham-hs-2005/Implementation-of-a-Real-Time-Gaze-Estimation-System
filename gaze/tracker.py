from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
LEFT_EYE = {"left": 33, "right": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"left": 362, "right": 263, "top": 386, "bottom": 374}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


@dataclass
class FaceObservation:
    gaze_norm: Point
    ear: float
    left_iris_px: Point
    right_iris_px: Point
    transform_matrix: Optional[Sequence[Sequence[float]]]
    iris_visibility: float


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
    left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, width, height)
    right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, width, height)
    ear = (left_ear + right_ear) / 2.0

    # MediaPipe FaceLandmarker often does not populate visibility (returns None).
    # If the face was detected and landmarks exist, we assume the iris is visible enough.
    iris_visibility = 1.0
    return FaceObservation(
        gaze_norm=gaze_norm,
        ear=ear,
        left_iris_px=left_iris,
        right_iris_px=right_iris,
        transform_matrix=transform_matrix,
        iris_visibility=iris_visibility,
    )
