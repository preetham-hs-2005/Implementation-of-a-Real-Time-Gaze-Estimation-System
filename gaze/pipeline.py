import math
import numpy as np


def iris_to_gaze_ray_head_space(
    left_rel: tuple[float, float],
    right_rel: tuple[float, float],
) -> np.ndarray:
    """
    Convert iris-relative positions to a unit gaze direction vector
    in head-local coordinate space.
    
    The eye can rotate roughly ±35 deg horizontally and ±25 deg vertically.
    Map iris_rel linearly to these angles then convert to unit vector.
    
    This vector is ONLY a function of where the iris points in the socket.
    It has zero dependency on head position or camera geometry.
    """
    # Average both eyes
    rel_x = (left_rel[0] + right_rel[0]) / 2.0
    rel_y = (left_rel[1] + right_rel[1]) / 2.0

    # Map to angles: full iris travel (±0.5 rel) maps to ±35 deg horizontal, ±25 deg vertical
    # These scale factors can be refined by the calibration — they are initial estimates
    MAX_H_RAD = math.radians(35)
    MAX_V_RAD = math.radians(25)

    angle_h = rel_x * (MAX_H_RAD / 0.5)   # horizontal gaze angle in head space
    angle_v = rel_y * (MAX_V_RAD / 0.5)   # vertical gaze angle in head space

    # Convert spherical to Cartesian unit vector
    # Convention: X right, Y down, Z forward (into screen)
    gx = math.sin(angle_h) * math.cos(angle_v)
    gy = math.sin(angle_v)
    gz = math.cos(angle_h) * math.cos(angle_v)

    ray = np.array([gx, gy, gz], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray


def head_gaze_to_world(
    gaze_head: np.ndarray,
    R_head: np.ndarray,
    R_neutral: np.ndarray,
) -> np.ndarray:
    """
    Transform the head-local gaze ray to a world-space direction
    that is independent of head position and neutral-pose-normalised.
    
    R_head:    current head rotation matrix from solvePnP (3x3)
    R_neutral: head rotation matrix captured during head reference setup (3x3)
    
    The neutral normalisation means: when the user looks straight ahead
    with head in any orientation, the output ray is always (0, 0, 1).
    Moving the head left/right does not change the output if the eye
    direction relative to the head is unchanged.
    """
    # Rotate gaze from head-local to camera space
    gaze_camera = R_head @ gaze_head

    # Express relative to neutral head orientation
    # R_neutral.T rotates camera-space back to neutral-head frame
    # This removes the contribution of head rotation from the gaze signal
    gaze_neutral = R_neutral.T @ gaze_camera
    gaze_neutral /= max(np.linalg.norm(gaze_neutral), 1e-6)
    return gaze_neutral


def gaze_ray_to_angles(gaze_world: np.ndarray) -> tuple[float, float]:
    """
    Project 3D gaze direction to (angle_x, angle_y) in radians.
    These are the only two values fed into calibration.
    They are pure direction — zero position information.
    """
    gx, gy, gz = float(gaze_world[0]), float(gaze_world[1]), float(gaze_world[2])

    # Ensure gz is positive (pointing toward screen)
    if gz < 0:
        gx, gy, gz = -gx, -gy, -gz
    gz = max(gz, 0.05)

    angle_x = math.atan2(gx, gz)   # positive = looking right
    angle_y = math.atan2(gy, gz)   # positive = looking down

    return angle_x, angle_y
