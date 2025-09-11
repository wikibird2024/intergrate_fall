
# fall/fall_detector.py
import math
from typing import List, Tuple, Optional

class FallDetector:
    def __init__(
        self,
        torso_angle_threshold: float = 60.0,   # degrees
        velocity_threshold: float = 0.5,       # pixels/frame (not normalized by fps)
        fall_duration_threshold: int = 15,     # consecutive frames
        fall_state_duration_threshold: int = 30 # frames lying down
    ):
        # Thresholds
        self.TORSO_ANGLE_THRESHOLD = torso_angle_threshold
        self.VELOCITY_THRESHOLD = velocity_threshold
        self.FALL_DURATION_THRESHOLD = fall_duration_threshold
        self.FALL_STATE_DURATION_THRESHOLD = fall_state_duration_threshold
        
        # Internal states
        self.consecutive_frames_fallen: int = 0
        self.frames_in_fall_state: int = 0
        self.previous_torso_center: Optional[Tuple[float, float]] = None

        # Landmarks index (MediaPipe Pose)
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24

    def reset(self):
        """Reset internal states."""
        self.consecutive_frames_fallen = 0
        self.frames_in_fall_state = 0
        self.previous_torso_center = None

    def _calculate_torso_angle(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculates the torso angle relative to vertical axis.
        """
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        torso_top = ((left_shoulder[0] + right_shoulder[0]) / 2,
                     (left_shoulder[1] + right_shoulder[1]) / 2)
        torso_bottom = ((left_hip[0] + right_hip[0]) / 2,
                        (left_hip[1] + right_hip[1]) / 2)

        # Torso vector
        torso_vector = (torso_top[0] - torso_bottom[0],
                        torso_top[1] - torso_bottom[1])

        # Normalize and compute angle with vertical vector (0,1)
        dot = torso_vector[0] * 0 + torso_vector[1] * 1
        mag = math.sqrt(torso_vector[0]**2 + torso_vector[1]**2)
        if mag == 0:
            return 0.0
        cos_angle = dot / mag
        angle = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))
        return angle

    def _calculate_vertical_velocity(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculates vertical velocity of torso center.
        """
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        torso_center = (
            (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4,
            (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4,
        )

        if self.previous_torso_center is None:
            self.previous_torso_center = torso_center
            return 0.0

        vertical_velocity = torso_center[1] - self.previous_torso_center[1]
        self.previous_torso_center = torso_center
        return vertical_velocity

    def detect_fall(self, landmarks: List[Tuple[float, float]]) -> bool:
        """
        Detects a fall event. Returns True if confirmed.
        """
        if not landmarks:
            return False

        torso_angle = self._calculate_torso_angle(landmarks)
        vertical_velocity = self._calculate_vertical_velocity(landmarks)

        # Initial fall condition
        if torso_angle > self.TORSO_ANGLE_THRESHOLD and vertical_velocity > self.VELOCITY_THRESHOLD:
            self.consecutive_frames_fallen += 1
        else:
            self.consecutive_frames_fallen = 0

        # Fall state (lying down)
        if torso_angle > 80:
            self.frames_in_fall_state += 1
        else:
            self.frames_in_fall_state = 0

        return (
            self.consecutive_frames_fallen >= self.FALL_DURATION_THRESHOLD
            or self.frames_in_fall_state >= self.FALL_STATE_DURATION_THRESHOLD
        )
