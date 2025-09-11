
# fall/fall_detector_simple.py
import math
from typing import List, Tuple, Optional

class FallDetector:
    def __init__(
        self,
        torso_angle_threshold: float = 70.0,   # degrees
        velocity_threshold: float = 1.0,       # pixels/frame
        fall_duration_threshold: int = 20,     # frames
        fall_state_duration_threshold: int = 30 # frames lying down
    ):
        self.TORSO_ANGLE_THRESHOLD = torso_angle_threshold
        self.VELOCITY_THRESHOLD = velocity_threshold
        self.FALL_DURATION_THRESHOLD = fall_duration_threshold
        self.FALL_STATE_DURATION_THRESHOLD = fall_state_duration_threshold

        self.consecutive_frames_fallen: int = 0
        self.frames_in_fall_state: int = 0
        self.previous_torso_center: Optional[Tuple[float, float]] = None

        # MediaPipe Pose landmarks
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24

    def reset(self):
        self.consecutive_frames_fallen = 0
        self.frames_in_fall_state = 0
        self.previous_torso_center = None

    def _valid_landmarks(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        # Kiểm tra đủ landmark và visibility > 0.5
        for idx in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP]:
            if idx >= len(landmarks) or landmarks[idx][3] < 0.5:
                return False
        return True

    def _calculate_torso_angle(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        torso_top = ((left_shoulder[0] + right_shoulder[0]) / 2,
                     (left_shoulder[1] + right_shoulder[1]) / 2)
        torso_bottom = ((left_hip[0] + right_hip[0]) / 2,
                        (left_hip[1] + right_hip[1]) / 2)

        torso_vector = (torso_top[0] - torso_bottom[0], torso_top[1] - torso_bottom[1])
        mag = math.hypot(*torso_vector)
        if mag == 0:
            return 0.0
        cos_angle = torso_vector[1] / mag  # angle with vertical
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def _calculate_vertical_velocity(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
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

    def detect_fall(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        if not self._valid_landmarks(landmarks):
            self.consecutive_frames_fallen = 0
            self.frames_in_fall_state = 0
            return False

        torso_angle = self._calculate_torso_angle(landmarks)
        vertical_velocity = self._calculate_vertical_velocity(landmarks)

        # Initial fall
        if torso_angle > self.TORSO_ANGLE_THRESHOLD and vertical_velocity > self.VELOCITY_THRESHOLD:
            self.consecutive_frames_fallen += 1
        else:
            self.consecutive_frames_fallen = 0

        # Fall state lying down
        if torso_angle > 85:
            self.frames_in_fall_state += 1
        else:
            self.frames_in_fall_state = 0

        return (
            self.consecutive_frames_fallen >= self.FALL_DURATION_THRESHOLD
            or self.frames_in_fall_state >= self.FALL_STATE_DURATION_THRESHOLD
        )
