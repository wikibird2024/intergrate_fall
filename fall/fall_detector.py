# fall/fall_detector.py
import math
import numpy as np
from typing import List, Tuple

class FallDetector:
    def __init__(self):
        # Thresholds for fall detection
        self.TORSO_ANGLE_THRESHOLD = 60 # degrees
        self.VELOCITY_THRESHOLD = 0.5 # vertical velocity in pixels/frame
        self.FALL_DURATION_THRESHOLD = 15 # frames (e.g., 0.5 seconds at 30fps)
        self.FALL_STATE_DURATION_THRESHOLD = 30 # frames (to confirm the person is still down)
        
        # Internal state for tracking
        self.consecutive_frames_fallen: int = 0
        self.frames_in_fall_state: int = 0
        self.previous_torso_center: Optional[Tuple[float, float]] = None

    def _calculate_torso_angle(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculates the angle of the torso relative to the vertical axis.
        Assumes landmarks are ordered (e.g., from MediaPipe).
        """
        # Torso landmarks (example using hip and shoulder)
        left_shoulder, right_shoulder = landmarks[11], landmarks[12]
        left_hip, right_hip = landmarks[23], landmarks[24]
        
        # Midpoint of shoulders and hips
        torso_top = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        torso_bottom = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Vector of the torso
        torso_vector = (torso_top[0] - torso_bottom[0], torso_top[1] - torso_bottom[1])

        # Angle from the vertical axis (in degrees)
        angle = math.degrees(math.atan2(torso_vector[0], torso_vector[1]))
        return abs(angle)

    def _calculate_vertical_velocity(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculates the vertical velocity of the torso center.
        """
        left_shoulder, right_shoulder = landmarks[11], landmarks[12]
        left_hip, right_hip = landmarks[23], landmarks[24]
        torso_center = ((left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4,
                        (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4)
        
        if self.previous_torso_center is None:
            self.previous_torso_center = torso_center
            return 0.0

        vertical_velocity = torso_center[1] - self.previous_torso_center[1]
        self.previous_torso_center = torso_center
        return vertical_velocity

    def detect_fall(self, landmarks: List[Tuple[float, float]]) -> bool:
        """
        Detects a fall by combining multiple logical checks.
        Returns True if a fall event is confirmed.
        """
        if not landmarks:
            return False

        torso_angle = self._calculate_torso_angle(landmarks)
        vertical_velocity = self._calculate_vertical_velocity(landmarks)

        # Check for initial fall conditions
        if torso_angle > self.TORSO_ANGLE_THRESHOLD and vertical_velocity > self.VELOCITY_THRESHOLD:
            self.consecutive_frames_fallen += 1
        else:
            self.consecutive_frames_fallen = 0

        # Check if the person is in a 'fall state' (e.g., lying down)
        if torso_angle > 80: # A higher angle to confirm the person is lying horizontally
            self.frames_in_fall_state += 1
        else:
            self.frames_in_fall_state = 0
            
        # Confirm a fall event if conditions are met for a sustained period
        return self.consecutive_frames_fallen >= self.FALL_DURATION_THRESHOLD or \
               self.frames_in_fall_state >= self.FALL_STATE_DURATION_THRESHOLD
