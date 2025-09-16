import math
from typing import List, Tuple, Optional

class FallDetector:
    """
    Fall detector improved to detect falls in all directions, including sideways and forward/backward.
    """
    def __init__(
        self,
        torso_angle_threshold_vertical: float = 60.0,
        torso_angle_threshold_horizontal: float = 45.0,
        velocity_threshold: float = 0.5,
        fall_duration_threshold: int = 5,
        fall_state_duration_threshold: int = 5,
        min_landmark_confidence: float = 0.5
    ):
        self.TORSO_ANGLE_THRESHOLD_VERTICAL = torso_angle_threshold_vertical
        self.TORSO_ANGLE_THRESHOLD_HORIZONTAL = torso_angle_threshold_horizontal
        self.VELOCITY_THRESHOLD = velocity_threshold
        self.FALL_DURATION_THRESHOLD = fall_duration_threshold
        self.FALL_STATE_DURATION_THRESHOLD = fall_state_duration_threshold
        self.MIN_LANDMARK_CONFIDENCE = min_landmark_confidence

        # Internal states
        self.consecutive_frames_fallen = 0
        self.frames_in_fall_state = 0
        self.previous_torso_center: Optional[Tuple[float,float]] = None

        # Landmarks
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24

    def reset(self):
        """Đặt lại trạng thái sau khi xử lý xong một lần té ngã."""
        self.consecutive_frames_fallen = 0
        self.frames_in_fall_state = 0
        self.previous_torso_center = None

    def _calculate_torso_angle(self, landmarks: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Tính góc thân người so với trục dọc/ngang."""
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        torso_top = ((left_shoulder[0] + right_shoulder[0]) / 2,
                     (left_shoulder[1] + right_shoulder[1]) / 2)
        torso_bottom = ((left_hip[0] + right_hip[0]) / 2,
                        (left_hip[1] + right_hip[1]) / 2)

        dx = torso_top[0] - torso_bottom[0]
        dy = torso_top[1] - torso_bottom[1]
        
        # Using atan2 to calculate angles
        angle_vertical = abs(math.degrees(math.atan2(dx, dy)))
        angle_horizontal = abs(math.degrees(math.atan2(dy, dx)))
        
        return angle_vertical, angle_horizontal

    def _calculate_velocity(self, landmarks: List[Tuple[float, float]]) -> float:
        """Tính vận tốc thân người dựa vào dịch chuyển giữa 2 frame."""
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

        vx = torso_center[0] - self.previous_torso_center[0]
        vy = torso_center[1] - self.previous_torso_center[1]
        velocity_mag = math.sqrt(vx**2 + vy**2)
        self.previous_torso_center = torso_center
        return velocity_mag

    def detect_fall(self, landmarks: List[Tuple[float, float]]) -> bool:
        """Phát hiện té ngã dựa trên góc và vận tốc."""
        # Check for missing or incomplete landmarks
        if not landmarks or len(landmarks) < 33:
            self.reset()
            return False

        # Check landmark confidence
        for idx in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP]:
            if len(landmarks[idx]) < 3 or landmarks[idx][2] < self.MIN_LANDMARK_CONFIDENCE:
                self.reset()
                return False

        # Only get (x, y) coords
        coords = [(lm[0], lm[1]) for lm in landmarks]

        angle_v, angle_h = self._calculate_torso_angle(coords)
        velocity = self._calculate_velocity(coords)

        is_falling = (
            (angle_v > self.TORSO_ANGLE_THRESHOLD_VERTICAL or angle_h > self.TORSO_ANGLE_THRESHOLD_HORIZONTAL)
            and velocity > self.VELOCITY_THRESHOLD
        )
        is_lying = (
            angle_v > self.TORSO_ANGLE_THRESHOLD_VERTICAL + 10 or
            angle_h > self.TORSO_ANGLE_THRESHOLD_HORIZONTAL + 10
        )

        # Update counters independently
        if is_falling:
            self.consecutive_frames_fallen += 1
            self.frames_in_fall_state = 0
        elif is_lying:
            self.consecutive_frames_fallen = 0
            self.frames_in_fall_state += 1
        else:
            self.consecutive_frames_fallen = 0
            self.frames_in_fall_state = 0

        # Check thresholds for final decision and reset
        is_fall_detected = (
            self.consecutive_frames_fallen >= self.FALL_DURATION_THRESHOLD or
            self.frames_in_fall_state >= self.FALL_STATE_DURATION_THRESHOLD
        )
        
        if is_fall_detected:
            self.reset()
            
        return is_fall_detected
