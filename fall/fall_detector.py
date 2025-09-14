import math
from typing import List, Tuple, Optional

class FallDetector:
    """
    Class to detect falls based on torso angle and vertical velocity
    from a person's body landmarks.
    """
    def __init__(
        self,
        torso_angle_threshold: float = 60.0,    # degrees, balanced for better detection
        velocity_threshold: float = 0.1,        # pixels/frame, kept low
        fall_duration_threshold: int = 5,       # consecutive frames, kept low
        fall_state_duration_threshold: int = 5, # frames lying down, kept low
        min_landmark_confidence: float = 0.5    # Lowered confidence threshold
    ):
        # Thresholds
        self.TORSO_ANGLE_THRESHOLD = torso_angle_threshold
        self.VELOCITY_THRESHOLD = velocity_threshold
        self.FALL_DURATION_THRESHOLD = fall_duration_threshold
        self.FALL_STATE_DURATION_THRESHOLD = fall_state_duration_threshold
        self.MIN_LANDMARK_CONFIDENCE = min_landmark_confidence
        
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
        if not landmarks or len(landmarks) < 33:
            self.reset()
            return False

        # Kiểm tra độ tin cậy của các điểm mốc quan trọng (vai và hông)
        relevant_landmarks = [
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.LEFT_HIP],
            landmarks[self.RIGHT_HIP],
        ]
        
        # Chỉ kiểm tra nếu các landmarks có độ tin cậy được cung cấp
        if all(len(lm) > 2 for lm in relevant_landmarks):
            if not all(lm[2] > self.MIN_LANDMARK_CONFIDENCE for lm in relevant_landmarks):
                self.reset()
                return False

        # Chuyển đổi sang định dạng (x, y) để tính toán
        landmark_coords = [(lm[0], lm[1]) for lm in landmarks]
        
        torso_angle = self._calculate_torso_angle(landmark_coords)
        vertical_velocity = self._calculate_vertical_velocity(landmark_coords)

        # Giai đoạn 1: Phát hiện chuyển động ngã
        is_falling = (torso_angle > self.TORSO_ANGLE_THRESHOLD and vertical_velocity > self.VELOCITY_THRESHOLD)
        
        # Giai đoạn 2: Phát hiện trạng thái nằm với ngưỡng mới
        is_lying_down = (torso_angle > self.TORSO_ANGLE_THRESHOLD + 10) # 70 độ

        if is_falling:
            self.consecutive_frames_fallen += 1
            self.frames_in_fall_state = 0
        elif is_lying_down:
            self.consecutive_frames_fallen = 0
            self.frames_in_fall_state += 1
        else:
            self.consecutive_frames_fallen = 0
            self.frames_in_fall_state = 0

        is_fall_detected = (
            self.consecutive_frames_fallen >= self.FALL_DURATION_THRESHOLD
            or self.frames_in_fall_state >= self.FALL_STATE_DURATION_THRESHOLD
        )
        
        return is_fall_detected
