import numpy as np

class FallDetector:
    def __init__(self):
        """Initialize fall detector."""
        self.fall_threshold = 0.3  # Example threshold for fall detection
        self.prev_positions = []

    def detect_fall(self, landmarks, mqtt_message):
        """
        Detect fall based on skeleton landmarks and MQTT message.
        Args:
            landmarks: List of landmarks [x, y, z, visibility].
            mqtt_message: MQTT message with fall/normal sign and GPS.
        Returns:
            Boolean indicating if a fall is detected.
        """
        if not landmarks or not mqtt_message:
            return False

        # Check MQTT message for fall indication
        if mqtt_message.get('status') == 'fall':
            return True

        # Analyze skeleton landmarks (e.g., rapid descent of hip landmarks)
        hip_indices = [23, 24]  # MediaPipe hip landmarks
        hip_positions = []
        for idx in hip_indices:
            if idx < len(landmarks):
                x, y, _, visibility = landmarks[idx]
                if visibility > 0.5:
                    hip_positions.append(y)

        if not hip_positions:
            return False

        avg_hip_y = np.mean(hip_positions)
        self.prev_positions.append(avg_hip_y)
        if len(self.prev_positions) > 10:
            self.prev_positions.pop(0)

        # Detect rapid descent
        if len(self.prev_positions) >= 2:
            delta_y = self.prev_positions[-2] - self.prev_positions[-1]
            if delta_y > self.fall_threshold:
                return True

        return False

