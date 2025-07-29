import numpy as np
from typing import List, Dict, Any, Optional


class FallDetector:
    def __init__(self, fall_threshold: float = 0.3, history_size: int = 10):
        """
        FallDetector using skeleton landmarks and external status input.

        Args:
            fall_threshold (float): Threshold to determine sudden drop in hip position.
            history_size (int): Number of previous hip positions to store.
        """
        self.fall_threshold = fall_threshold
        self.history_size = history_size
        self.prev_positions: List[float] = []

    def detect_fall(
        self,
        landmarks: Optional[List[List[float]]],
        mqtt_message: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Determine if a fall has occurred.

        Args:
            landmarks: List of [x, y, z, visibility] for each skeleton joint.
            mqtt_message: Dictionary with keys like 'status', 'gps', etc.

        Returns:
            True if a fall is detected, else False.
        """
        if not landmarks or not mqtt_message:
            return False

        # Priority: MQTT message status
        if mqtt_message.get("status", "").lower() == "fall":
            return True

        # Check hip landmarks for sudden descent
        hip_indices = [23, 24]  # MediaPipe left and right hips
        visible_hips = [
            landmarks[idx][1]
            for idx in hip_indices
            if idx < len(landmarks) and landmarks[idx][3] > 0.5
        ]

        if not visible_hips:
            return False

        avg_hip_y = float(np.mean(visible_hips))
        self.prev_positions.append(avg_hip_y)
        if len(self.prev_positions) > self.history_size:
            self.prev_positions.pop(0)

        # Detect sudden downward movement
        if len(self.prev_positions) >= 2:
            delta_y = self.prev_positions[-2] - self.prev_positions[-1]
            if delta_y > self.fall_threshold:
                return True

        return False

    def reset(self):
        """Clear stored position history."""
        self.prev_positions.clear()
