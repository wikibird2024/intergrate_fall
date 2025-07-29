import cv2
import mediapipe as mp
import numpy as np


class SkeletonTracker:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def track_from_box(self, frame, box):
        x1, y1, x2, y2 = map(int, box[:4])
        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(crop_rgb)
        if not results.pose_landmarks:
            return []

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            lx = int(lm.x * (x2 - x1)) + x1
            ly = int(lm.y * (y2 - y1)) + y1
            lz = lm.z
            visibility = lm.visibility
            landmarks.append([lx, ly, lz, visibility])

        return landmarks

    def close(self):
        self.pose.close()


import cv2
import mediapipe as mp
import numpy as np


class SkeletonTracker:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def track_from_box(self, frame, box):
        x1, y1, x2, y2 = map(int, box[:4])
        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(crop_rgb)
        if not results.pose_landmarks:
            return []

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            lx = int(lm.x * (x2 - x1)) + x1
            ly = int(lm.y * (y2 - y1)) + y1
            lz = lm.z
            visibility = lm.visibility
            landmarks.append([lx, ly, lz, visibility])

        return landmarks

    def close(self):
        self.pose.close()
