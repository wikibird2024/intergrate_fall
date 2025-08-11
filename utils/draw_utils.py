import cv2
import numpy as np
import mediapipe as mp
from config.config import (
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    SKELETON_LINE_COLOR,
    SKELETON_LINE_THICKNESS,
    SKELETON_POINT_COLOR,
    SKELETON_POINT_RADIUS
)

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


def draw_bounding_box(frame, box, color=BOUNDING_BOX_COLOR, thickness=BOUNDING_BOX_THICKNESS):
    """
    Vẽ hộp bao quanh người.

    Args:
        frame: Ảnh gốc.
        box: List hoặc tuple (x1, y1, x2, y2, conf, class_id).
        color: Màu hộp vẽ (BGR).
        thickness: Độ dày đường viền.
    """
    x1, y1, x2, y2, conf, cls_id = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        f"Conf: {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_skeleton(frame, landmarks, visibility_th=0.5,
                  line_color=SKELETON_LINE_COLOR, line_thickness=SKELETON_LINE_THICKNESS,
                  point_color=SKELETON_POINT_COLOR, point_radius=SKELETON_POINT_RADIUS):
    """
    Vẽ bộ xương người dựa trên danh sách landmarks dạng [x, y, z, visibility].

    Args:
        frame: Ảnh đầu vào (numpy array).
        landmarks: List các landmark dạng [x, y, z, visibility].
        visibility_th: Ngưỡng visibility để vẽ.
        line_color: Màu đường kẻ bộ xương (BGR).
        line_thickness: Độ dày đường kẻ bộ xương.
        point_color: Màu điểm khớp (BGR).
        point_radius: Bán kính điểm khớp.
    """
    if not landmarks:
        return

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            if start[3] > visibility_th and end[3] > visibility_th:
                x1, y1 = int(start[0]), int(start[1])
                x2, y2 = int(end[0]), int(end[1])
                cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

    for lm in landmarks:
        if lm[3] > visibility_th:
            cx, cy = int(lm[0]), int(lm[1])
            cv2.circle(frame, (cx, cy), point_radius, point_color, -1)
