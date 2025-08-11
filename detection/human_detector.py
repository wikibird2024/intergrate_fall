# human_detector.py
import cv2
import numpy as np
from ultralytics import YOLO


class HumanDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.class_id_person = 0

    # Add the 'conf_threshold' and 'iou_threshold' parameters here
    def detect_humans(self, frame, conf_threshold, iou_threshold):
        """
        Detects humans in a video frame with adjustable confidence and IoU thresholds.
        """
        results = self.model.predict(
            frame, conf=conf_threshold, iou=iou_threshold, verbose=False
        )[0]

        if results.boxes is None or results.boxes.xyxy is None:
            return []

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        person_boxes = []
        for box, conf, cls in zip(boxes, confs, clss):
            if cls == self.class_id_person:
                x1, y1, x2, y2 = map(int, box)
                person_boxes.append([x1, y1, x2, y2, float(conf), cls])

        return person_boxes
