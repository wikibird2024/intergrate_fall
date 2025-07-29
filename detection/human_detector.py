import cv2
import numpy as np
from ultralytics import YOLO


class HumanDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Khởi tạo mô hình YOLOv8n để phát hiện người.

        Args:
            model_path (str): Đường dẫn đến file .pt của mô hình YOLOv8.
        """
        self.model = YOLO(model_path)
        self.class_id_person = 0  # Class ID của 'person' trong COCO dataset

    def detect_humans(self, frame):
        """
        Phát hiện người trong một frame ảnh.

        Args:
            frame (np.ndarray): Ảnh đầu vào dạng BGR (từ OpenCV).

        Returns:
            List[List[int|float]]: Danh sách bounding boxes người:
                [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model.predict(frame, verbose=False)[0]  # lấy kết quả đầu

        # Nếu không có box nào, trả về rỗng
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


import cv2
import numpy as np
from ultralytics import YOLO


class HumanDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Khởi tạo mô hình YOLOv8n để phát hiện người.

        Args:
            model_path (str): Đường dẫn đến file .pt của mô hình YOLOv8.
        """
        self.model = YOLO(model_path)
        self.class_id_person = 0  # Class ID của 'person' trong COCO dataset

    def detect_humans(self, frame):
        """
        Phát hiện người trong một frame ảnh.

        Args:
            frame (np.ndarray): Ảnh đầu vào dạng BGR (từ OpenCV).

        Returns:
            List[List[int|float]]: Danh sách bounding boxes người:
                [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model.predict(frame, verbose=False)[0]  # lấy kết quả đầu

        # Nếu không có box nào, trả về rỗng
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
