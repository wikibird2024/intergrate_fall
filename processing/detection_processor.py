import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fall.fall_detector import FallDetector
from comm.ami_trigger import AMITrigger
from comm.telegram_bot import TelegramBot
from database.database_manager import insert_fall_event
from utils.draw_utils import draw_bounding_box, draw_skeleton


class DetectionProcessor:
    def __init__(self, ami_trigger: AMITrigger, telegram_bot: Optional[TelegramBot]):
        self.fall_detectors: Dict[str, FallDetector] = {}
        # Lưu trữ timestamp của lần cảnh báo cuối cùng cho từng thực thể
        self.last_alert_timestamps: Dict[str, datetime] = {}
        self.ami_trigger = ami_trigger
        self.telegram_bot = telegram_bot

    async def handle_camera_data(self, frame: np.ndarray, person_id: int, box: list, landmarks: list):
        """
        Xử lý dữ liệu từ luồng camera để phát hiện té ngã.
        Mỗi người được phát hiện có một ID riêng, ví dụ: 'camera_0_person_1'.
        """
        entity_id = f"camera_person_{person_id}"
        
        # Sử dụng FallDetector riêng cho mỗi đối tượng
        detector = self._get_fall_detector(entity_id)
        is_fall = detector.detect_fall(landmarks)

        # Vẽ hình ảnh trực quan trên khung hình
        status = 'fall' if is_fall else 'normal'
        draw_bounding_box(frame, box, entity_id, status)
        draw_skeleton(frame, landmarks)
        
        # Nếu phát hiện té ngã và đã hết thời gian chờ
        if is_fall and self._should_alert(entity_id):
            fall_event = {
                "timestamp": datetime.now().timestamp(),
                "source": "camera",
                "entity_id": entity_id,
                "fall_detected": True,
                "latitude": 0,  # Không có GPS từ camera
                "longitude": 0,
                "has_gps_fix": False
            }
            fall_id = insert_fall_event(fall_event)
            alert_msg = f"⚠️ Fall detected by camera for {entity_id}. Event ID: {fall_id}"
            print(alert_msg)

            await self.ami_trigger.alert_devices(alert_msg)
            if self.telegram_bot:
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                await self.telegram_bot.send_photo(img_bytes, alert_msg)
            
            self._update_alert_status(entity_id)

    async def handle_mqtt_data(self, mqtt_msg: Dict[str, Any]):
        """
        Xử lý dữ liệu từ luồng MQTT được gửi bởi thiết bị ESP32.
        Sử dụng 'device_id' từ tin nhắn làm ID thực thể.
        """
        entity_id = mqtt_msg.get('device_id')
        
        # Kiểm tra nếu tin nhắn có đầy đủ thông tin để xử lý
        if not entity_id or not mqtt_msg.get('fall_detected'):
            return

        # Kiểm tra thời gian chờ để tránh gửi cảnh báo lặp
        if self._should_alert(entity_id):
            fall_id = insert_fall_event({
                "timestamp": datetime.now().timestamp(),
                "source": "esp32",
                "entity_id": entity_id,
                "fall_detected": True,
                "latitude": mqtt_msg.get('latitude', 0),
                "longitude": mqtt_msg.get('longitude', 0),
                "has_gps_fix": mqtt_msg.get('has_gps_fix', False)
            })

            gps_info = f"{mqtt_msg.get('latitude', 'Unknown')}, {mqtt_msg.get('longitude', 'Unknown')}"
            alert_msg = f"🚨 Fall detected by {entity_id} at GPS: {gps_info}. Event ID: {fall_id}"
            print(alert_msg)
            
            await self.ami_trigger.alert_devices(alert_msg)
            if self.telegram_bot:
                await self.telegram_bot.send_message(alert_msg)

            self._update_alert_status(entity_id)
    
    def _get_fall_detector(self, entity_id: str) -> FallDetector:
        """Helper function to get or create a FallDetector instance for an entity."""
        if entity_id not in self.fall_detectors:
            self.fall_detectors[entity_id] = FallDetector()
        return self.fall_detectors[entity_id]

    def _should_alert(self, entity_id: str, cooldown_minutes: int = 5) -> bool:
        """
        Kiểm tra xem đã hết thời gian chờ để gửi cảnh báo mới hay chưa.
        Giúp tránh spam thông báo.
        """
        now = datetime.now()
        last_alert_time = self.last_alert_timestamps.get(entity_id)
        return last_alert_time is None or (now - last_alert_time) > timedelta(minutes=cooldown_minutes)

    def _update_alert_status(self, entity_id: str):
        """Cập nhật thời gian cảnh báo cuối cùng cho một thực thể."""
        self.last_alert_timestamps[entity_id] = datetime.now()
