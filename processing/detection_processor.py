
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio

from fall.fall_detector import FallDetector
from comm.ami_trigger import AMITrigger
from comm.telegram_bot import TelegramBot
from database.database_manager import insert_fall_event
from utils.draw_utils import draw_person


class DetectionProcessor:
    def __init__(self, ami_trigger: AMITrigger, telegram_bot: Optional[TelegramBot]):
        self.fall_detectors: Dict[str, FallDetector] = {}
        self.last_alert_timestamps: Dict[str, datetime] = {}
        self.ami_trigger = ami_trigger
        self.telegram_bot = telegram_bot

    async def handle_camera_data(self, frame: Optional[np.ndarray], person_id: int, box: list, landmarks: list):
        """Xử lý frame từ camera, phát hiện fall và gửi cảnh báo."""
        entity_id = f"camera_person_{person_id}"
        detector = self._get_fall_detector(entity_id)
        is_fall = detector.detect_fall(landmarks)

        # Vẽ bounding box + skeleton
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            status = "fall" if is_fall else "normal"
            draw_person(frame, box, landmarks, entity_id, status)

        if is_fall and self._should_alert(entity_id):
            fall_event = {
                "timestamp": datetime.now().timestamp(),
                "source": "camera",
                "entity_id": entity_id,
                "fall_detected": True,
                "latitude": 0,
                "longitude": 0,
                "has_gps_fix": False,
            }
            fall_id = insert_fall_event(fall_event)
            alert_msg = f"⚠️ Fall detected by camera for {entity_id}. Event ID: {fall_id}"
            print(alert_msg)

            await self.ami_trigger.alert_devices(alert_msg)
            await self._safe_send_telegram(frame, alert_msg)
            self._update_alert_status(entity_id)

    async def handle_mqtt_data(self, mqtt_msg: Dict[str, Any]):
        """Xử lý dữ liệu từ ESP32 gửi qua MQTT."""
        entity_id = mqtt_msg.get("device_id")
        if not entity_id or not mqtt_msg.get("fall_detected"):
            return

        if self._should_alert(entity_id):
            fall_id = insert_fall_event({
                "timestamp": datetime.now().timestamp(),
                "source": "esp32",
                "entity_id": entity_id,
                "fall_detected": True,
                "latitude": mqtt_msg.get("latitude", 0),
                "longitude": mqtt_msg.get("longitude", 0),
                "has_gps_fix": mqtt_msg.get("has_gps_fix", False),
            })

            gps_info = f"{mqtt_msg.get('latitude', 'Unknown')}, {mqtt_msg.get('longitude', 'Unknown')}"
            alert_msg = f"🚨 Fall detected by {entity_id} at GPS: {gps_info}. Event ID: {fall_id}"
            print(alert_msg)

            await self.ami_trigger.alert_devices(alert_msg)
            await self._safe_send_telegram(None, alert_msg)  # fallback text-only
            self._update_alert_status(entity_id)

    async def _safe_send_telegram(
        self, frame: Optional[np.ndarray], msg: str, retries: int = 3, delay: float = 1.0
    ):
        """Gửi ảnh Telegram an toàn với retry, fallback text nếu frame invalid."""
        if not self.telegram_bot:
            return

        for attempt in range(retries):
            try:
                # Debug thông tin frame
                if frame is not None and isinstance(frame, np.ndarray):
                    print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}, size: {frame.size}")

                if isinstance(frame, np.ndarray) and frame.size > 0:
                    frame_safe = self._prepare_frame(frame)
                    if frame_safe.size > 0:
                        success, img_encoded = cv2.imencode(".jpg", frame_safe)
                        if success:
                            await self.telegram_bot.send_photo(img_encoded.tobytes(), msg)
                            return  # Gửi thành công
                # Fallback text
                await self.telegram_bot.send_message(msg)
                return
            except Exception as e:
                print(f"[TELEGRAM] ❌ Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(delay)

        # Sau retries vẫn fail -> gửi text cuối cùng
        try:
            await self.telegram_bot.send_message(msg)
        except Exception as e:
            print(f"[TELEGRAM] ❌ Failed sending fallback text: {e}")

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        """Chuẩn hóa frame: uint8, 3 kênh (BGR), tránh empty frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.array([], dtype=np.uint8)

        frame_safe = cv2.convertScaleAbs(frame) if frame.dtype != np.uint8 else frame.copy()

        if frame_safe.ndim == 2:
            # Gray -> BGR
            frame_safe = cv2.cvtColor(frame_safe, cv2.COLOR_GRAY2BGR)
        elif frame_safe.ndim == 3 and frame_safe.shape[2] == 4:
            # BGRA -> BGR
            frame_safe = cv2.cvtColor(frame_safe, cv2.COLOR_BGRA2BGR)

        return frame_safe

    def _get_fall_detector(self, entity_id: str) -> FallDetector:
        """Lấy hoặc tạo FallDetector cho một entity."""
        if entity_id not in self.fall_detectors:
            self.fall_detectors[entity_id] = FallDetector()
        return self.fall_detectors[entity_id]

    def _should_alert(self, entity_id: str, cooldown_minutes: int = 5) -> bool:
        """Kiểm tra cooldown trước khi gửi cảnh báo mới."""
        now = datetime.now()
        last_alert_time = self.last_alert_timestamps.get(entity_id)
        return last_alert_time is None or (now - last_alert_time) > timedelta(minutes=cooldown_minutes)

    def _update_alert_status(self, entity_id: str):
        """Cập nhật thời gian cảnh báo cuối cùng."""
        self.last_alert_timestamps[entity_id] = datetime.now()
