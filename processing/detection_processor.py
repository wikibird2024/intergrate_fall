import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio
import logging
from telegram.error import TelegramError, RetryAfter, TimedOut, NetworkError

from fall.fall_detector import FallDetector
from comm.ami_trigger import AMITrigger
from comm.telegram_bot import TelegramBot
from database.database_manager import insert_fall_event
from utils.draw_utils import draw_person

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Giới hạn số yêu cầu Telegram đồng thời
telegram_semaphore = asyncio.Semaphore(5)  # Giới hạn 5 yêu cầu đồng thời

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
            logger.info(alert_msg)

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
            logger.info(alert_msg)

            await self.ami_trigger.alert_devices(alert_msg)
            await self._safe_send_telegram(None, alert_msg)  # fallback text-only
            self._update_alert_status(entity_id)

    async def _safe_send_telegram(
        self, frame: Optional[np.ndarray], msg: str, retries: int = 3, delay: float = 2.0
    ):
        """Gửi ảnh Telegram an toàn với retry, fallback text nếu frame invalid."""
        if not self.telegram_bot:
            logger.warning("[TELEGRAM] TelegramBot not initialized, skipping send")
            return

        async with telegram_semaphore:  # Giới hạn số yêu cầu đồng thời
            for attempt in range(retries):
                try:
                    if isinstance(frame, np.ndarray) and frame.size > 0 and frame.shape[0] >= 10 and frame.shape[1] >= 10:
                        logger.debug(f"[TELEGRAM] Frame shape: {frame.shape}, dtype: {frame.dtype}, size: {frame.size}")
                        # Thu nhỏ frame nếu quá lớn (giữ tỷ lệ khung hình)
                        if max(frame.shape[0], frame.shape[1]) > 1080:
                            scale = 1080 / max(frame.shape[0], frame.shape[1])
                            frame = cv2.resize(frame, None, fx=scale, fy=scale)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Giảm chất lượng JPEG
                        success, img_encoded = cv2.imencode(".jpg", frame, encode_param)
                        if success:
                            file_size_mb = len(img_encoded) / (1024 * 1024)
                            logger.debug(f"[TELEGRAM] Encoded image size: {file_size_mb:.2f} MB")
                            if file_size_mb <= 10:
                                await self.telegram_bot.send_photo(frame, msg)
                                return
                    logger.warning("[TELEGRAM] ⚠️ Frame invalid or too large, sending text only")
                    await self.telegram_bot.send_message(msg)
                    return
                except (RetryAfter, TimedOut, NetworkError) as e:
                    logger.error(f"[TELEGRAM] ❌ Attempt {attempt + 1} failed (network issue): {e}")
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                except TelegramError as e:
                    logger.error(f"[TELEGRAM] ❌ Attempt {attempt + 1} failed (Telegram error): {e}")
                    await asyncio.sleep(delay * (2 ** attempt))
                except Exception as e:
                    logger.error(f"[TELEGRAM] ❌ Attempt {attempt + 1} failed (unexpected): {e}")
                    await asyncio.sleep(delay * (2 ** attempt))

        logger.warning("[TELEGRAM] ⚠️ All attempts failed, sending fallback text")
        try:
            await self.telegram_bot.send_message(msg)
        except TelegramError as e:
            logger.error(f"[TELEGRAM] ❌ Failed sending fallback text (Telegram error): {e}")
        except Exception as e:
            logger.error(f"[TELEGRAM] ❌ Failed sending fallback text (unexpected): {e}")

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
