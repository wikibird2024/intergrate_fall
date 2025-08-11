# processing/detection_processor.py
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from fall.fall_detector import FallDetector
from comm.ami_trigger import AMITrigger
from comm.telegram_bot import TelegramBot
from database.database_manager import insert_fall_event
from utils.draw_utils import draw_bounding_box, draw_skeleton


class DetectionProcessor:
    def __init__(self, ami_trigger: AMITrigger, telegram_bot: Optional[TelegramBot]):
        self.fall_detectors: Dict[int, FallDetector] = {}
        self.person_status: Dict[int, str] = {}
        self.ami_trigger = ami_trigger
        self.telegram_bot = telegram_bot

    async def process_person(self, frame: np.ndarray, person_id: int, box: list, landmarks: Optional[list], mqtt_msg: Optional[Dict[str, Any]]):
        """
        Processes a single person, handling fall detection, status, and alerts.
        """
        if person_id not in self.fall_detectors:
            self.fall_detectors[person_id] = FallDetector()
            self.person_status[person_id] = 'normal'

        person_detector = self.fall_detectors[person_id]
        is_fall = False
        
        if landmarks:
            # Check for MQTT message specific to this person's device
            mqtt_status_for_person = None
            if mqtt_msg and mqtt_msg.get('device_id') == f"ESP32_DEV_{person_id}":
                mqtt_status_for_person = mqtt_msg
            
            is_fall = person_detector.detect_fall(landmarks, mqtt_status_for_person)

        # Update status
        if is_fall:
            self.person_status[person_id] = 'fall'
        elif landmarks:
            self.person_status[person_id] = 'normal'

        # Draw visuals
        draw_bounding_box(frame, box, person_id, self.person_status.get(person_id, 'normal'))
        if landmarks:
            draw_skeleton(frame, landmarks)
            
        # Handle fall events and alerts
        if is_fall:
            if mqtt_msg and mqtt_msg.get('device_id') == f"ESP32_DEV_{person_id}":
                fall_id = insert_fall_event(mqtt_msg)
                gps_info = f"{mqtt_msg.get('latitude', 'Unknown')}, {mqtt_msg.get('longitude', 'Unknown')}"
                alert_msg = f"Fall detected at GPS: {gps_info}. Event ID: {fall_id}"
            else:
                fall_id = insert_fall_event({
                    "timestamp": datetime.now().timestamp(),
                    "device_id": f"Camera_{person_id}",
                    "fall_detected": True,
                    "latitude": 0,
                    "longitude": 0,
                    "has_gps_fix": False
                })
                alert_msg = f"Fall detected via camera. No MQTT data. Event ID: {fall_id}"

            print("⚠️ " + alert_msg)
            await self.ami_trigger.alert_devices(alert_msg)
            
            if self.telegram_bot:
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                await self.telegram_bot.send_photo(img_bytes, alert_msg)
