
import cv2
import asyncio
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Import application modules
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from detection.person_tracker import PersonTracker
from processing.detection_processor import DetectionProcessor
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
from comm.telegram_bot import TelegramBot
from utils.draw_utils import draw_bounding_box, draw_skeleton
from database.database_manager import create_table

# Import all configuration variables
from config.config import (
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC, MQTT_USERNAME, MQTT_PASSWORD, ENABLE_MQTT,
    AMI_HOST, AMI_PORT, AMI_USERNAME, AMI_SECRET, ENABLE_AMI,
    HUMAN_DETECTION_CONFIDENCE_THRESHOLD, IOU_THRESHOLD, POSE_MODEL_COMPLEXITY, POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE, ENABLE_TELEGRAM, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)


async def mqtt_processor_loop(mqtt_client: MQTTClient, detection_processor: DetectionProcessor):
    """
    Background task to process MQTT messages from the queue.
    """
    while True:
        try:
            mqtt_msg = await mqtt_client.get_message()
            await detection_processor.handle_mqtt_data(mqtt_msg)
        except asyncio.CancelledError:
            print("[MQTT] Processor task cancelled.")
            break
        except Exception as e:
            print(f"[MQTT] Processor error: {e}")


async def main():
    """Main async function to run the fall detection system."""

    # 1. Initialize modules
    print("[SYSTEM] Initializing modules...")
    human_detector = HumanDetector(
        conf_threshold=HUMAN_DETECTION_CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    skeleton_tracker = SkeletonTracker(
        model_complexity=POSE_MODEL_COMPLEXITY,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE
    )
    person_tracker = PersonTracker(iou_threshold=IOU_THRESHOLD)
    ami_trigger = AMITrigger(
        host=AMI_HOST, port=AMI_PORT, username=AMI_USERNAME, secret=AMI_SECRET
    )

    telegram_bot = None
    if ENABLE_TELEGRAM:
        telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        print("[TELEGRAM] Bot initialized.")

    detection_processor = DetectionProcessor(ami_trigger, telegram_bot)

    mqtt_client = None
    mqtt_listener_task = None
    mqtt_processor_job = None

    # 2. Initialize the database table
    create_table()
    print("[DB] Database and table created successfully.")

    # 3. Start MQTT (if enabled)
    if ENABLE_MQTT:
        try:
            print(f"[MQTT] 🔄 Starting MQTT client for {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}, topic={MQTT_TOPIC}")
            mqtt_client = MQTTClient(
                broker=MQTT_BROKER_HOST,
                port=MQTT_BROKER_PORT,
                topic=MQTT_TOPIC,
                username=MQTT_USERNAME,
                password=MQTT_PASSWORD,
            )
            mqtt_listener_task = asyncio.create_task(mqtt_client.run_forever())
            mqtt_processor_job = asyncio.create_task(
                mqtt_processor_loop(mqtt_client, detection_processor)
            )
        except Exception as e:
            print(f"[MQTT] ⚠️ Could not start MQTT: {e}")
            mqtt_client = None
    else:
        print("[MQTT] Disabled by configuration.")

    # 4. Connect AMI trigger
    if ENABLE_AMI:
        try:
            await ami_trigger.connect()
        except Exception as e:
            print(f"[AMI] ❌ Error connecting to AMI: {e}. Proceeding without AMI functionality.")
            ami_trigger.is_connected = False
    else:
        print("[AMI] AMI functionality is disabled. Proceeding without AMI.")
        ami_trigger.is_connected = False

    # 5. Open webcam
    print("[CAMERA] Opening video capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video capture.")
        return

    try:
        while True:
            # Handle Camera Data
            ret, frame = cap.read()
            if not ret:
                break

            detected_boxes = human_detector.detect_humans(frame)
            tracked_people = person_tracker.update(detected_boxes)

            for person_id, box in tracked_people:
                landmarks = skeleton_tracker.track_from_box(frame, box)
                await detection_processor.handle_camera_data(frame, person_id, box, landmarks)

            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup logic
        print("[SYSTEM] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        skeleton_tracker.close()

        # Cleanup MQTT tasks
        if mqtt_listener_task:
            mqtt_listener_task.cancel()
            try:
                await mqtt_listener_task
            except asyncio.CancelledError:
                print("[MQTT] Listener task cancelled successfully.")

        if mqtt_processor_job:
            mqtt_processor_job.cancel()
            try:
                await mqtt_processor_job
            except asyncio.CancelledError:
                print("[MQTT] Processor task cancelled successfully.")

        if ami_trigger and getattr(ami_trigger, "is_connected", False):
            await ami_trigger.close()


if __name__ == "__main__":
    asyncio.run(main())
