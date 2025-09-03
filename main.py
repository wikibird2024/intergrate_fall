# main.py
import cv2
import asyncio
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Import application modules
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from detection.person_tracker import PersonTracker
from processing.detection_processor import DetectionProcessor  # NEW import
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
from comm.telegram_bot import TelegramBot
from utils.draw_utils import draw_bounding_box, draw_skeleton
from database.database_manager import create_table, insert_fall_event

# Import all configuration variables
from config.config import (
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_TOPIC,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_KEEPALIVE,
    ENABLE_MQTT,
    AMI_HOST,
    AMI_PORT,
    AMI_USERNAME,
    AMI_SECRET,
    ENABLE_AMI,
    HUMAN_DETECTION_CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    POSE_MODEL_COMPLEXITY,            
    POSE_MIN_DETECTION_CONFIDENCE,    
    POSE_MIN_TRACKING_CONFIDENCE,
    ENABLE_TELEGRAM,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID
)

async def mqtt_listener_task(mqtt_client: MQTTClient, message_queue: asyncio.Queue):
    """Background task to receive messages and forward to main loop."""
    try:
        async for message in mqtt_client:
            if isinstance(message, dict):
                await message_queue.put(message)
    except asyncio.CancelledError:
        print("[MQTT] Listener task cancelled.")
    except Exception as e:
        print(f"[MQTT] Listener error: {e}")

async def main():
    """Main async function to run the fall detection system."""

    # 1. Initialize modules
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

    # NEW: Initialize the detection processor
    detection_processor = DetectionProcessor(ami_trigger, telegram_bot)

    mqtt_client = None
    mqtt_task = None
    mqtt_loop_task = None
    message_queue = asyncio.Queue()

    # 2. Initialize the database table
    create_table()

    # 3. Start MQTT (if enabled)
    if ENABLE_MQTT:
        try:
            mqtt_client = MQTTClient(
                broker=MQTT_BROKER_HOST,
                port=MQTT_BROKER_PORT,
                topic=MQTT_TOPIC,
                username=MQTT_USERNAME,
                password=MQTT_PASSWORD,
                keepalive=MQTT_KEEPALIVE,
            )
            mqtt_loop_task = asyncio.create_task(mqtt_client.run_forever())
            mqtt_task = asyncio.create_task(
                mqtt_listener_task(mqtt_client, message_queue)
            )
        except Exception as e:
            print(f"[MQTT] ⚠️ Could not start MQTT: {e}")
            mqtt_client = None

    # 4. Connect AMI trigger
    if ENABLE_AMI:
        try:
            await ami_trigger.connect()
        except ConnectionRefusedError as e:
            print(f"❌ Error connecting to AMI: {e}. Proceeding without AMI functionality.")
            ami_trigger.is_connected = False
        except Exception as e:
            print(f"❌ Unexpected error connecting to AMI: {e}. Proceeding without AMI functionality.")
            ami_trigger.is_connected = False
    else:
        print("[AMI] AMI functionality is disabled. Proceeding without AMI.")
        ami_trigger.is_connected = False

    # 5. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video capture.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Core processing loop
            detected_boxes = human_detector.detect_humans(frame)
            tracked_people = person_tracker.update(detected_boxes)
            
            mqtt_msg: Optional[Dict[str, Any]] = None
            if not message_queue.empty():
                mqtt_msg = message_queue.get_nowait()

            for person_id, box in tracked_people:
                landmarks = skeleton_tracker.track_from_box(frame, box)
                await detection_processor.process_person(frame, person_id, box, landmarks, mqtt_msg)

            cv2.imshow("Fall Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 1. Release camera and OpenCV resources
        try:
            cap.release()
            cv2.destroyAllWindows()
            skeleton_tracker.close()
        except Exception as e:
            print(f"[Cleanup] Error while releasing vision resources: {e}")

        # 2. Stop MQTT client and cancel related tasks
        if mqtt_client:
            try:
                await mqtt_client.stop()
            except Exception as e:
                print(f"[MQTT] Error while stopping client: {e}")

        for task, name in [(mqtt_task, "listener"), (mqtt_loop_task, "loop")]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    print(f"[MQTT] {name.capitalize()} task cancelled successfully.")
                except Exception as e:
                    print(f"[MQTT] Error while cancelling {name} task: {e}")

        # 3. Close AMI trigger if connected
        if ami_trigger and getattr(ami_trigger, "is_connected", False):
            try:
                await ami_trigger.close()
            except Exception as e:
                print(f"[AMI] Error while closing connection: {e}")

if __name__ == "__main__":
    asyncio.run(main())
