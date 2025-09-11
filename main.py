import cv2
import asyncio
from datetime import datetime
import json
from typing import Dict, Any, Optional
import time

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
    print("[MQTT] 🔄 Processor started")
    try:
        while True:
            mqtt_msg = await mqtt_client.get_message()
            print(f"[MQTT] 📨 Processing message: {mqtt_msg}")
            await detection_processor.handle_mqtt_data(mqtt_msg)
    except asyncio.CancelledError:
        print("[MQTT] Processor task cancelled.")
    except Exception as e:
        print(f"[MQTT] Processor error: {e}")
        raise


def init_camera_with_fallback():
    """Initialize camera with multiple fallback options."""
    camera_sources = [0, 1, 2, '/dev/video0', '/dev/video1']
    
    for source in camera_sources:
        print(f"[CAMERA] Trying camera source: {source}")
        cap = cv2.VideoCapture(source)
        
        if cap.isOpened():
            # Test if we can actually read frames
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[CAMERA] ✅ Successfully opened camera: {source}")
                return cap, True
            else:
                print(f"[CAMERA] ⚠️ Camera {source} opened but cannot read frames")
                cap.release()
        else:
            print(f"[CAMERA] ❌ Cannot open camera: {source}")
    
    print("[CAMERA] ⚠️ No working camera found, running in MQTT-only mode")
    return None, False


async def main():
    """Main async function to run the fall detection system."""
    
    # Global variables for cleanup
    mqtt_client = None
    mqtt_listener_task = None
    mqtt_processor_job = None
    cap = None
    skeleton_tracker = None
    ami_trigger = None

    try:
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

        # 2. Initialize the database table
        create_table()
        print("[DB] Database and table created successfully.")

        # 3. Start MQTT (if enabled) - MOVED BEFORE CAMERA
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
                
                # Start MQTT tasks
                mqtt_listener_task = asyncio.create_task(mqtt_client.run_forever())
                mqtt_processor_job = asyncio.create_task(
                    mqtt_processor_loop(mqtt_client, detection_processor)
                )
                
                # Give MQTT some time to connect
                print("[MQTT] ⏳ Waiting for MQTT connection...")
                await asyncio.sleep(2)
                print("[MQTT] ✅ MQTT tasks started successfully")
                
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

        # 5. Initialize camera (with fallback)
        print("[CAMERA] Opening video capture...")
        cap, camera_available = init_camera_with_fallback()

        # 6. Main processing loop
        if camera_available and cap:
            print("[SYSTEM] ✅ Running in CAMERA + MQTT mode")
            await run_camera_mode(cap, human_detector, skeleton_tracker, person_tracker, detection_processor)
        else:
            print("[SYSTEM] ✅ Running in MQTT-ONLY mode")
            await run_mqtt_only_mode()

    except KeyboardInterrupt:
        print("[SYSTEM] 🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"[SYSTEM] ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Comprehensive cleanup logic
        await cleanup_resources(cap, skeleton_tracker, mqtt_listener_task, mqtt_processor_job, ami_trigger)


async def run_camera_mode(cap, human_detector, skeleton_tracker, person_tracker, detection_processor):
    """Run main loop with camera processing."""
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] ⚠️ Failed to read frame, attempting to reconnect...")
                break

            # Process every frame (you can add frame skipping if needed)
            detected_boxes = human_detector.detect_humans(frame)
            tracked_people = person_tracker.update(detected_boxes)

            for person_id, box in tracked_people:
                landmarks = skeleton_tracker.track_from_box(frame, box)
                await detection_processor.handle_camera_data(frame, person_id, box, landmarks)

            cv2.imshow("Fall Detection", frame)
            
            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[CAMERA] User pressed 'q', exiting...")
                break
                
            frame_count += 1
            if frame_count % 100 == 0:  # Log every 100 frames
                print(f"[CAMERA] Processed {frame_count} frames")

    except Exception as e:
        print(f"[CAMERA] Error in camera mode: {e}")


async def run_mqtt_only_mode():
    """Run in MQTT-only mode when camera is not available."""
    print("[SYSTEM] 📡 MQTT-only mode active. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(10)  # Keep the program alive
            print("[SYSTEM] 💓 MQTT-only mode heartbeat")
    except KeyboardInterrupt:
        print("[SYSTEM] MQTT-only mode interrupted")


async def cleanup_resources(cap, skeleton_tracker, mqtt_listener_task, mqtt_processor_job, ami_trigger):
    """Comprehensive cleanup of all resources."""
    print("[SYSTEM] 🧹 Cleaning up resources...")

    # Cleanup camera
    if cap:
        print("[CAMERA] Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

    # Cleanup skeleton tracker
    if skeleton_tracker:
        print("[SKELETON] Closing skeleton tracker...")
        try:
            skeleton_tracker.close()
        except Exception as e:
            print(f"[SKELETON] Error closing: {e}")

    # Cleanup MQTT tasks
    cleanup_tasks = []
    
    if mqtt_listener_task and not mqtt_listener_task.done():
        print("[MQTT] Cancelling listener task...")
        mqtt_listener_task.cancel()
        cleanup_tasks.append(mqtt_listener_task)

    if mqtt_processor_job and not mqtt_processor_job.done():
        print("[MQTT] Cancelling processor task...")
        mqtt_processor_job.cancel()
        cleanup_tasks.append(mqtt_processor_job)

    # Wait for all tasks to complete
    if cleanup_tasks:
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            print("[MQTT] All MQTT tasks cleaned up successfully.")
        except Exception as e:
            print(f"[MQTT] Error during cleanup: {e}")

    # Cleanup AMI
    if ami_trigger and getattr(ami_trigger, "is_connected", False):
        print("[AMI] Closing AMI connection...")
        try:
            await ami_trigger.close()
        except Exception as e:
            print(f"[AMI] Error closing: {e}")

    print("[SYSTEM] ✅ Cleanup completed")


if __name__ == "__main__":
    print("[SYSTEM] 🚀 Starting Fall Detection System...")
    asyncio.run(main())
