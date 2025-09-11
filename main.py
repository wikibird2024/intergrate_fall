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


async def camera_processing_loop(human_detector, skeleton_tracker, person_tracker, detection_processor):
    """Camera processing loop that can be cancelled."""
    cap = None
    
    try:
        # Initialize camera with fallback
        camera_sources = [0, 1, 2, '/dev/video0', '/dev/video1']
        
        for source in camera_sources:
            print(f"[CAMERA] Trying camera source: {source}")
            cap = cv2.VideoCapture(source)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[CAMERA] ✅ Successfully opened camera: {source}")
                    break
                else:
                    print(f"[CAMERA] ⚠️ Camera {source} opened but cannot read frames")
                    cap.release()
                    cap = None
            else:
                print(f"[CAMERA] ❌ Cannot open camera: {source}")
                cap = None
        
        if not cap:
            print("[CAMERA] ⚠️ No working camera found, camera loop will exit")
            return
        
        # Main camera processing loop
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] ⚠️ Failed to read frame")
                break

            # Process frame
            detected_boxes = human_detector.detect_humans(frame)
            tracked_people = person_tracker.update(detected_boxes)

            for person_id, box in tracked_people:
                landmarks = skeleton_tracker.track_from_box(frame, box)
                await detection_processor.handle_camera_data(frame, person_id, box, landmarks)

            # Display frame
            cv2.imshow("Fall Detection", frame)
            
            # Non-blocking key check
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[CAMERA] User pressed 'q', stopping camera processing...")
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[CAMERA] Processed {frame_count} frames")
                
            # Allow other tasks to run
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        print("[CAMERA] Camera processing task cancelled")
    except Exception as e:
        print(f"[CAMERA] Error in camera processing: {e}")
    finally:
        if cap:
            print("[CAMERA] Releasing camera...")
            cap.release()
            cv2.destroyAllWindows()


async def heartbeat_loop():
    """Heartbeat loop to keep program alive and show it's working."""
    try:
        while True:
            await asyncio.sleep(60)  # Every 30 seconds
            print("[SYSTEM] 💓 System heartbeat - all tasks running")
    except asyncio.CancelledError:
        print("[SYSTEM] Heartbeat cancelled")


async def main():
    """Main async function to run the fall detection system."""
    
    # Global variables for cleanup
    mqtt_client = None
    skeleton_tracker = None
    ami_trigger = None
    tasks = []

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

        # 3. Start MQTT tasks (if enabled)
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
                
                tasks.append(asyncio.create_task(mqtt_client.run_forever(), name="mqtt_listener"))
                print("[MQTT] ✅ MQTT listener task created")

            except Exception as e:
                print(f"[MQTT] ⚠️ Could not start MQTT: {e}")
                mqtt_client = None
        else:
            print("[MQTT] Disabled by configuration.")

        # Wait for MQTT connection before starting the processor
        if mqtt_client:
            print("[MQTT] ⏳ Waiting for MQTT client to connect...")
            try:
                await asyncio.wait_for(mqtt_client.connected_event.wait(), timeout=10)
                print("[MQTT] ✅ MQTT client connected.")
                tasks.append(asyncio.create_task(mqtt_processor_loop(mqtt_client, detection_processor), name="mqtt_processor"))
                print("[MQTT] ✅ MQTT processor task created")
            except asyncio.TimeoutError:
                print("[MQTT] ❌ Timeout waiting for MQTT connection. Proceeding without MQTT processing.")

        # 4. Connect AMI trigger
        if ENABLE_AMI:
            try:
                await ami_trigger.connect()
            except Exception as e:
                print(f"[AMI] ❌ Error connecting to AMI: {e}. Proceeding without AMI functionality.")
                ami_trigger.is_connected = False
        else:
            print("[AMI] AMI functionality is disabled.")
            ami_trigger.is_connected = False

        # 5. Start camera processing task
        print("[CAMERA] Starting camera processing task...")
        camera_task = asyncio.create_task(
            camera_processing_loop(human_detector, skeleton_tracker, person_tracker, detection_processor),
            name="camera_processing"
        )
        tasks.append(camera_task)

        # 6. Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_loop(), name="heartbeat")
        tasks.append(heartbeat_task)

        # 7. Wait for any task to complete or fail
        print(f"[SYSTEM] ✅ Starting {len(tasks)} concurrent tasks...")
        print("[SYSTEM] 📌 Tasks:", [task.get_name() for task in tasks])
        print("[SYSTEM] Press Ctrl+C to shutdown gracefully")
        
        done, pending = await asyncio.wait(
            tasks, 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Check what completed
        for task in done:
            if task.exception():
                print(f"[SYSTEM] ❌ Task {task.get_name()} failed with exception: {task.exception()}")
            else:
                print(f"[SYSTEM] ✅ Task {task.get_name()} completed normally.")

        # Cancel remaining tasks
        print(f"[SYSTEM] 🛑 Cancelling {len(pending)} remaining tasks...")
        for task in pending:
            task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending, timeout=5.0)

    except KeyboardInterrupt:
        print("[SYSTEM] 🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"[SYSTEM] ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        await cleanup_resources(skeleton_tracker, ami_trigger)


async def mqtt_processor_loop(mqtt_client: MQTTClient, detection_processor: DetectionProcessor):
    """
    Background task to process MQTT messages from the queue.
    This version correctly uses the MQTTClient's is_running flag.
    """
    print("[MQTT] 🔄 Processor started")
    try:
        while mqtt_client.is_running():
            try:
                mqtt_msg = await mqtt_client.get_message(timeout=1.0)
                print(f"[MQTT] 📨 Processing message: {mqtt_msg}")
                await detection_processor.handle_mqtt_data(mqtt_msg)
            except asyncio.TimeoutError:
                # No message received in the last second, continue the loop
                pass
    except asyncio.CancelledError:
        print("[MQTT] Processor task cancelled.")
    except Exception as e:
        print(f"[MQTT] Processor error: {e}")
        raise


async def cleanup_resources(skeleton_tracker, ami_trigger):
    """Final cleanup of resources that need explicit closing."""
    print("[SYSTEM] 🧹 Final cleanup...")

    # Cleanup skeleton tracker
    if skeleton_tracker:
        try:
            skeleton_tracker.close()
            print("[SKELETON] ✅ Skeleton tracker closed")
        except Exception as e:
            print(f"[SKELETON] ⚠️ Error closing: {e}")

    # Cleanup AMI
    if ami_trigger and getattr(ami_trigger, "is_connected", False):
        try:
            await ami_trigger.close()
            print("[AMI] ✅ AMI connection closed")
        except Exception as e:
            print(f"[AMI] ⚠️ Error closing: {e}")

    print("[SYSTEM] ✅ Cleanup completed")


if __name__ == "__main__":
    print("[SYSTEM] 🚀 Starting Fall Detection System...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[SYSTEM] 🛑 Program interrupted by user")
    finally:
        print("[SYSTEM] 👋 Goodbye!")
