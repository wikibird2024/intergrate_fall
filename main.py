import cv2
import asyncio
from datetime import datetime
import json
from typing import Dict, Any

# Import application modules
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from detection.person_tracker import PersonTracker # NEW import
from fall.fall_detector import FallDetector
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
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
    POSE_MIN_TRACKING_CONFIDENCE      
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
    
    # NEW: Initialize a person tracker and dictionaries for state management
    person_tracker = PersonTracker(iou_threshold=IOU_THRESHOLD)
    fall_detectors: Dict[int, FallDetector] = {}
    person_status: Dict[int, str] = {}
    
    ami_trigger = AMITrigger(
        host=AMI_HOST, port=AMI_PORT, username=AMI_USERNAME, secret=AMI_SECRET
    )

    mqtt_client = None
    mqtt_task = None
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
            asyncio.create_task(mqtt_client.run_forever())
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

            # 1. Detect all humans and get tracked IDs
            detected_boxes = human_detector.detect_humans(frame)
            tracked_people = person_tracker.update(detected_boxes)
            
            mqtt_msg: Optional[Dict[str, Any]] = None
            if not message_queue.empty():
                mqtt_msg = message_queue.get_nowait()

            # 2. Iterate through each tracked person
            for person_id, box in tracked_people:
                # Get or create a FallDetector for this person
                if person_id not in fall_detectors:
                    fall_detectors[person_id] = FallDetector()
                    person_status[person_id] = 'normal' # Initialize new person status
                
                person_detector = fall_detectors[person_id]
                
                landmarks = skeleton_tracker.track_from_box(frame, box)
                
                is_fall = False
                if landmarks:
                    # Check for MQTT message specific to this person's device
                    mqtt_status_for_person = None
                    if mqtt_msg and mqtt_msg.get('device_id') == f"ESP32_DEV_{person_id}":
                        mqtt_status_for_person = mqtt_msg
                        
                    is_fall = person_detector.detect_fall(landmarks, mqtt_status_for_person)

                # 3. Update status and draw visuals
                if is_fall:
                    person_status[person_id] = 'fall'
                elif landmarks: # Only update to normal if they are visible
                    person_status[person_id] = 'normal'

                draw_bounding_box(frame, box, person_id, person_status.get(person_id, 'normal'))
                if landmarks:
                    draw_skeleton(frame, landmarks)
                
                # 4. Handle fall events and alerts
                if is_fall:
                    if mqtt_msg and mqtt_msg.get('device_id') == f"ESP32_DEV_{person_id}":
                        fall_id = insert_fall_event(mqtt_msg)
                        gps_info = f"{mqtt_msg.get('latitude', 'Unknown')}, {mqtt_msg.get('longitude', 'Unknown')}"
                        alert_msg = f"Fall detected at GPS: {gps_info}. Event ID: {fall_id}"
                    else:
                        fall_id = insert_fall_event({
                            "timestamp": datetime.now().timestamp(),
                            "device_id": f"Camera_{person_id}", # Use the person ID for unique events
                            "fall_detected": True,
                            "latitude": 0,
                            "longitude": 0,
                            "has_gps_fix": False
                        })
                        alert_msg = f"Fall detected via camera. No MQTT data. Event ID: {fall_id}"

                    print("⚠️ " + alert_msg)
                    await ami_trigger.alert_devices(alert_msg)

            cv2.imshow("Fall Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        skeleton_tracker.close()

        if mqtt_client:
            await mqtt_client.stop()

        if mqtt_task:
            mqtt_task.cancel()
            try:
                await mqtt_task
            except asyncio.CancelledError:
                pass

        await ami_trigger.close()


if __name__ == "__main__":
    asyncio.run(main())
