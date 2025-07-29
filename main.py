import cv2
import asyncio
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from fall.fall_detector import FallDetector
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
from utils.draw_utils import draw_bounding_box, draw_skeleton

ENABLE_MQTT = False  # Set True if you want to enable MQTT


async def main():
    # Initialize components
    human_detector = HumanDetector()
    skeleton_tracker = SkeletonTracker()
    fall_detector = FallDetector()
    mqtt_client = MQTTClient() if ENABLE_MQTT else None
    ami_trigger = AMITrigger()

    if ENABLE_MQTT:
        mqtt_client.start()
    await ami_trigger.connect()

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = human_detector.detect_humans(frame)

            for box in boxes:
                draw_bounding_box(frame, box)

                # Get skeleton landmarks (adjusted to global coordinates internally)
                landmarks = skeleton_tracker.track_from_box(frame, box)

                if landmarks:
                    draw_skeleton(frame, landmarks)

                    # Fall detection
                    mqtt_msg = mqtt_client.get_latest_message() if ENABLE_MQTT else None
                    if fall_detector.detect_fall(landmarks, mqtt_msg):
                        gps = mqtt_msg.get("gps", "Unknown") if mqtt_msg else "Unknown"
                        await ami_trigger.alert_devices(f"Fall detected at GPS: {gps}")
                        print("⚠️ Fall detected! Alert sent.")

            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        skeleton_tracker.close()
        if ENABLE_MQTT:
            mqtt_client.stop()
        await ami_trigger.close()


if __name__ == "__main__":
    asyncio.run(main())
import cv2
import asyncio
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from fall.fall_detector import FallDetector
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
from utils.draw_utils import draw_bounding_box, draw_skeleton

ENABLE_MQTT = False  # Set True if you want to enable MQTT


async def main():
    # Initialize components
    human_detector = HumanDetector()
    skeleton_tracker = SkeletonTracker()
    fall_detector = FallDetector()
    mqtt_client = MQTTClient() if ENABLE_MQTT else None
    ami_trigger = AMITrigger()

    if ENABLE_MQTT:
        mqtt_client.start()
    await ami_trigger.connect()

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = human_detector.detect_humans(frame)

            for box in boxes:
                draw_bounding_box(frame, box)

                # Get skeleton landmarks (adjusted to global coordinates internally)
                landmarks = skeleton_tracker.track_from_box(frame, box)

                if landmarks:
                    draw_skeleton(frame, landmarks)

                    # Fall detection
                    mqtt_msg = mqtt_client.get_latest_message() if ENABLE_MQTT else None
                    if fall_detector.detect_fall(landmarks, mqtt_msg):
                        gps = mqtt_msg.get("gps", "Unknown") if mqtt_msg else "Unknown"
                        await ami_trigger.alert_devices(f"Fall detected at GPS: {gps}")
                        print("⚠️ Fall detected! Alert sent.")

            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        skeleton_tracker.close()
        if ENABLE_MQTT:
            mqtt_client.stop()
        await ami_trigger.close()


if __name__ == "__main__":
    asyncio.run(main())
