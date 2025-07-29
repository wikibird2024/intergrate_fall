import cv2
import asyncio

# Import application modules
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from fall.fall_detector import FallDetector
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
from utils.draw_utils import draw_bounding_box, draw_skeleton

# Toggle to enable or disable MQTT functionality
ENABLE_MQTT = True


async def mqtt_listener_task(mqtt_client: MQTTClient, message_queue: asyncio.Queue):
    """Background task to receive messages and forward to main loop."""
    try:
        async for message in mqtt_client:
            await message_queue.put(message)
    except asyncio.CancelledError:
        print("[MQTT] Listener task cancelled.")
    except Exception as e:
        print(f"[MQTT] Listener error: {e}")


async def main():
    """
    Main async function to run the fall detection system.
    Components:
        - Human detection
        - Skeleton tracking
        - Fall detection
        - MQTT message handling
        - AMI alert triggering
    """

    # 1. Initialize modules
    human_detector = HumanDetector()
    skeleton_tracker = SkeletonTracker()
    fall_detector = FallDetector()
    ami_trigger = AMITrigger()
    mqtt_client = None
    mqtt_task = None
    message_queue = asyncio.Queue()

    # 2. Start MQTT (if enabled)
    if ENABLE_MQTT:
        try:
            mqtt_client = MQTTClient()
            asyncio.create_task(mqtt_client.run_forever())  # Run MQTT in background
            mqtt_task = asyncio.create_task(
                mqtt_listener_task(mqtt_client, message_queue)
            )
        except Exception as e:
            print(f"[MQTT] ⚠️ Could not start MQTT: {e}")
            mqtt_client = None

    # 3. Connect AMI trigger (e.g., for SMS/call alert)
    await ami_trigger.trigger()

    # 4. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video capture.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = human_detector.detect_humans(frame)

            # Try to get latest MQTT message (non-blocking)
            mqtt_msg = None
            if not message_queue.empty():
                mqtt_msg = message_queue.get_nowait()

            for box in boxes:
                draw_bounding_box(frame, box)
                landmarks = skeleton_tracker.track_from_box(frame, box)

                if landmarks:
                    draw_skeleton(frame, landmarks)
                    is_fall = fall_detector.detect_fall(landmarks, mqtt_msg)

                    if is_fall:
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
import cv2
import asyncio

# Import application modules
from detection.human_detector import HumanDetector
from detection.skeleton_tracker import SkeletonTracker
from fall.fall_detector import FallDetector
from comm.mqtt_client import MQTTClient
from comm.ami_trigger import AMITrigger
from utils.draw_utils import draw_bounding_box, draw_skeleton

# Toggle to enable or disable MQTT functionality
ENABLE_MQTT = True


async def mqtt_listener_task(mqtt_client: MQTTClient, message_queue: asyncio.Queue):
    """Background task to receive messages and forward to main loop."""
    try:
        async for message in mqtt_client:
            await message_queue.put(message)
    except asyncio.CancelledError:
        print("[MQTT] Listener task cancelled.")
    except Exception as e:
        print(f"[MQTT] Listener error: {e}")


async def main():
    """
    Main async function to run the fall detection system.
    Components:
        - Human detection
        - Skeleton tracking
        - Fall detection
        - MQTT message handling
        - AMI alert triggering
    """

    # 1. Initialize modules
    human_detector = HumanDetector()
    skeleton_tracker = SkeletonTracker()
    fall_detector = FallDetector()
    ami_trigger = AMITrigger()
    mqtt_client = None
    mqtt_task = None
    message_queue = asyncio.Queue()

    # 2. Start MQTT (if enabled)
    if ENABLE_MQTT:
        try:
            mqtt_client = MQTTClient()
            await mqtt_client.start()
            mqtt_task = asyncio.create_task(
                mqtt_listener_task(mqtt_client, message_queue)
            )
        except Exception as e:
            print(f"[MQTT] ⚠️ Could not start MQTT: {e}")
            mqtt_client = None

    # 3. Connect AMI trigger (e.g., for SMS/call alert)
    await ami_trigger.trigger()

    # 4. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video capture.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = human_detector.detect_humans(frame)

            # Try to get latest MQTT message (non-blocking)
            mqtt_msg = None
            if not message_queue.empty():
                mqtt_msg = message_queue.get_nowait()

            for box in boxes:
                draw_bounding_box(frame, box)
                landmarks = skeleton_tracker.track_from_box(frame, box)

                if landmarks:
                    draw_skeleton(frame, landmarks)
                    is_fall = fall_detector.detect_fall(landmarks, mqtt_msg)

                    if is_fall:
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
