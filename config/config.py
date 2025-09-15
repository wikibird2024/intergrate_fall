
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ==============================
# MQTT Configuration
# ==============================
MQTT_BROKER_HOST = "io.adafruit.com"
MQTT_BROKER_PORT = 1883
MQTT_USERNAME = os.getenv("MQTT_USERNAME")        # bí mật -> .env
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")        # bí mật -> .env
MQTT_TOPIC = "tranhao/feeds/fall_alert"
MQTT_KEEPALIVE = 60
ENABLE_MQTT = True

# ==============================
# Telegram Configuration
# ==============================
ENABLE_TELEGRAM = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # bí mật -> .env
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # bí mật -> .env

# ==============================
# Video Sources (simple + practical)
# ==============================
VIDEO_SOURCES = []

# 1. ESP32-CAM HTTP stream (env variable, nếu có)
stream_url = os.getenv("VIDEO_STREAM_URL")
if stream_url:
    VIDEO_SOURCES.append(stream_url)

# 2. Webcam USB gắn ngoài (index 1), nếu có
VIDEO_SOURCES.append(1)

# 3. Webcam mặc định của laptop (index 0) – fallback
VIDEO_SOURCES.append(0)


def get_video_sources():
    """
    Trả về danh sách nguồn video theo thứ tự ưu tiên:
    HTTP stream (nếu có) -> webcam ngoài (1) -> webcam mặc định (0).
    """
    return VIDEO_SOURCES

# ==============================
# Debug / Logging
# ==============================
ENABLE_DEBUG = True
LOG_LEVEL = "INFO"

# ==============================
# Skeleton Tracker
# ==============================
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# ==============================
# Human Detection
# ==============================
HUMAN_DETECTION_CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
FALL_DETECTION_THRESHOLD = 0.5

# ==============================
# Drawing / Visualization
# ==============================
BOUNDING_BOX_COLOR = (0, 255, 0)
BOUNDING_BOX_THICKNESS = 2
SKELETON_LINE_COLOR = (0, 0, 255)
SKELETON_LINE_THICKNESS = 2
SKELETON_POINT_COLOR = (255, 0, 0)
SKELETON_POINT_RADIUS = 5

# ==============================
# AMI / Asterisk Trigger
# ==============================
AMI_HOST = "127.0.0.1"
AMI_PORT = 5038
AMI_USERNAME = os.getenv("AMI_USERNAME")      # bí mật -> .env
AMI_SECRET = os.getenv("AMI_SECRET")          # bí mật -> .env
ENABLE_AMI = True

EXTENSIONS = ['6001', '6002', '6003']
ALERT_MESSAGE = "Fall detected! Please check immediately."
CALLER_ID = "FallAlertSystem <1000>"

# ==============================
# Optional AMI REST API
# ==============================
AMI_API_ENDPOINT = "http://your-ami-api-endpoint"
AMI_API_KEY = os.getenv("AMI_API_KEY")        # bí mật -> .env
