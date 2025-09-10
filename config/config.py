import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# ==============================
# General Configuration
# ==============================
"""
Centralized configuration file for the fall detection project.
Edit values here to adjust system behavior without touching core logic.
"""

# ==============================
# MQTT Configuration
# ==============================
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "io.adafruit.com")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "tranhao")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "tranhao/feeds/json_data")
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))
ENABLE_MQTT = os.getenv("ENABLE_MQTT", "True").lower() == "true"

# ==============================
# Telegram Parameters
# ==============================
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "True").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==============================
# Video Capture Configuration
# ==============================
VIDEO_SOURCE = int(os.getenv("VIDEO_SOURCE", 0))

# ==============================
# Debug and Logging
# ==============================
ENABLE_DEBUG = os.getenv("ENABLE_DEBUG", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ==============================
# Skeleton Tracker Parameters
# ==============================
POSE_MODEL_COMPLEXITY = int(os.getenv("POSE_MODEL_COMPLEXITY", 1))
POSE_MIN_DETECTION_CONFIDENCE = float(os.getenv("POSE_MIN_DETECTION_CONFIDENCE", 0.5))
POSE_MIN_TRACKING_CONFIDENCE = float(os.getenv("POSE_MIN_TRACKING_CONFIDENCE", 0.5))

# ==============================
# Human Detection Parameters
# ==============================
HUMAN_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("HUMAN_DETECTION_CONFIDENCE_THRESHOLD", 0.5))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.7))
FALL_DETECTION_THRESHOLD = float(os.getenv("FALL_DETECTION_THRESHOLD", 0.5))

# ==============================
# Drawing Parameters
# ==============================
BOUNDING_BOX_COLOR = tuple(map(int, os.getenv("BOUNDING_BOX_COLOR", "0, 255, 0").split(',')))
BOUNDING_BOX_THICKNESS = int(os.getenv("BOUNDING_BOX_THICKNESS", 2))
SKELETON_LINE_COLOR = tuple(map(int, os.getenv("SKELETON_LINE_COLOR", "0, 0, 255").split(',')))
SKELETON_LINE_THICKNESS = int(os.getenv("SKELETON_LINE_THICKNESS", 2))
SKELETON_POINT_COLOR = tuple(map(int, os.getenv("SKELETON_POINT_COLOR", "255, 0, 0").split(',')))
SKELETON_POINT_RADIUS = int(os.getenv("SKELETON_POINT_RADIUS", 5))

# ==============================
# AMI Trigger Configuration
# ==============================
AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", 5038))
AMI_USERNAME = os.getenv("AMI_USERNAME", "admin")
AMI_SECRET = os.getenv("AMI_SECRET")
ENABLE_AMI = os.getenv("ENABLE_AMI", "True").lower() == "true"

EXTENSIONS = ['6001', '6002', '6003']
ALERT_MESSAGE = os.getenv("ALERT_MESSAGE", "Fall detected! Please check immediately.")
CALLER_ID = os.getenv("CALLER_ID", "FallAlertSystem <1000>")

# ==============================
# Optional REST API for AMI
# ==============================
AMI_API_ENDPOINT = os.getenv("AMI_API_ENDPOINT", "http://your-ami-api-endpoint")
AMI_API_KEY = os.getenv("AMI_API_KEY")
