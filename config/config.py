# config.py
"""
Centralized configuration file for the fall detection project.
Edit values here to adjust system behavior without touching core logic.
"""
# config.py
# ...

# ==============================
# MQTT Configuration
# ==============================
MQTT_BROKER_HOST = ""
MQTT_BROKER_PORT = 1883  # Use 8883 for SSL/TLS encrypted connection
MQTT_USERNAME = "tranhao"  # Replace with your AIO username
MQTT_PASSWORD = ""  # Replace with your AIO Key
MQTT_TOPIC = "tranhao/feeds/json_data"  # Topic format for Adafruit IO
MQTT_KEEPALIVE = 60  # Keepalive interval
ENABLE_MQTT = True  # Toggle MQTT communication

# ==============================
# Video Capture Configuration
# ==============================
VIDEO_SOURCE = 0  # Webcam index or path to video file

# ==============================
# Debug and Logging
# ==============================
ENABLE_DEBUG = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ==============================
# Fall Detection Parameters
# ==============================
FALL_DETECTION_THRESHOLD = 0.5  # Confidence threshold

# ==============================
# AMI Trigger Configuration
# ==============================
AMI_HOST = "127.0.0.1"  # or "localhost"
AMI_PORT = 5038
AMI_USERNAME = "admin"
AMI_SECRET = "123"

# The three missing variables below:
EXTENSIONS = ["1001", "1002", "1003"]  # List of phone extensions to call
ALERT_MESSAGE = "Fall detected! Please check immediately."  # Message content
CALLER_ID = "FallAlertSystem <1000>"  # Caller ID for outgoing calls

# ==============================
# Optional REST API for AMI
# ==============================
AMI_API_ENDPOINT = "http://your-ami-api-endpoint"
AMI_API_KEY = "your_api_key_here"
