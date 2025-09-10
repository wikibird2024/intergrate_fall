
import cv2
import os
import requests
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class ESP32StreamWrapper:
    """
    Wrapper to emulate cv2.VideoCapture from an ESP32-CAM JPEG stream.
    """
    def __init__(self, url, alive_timeout=10, connect_timeout=5, max_buffer=1024*1024):
        self.url = url
        self.stream = None
        self.bytes_buffer = b''
        self.last_frame_time = time.time()
        self.alive_timeout = alive_timeout
        self.connect_timeout = connect_timeout
        self.max_buffer = max_buffer

        # Attempt initial connection
        if not self.connect():
            raise ConnectionError(f"Failed to connect to stream: {url}")

    def connect(self):
        """Initialize or reconnect to the stream."""
        try:
            if self.stream:
                self.stream.close()
            self.stream = requests.get(self.url, stream=True, timeout=self.connect_timeout)
            self.stream.raise_for_status()
            self.bytes_buffer = b''  # Reset buffer
            logger.info(f"Successfully opened video stream: {self.url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to stream {self.url}: {e}")
            self.stream = None
            return False

    def read(self):
        """Read a frame from the stream."""
        if not self.stream:
            if not self.connect():
                return False, None

        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                if not chunk:
                    break

                self.bytes_buffer += chunk

                # Limit buffer size to prevent memory growth
                if len(self.bytes_buffer) > self.max_buffer:
                    self.bytes_buffer = self.bytes_buffer[-self.max_buffer:]

                start = self.bytes_buffer.find(b'\xff\xd8')
                end = self.bytes_buffer.find(b'\xff\xd9')

                if start != -1 and end != -1:
                    jpg = self.bytes_buffer[start:end+2]
                    self.bytes_buffer = self.bytes_buffer[end+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if img is not None:
                        self.last_frame_time = time.time()
                        return True, img

            # If no frame is found, check for reconnect
            if not self.is_alive():
                logger.warning("Stream timeout, attempting reconnect...")
                if not self.connect():
                    return False, None

        except Exception as e:
            logger.error(f"ESP32 stream read error: {e}")
            self.release()
            return False, None

        return False, None

    def release(self):
        """Close the stream."""
        try:
            if self.stream:
                self.stream.close()
                self.stream = None
        except Exception:
            pass

    def is_alive(self):
        """Check if the stream is still alive."""
        return (time.time() - self.last_frame_time) < self.alive_timeout


def get_video_source(priority_sources, fallback_index=0):
    """
    Attempt to open a video source from a list of priority sources.
    """
    for source in priority_sources:
        logger.info(f"Attempting to open video source: {source}")

        # Case 1: URL (ESP32 stream)
        if isinstance(source, str) and source.startswith("http"):
            try:
                stream_wrapper = ESP32StreamWrapper(source)
                if stream_wrapper.stream:
                    return stream_wrapper
            except Exception as e:
                logger.error(f"Failed to initialize ESP32 stream {source}: {e}")
                if 'stream_wrapper' in locals():
                    stream_wrapper.release()

        # Case 2: Webcam index
        elif isinstance(source, int):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                logger.info(f"Successfully opened webcam index: {source}")
                return cap
            else:
                logger.error(f"Failed to open webcam index: {source}")
                cap.release()

    # If all priority sources fail → fallback webcam
    logger.warning("All priority sources failed. Attempting fallback webcam...")
    cap = cv2.VideoCapture(fallback_index)
    if cap.isOpened():
        logger.info(f"Successfully opened fallback webcam: {fallback_index}")
        return cap
    else:
        logger.error("Failed to open fallback webcam. No video source available.")
        cap.release()

    return None


def get_config_sources():
    """
    Load video sources from environment variables.
    """
    sources = []

    stream_url = os.getenv("VIDEO_STREAM_URL")
    if stream_url:
        sources.append(stream_url)

    webcam_index = os.getenv("VIDEO_WEBCAM_INDEX", "0")
    try:
        sources.append(int(webcam_index))
    except ValueError:
        sources.append(0)

    return sources
