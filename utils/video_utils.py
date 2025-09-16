import cv2
import os
import requests
import numpy as np
import time
import logging
import asyncio

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("video_utils")

# ---

class ESP32StreamWrapper:
    """
    Wrapper to emulate cv2.VideoCapture from an ESP32-CAM JPEG stream.
    Fully compatible with async code via asyncio.to_thread().
    """

    def __init__(self, url, alive_timeout=10, connect_timeout=5, max_buffer=1024*1024):
        self.url = url
        self.stream = None
        self.bytes_buffer = b''
        self.last_frame_time = time.time()
        self.alive_timeout = alive_timeout
        self.connect_timeout = connect_timeout
        self.max_buffer = max_buffer
        self.read_chunk_size = 8192

        if not self.connect():
            raise ConnectionError(f"Failed to connect to ESP32 stream: {self.url}")

    def connect(self) -> bool:
        """Initialize or reconnect to the ESP32 stream."""
        try:
            if self.stream:
                self.release()

            self.stream = requests.get(self.url, stream=True, timeout=self.connect_timeout)
            self.stream.raise_for_status()
            self.bytes_buffer = b''
            self.last_frame_time = time.time()
            logger.info(f"[ESP32] Connected to stream: {self.url}")
            return True
        except requests.RequestException as e:
            logger.error(f"[ESP32] Connection failed ({self.url}): {e}")
            self.stream = None
            return False

    def read(self) -> (bool, np.ndarray):
        """Read a single frame from the ESP32 stream."""
        if not self.is_alive():
            if not self.connect():
                return False, None
        
        if self.stream is None:
            return False, None

        try:
            chunk = next(self.stream.iter_content(chunk_size=self.read_chunk_size), None)
            if not chunk:
                logger.warning("[ESP32] Stream seems to have ended. Attempting to reconnect.")
                if not self.connect():
                    return False, None
                return False, None

            self.bytes_buffer += chunk
            
            start_delimiter = b'\xff\xd8'
            end_delimiter = b'\xff\xd9'
            start = self.bytes_buffer.find(start_delimiter)
            end = self.bytes_buffer.find(end_delimiter)

            if start != -1 and end != -1 and end > start:
                jpg_data = self.bytes_buffer[start:end + len(end_delimiter)]
                self.bytes_buffer = self.bytes_buffer[end + len(end_delimiter):]
                
                img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    self.last_frame_time = time.time()
                    return True, img
        except requests.exceptions.Timeout:
            logger.warning(f"[ESP32] Timeout reading stream: {self.url}. Reconnecting.")
            self.connect()
        except Exception as e:
            logger.error(f"[ESP32] Read error: {e}")
            self.release()

        return False, None

    def release(self):
        """Release the stream connection."""
        if self.stream:
            self.stream.close()
            self.stream = None
            logger.info(f"[ESP32] Stream released: {self.url}")

    def is_alive(self) -> bool:
        """Check if stream is alive based on last frame read time."""
        return (time.time() - self.last_frame_time) < self.alive_timeout

# ---

def find_and_connect_source(priority_sources):
    """
    Attempt to open a video source from a list of priority sources.
    Returns either an ESP32StreamWrapper or cv2.VideoCapture object.
    """
    for source in priority_sources:
        logger.info(f"[VIDEO] Trying source: {source}")

        if isinstance(source, str) and source.startswith("http"):
            try:
                wrapper = ESP32StreamWrapper(source)
                return wrapper
            except ConnectionError as e:
                logger.error(f"[VIDEO] Failed to initialize ESP32 stream {source}: {e}")
                continue
            except Exception as e:
                logger.error(f"[VIDEO] An unexpected error occurred with ESP32 stream {source}: {e}")
                continue

        elif isinstance(source, (int, str)):
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    logger.info(f"[VIDEO] Opened webcam: {source}")
                    return cap
                else:
                    logger.error(f"[VIDEO] Cannot open webcam: {source}")
                    cap.release()
            except Exception as e:
                logger.error(f"[VIDEO] An unexpected error occurred with webcam {source}: {e}")
                continue  # Sửa: Thêm 'continue' để tiếp tục vòng lặp sau khi gặp lỗi

    logger.error("[VIDEO] No valid video source found")
    return None

# ---

def get_config_sources():
    """
    Load video sources from environment variables.
    Returns list of sources (str for HTTP, int/str for webcam).
    """
    sources = []
    stream_url = os.getenv("VIDEO_STREAM_URL")
    webcam_index_str = os.getenv("VIDEO_WEBCAM_INDEX")
    
    if stream_url:
        sources.append(stream_url)

    if webcam_index_str:
        try:
            if webcam_index_str.isdigit():
                sources.append(int(webcam_index_str))
            else:
                sources.append(webcam_index_str)
        except ValueError:
            logger.warning(f"[VIDEO] Invalid VIDEO_WEBCAM_INDEX: {webcam_index_str}")

    if not sources:
        sources.append(0)

    return sources
