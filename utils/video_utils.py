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
        self.read_chunk_size = 8192  # Optimized chunk size

        # Attempt initial connection
        if not self.connect():
            raise ConnectionError(f"Failed to connect to stream: {self.url}")

    def connect(self):
        """Initialize or reconnect to the stream."""
        try:
            if self.stream:
                self.stream.close()
            self.stream = requests.get(self.url, stream=True, timeout=self.connect_timeout)
            self.stream.raise_for_status()
            self.bytes_buffer = b''
            self.last_frame_time = time.time()
            logger.info(f"Successfully opened video stream: {self.url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to stream {self.url}: {e}")
            self.stream = None
            return False

    def read(self):
        """Read a frame from the stream."""
        if not self.is_alive() and not self.connect():
            return False, None

        try:
            # Read a single chunk to prevent infinite loop
            chunk = next(self.stream.iter_content(chunk_size=self.read_chunk_size), None)
            if not chunk:
                # End of stream or no data, check for reconnect
                if not self.connect():
                    return False, None
                return False, None # No new data, return empty frame
            
            self.bytes_buffer += chunk

            start = self.bytes_buffer.find(b'\xff\xd8')
            end = self.bytes_buffer.find(b'\xff\xd9')

            if start != -1 and end != -1 and end > start:
                jpg_data = self.bytes_buffer[start:end+2]
                self.bytes_buffer = self.bytes_buffer[end+2:] # Truncate buffer
                
                img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    self.last_frame_time = time.time()
                    return True, img
        
        except requests.exceptions.Timeout:
            logger.warning(f"Stream read timeout from {self.url}, attempting reconnect...")
            self.connect()
            return False, None
        except Exception as e:
            logger.error(f"ESP32 stream read error: {e}")
            self.release()
            return False, None

        return False, None

    def release(self):
        """Close the stream."""
        if self.stream:
            self.stream.close()
            self.stream = None
            logger.info(f"Stream from {self.url} released.")

    def is_alive(self):
        """Check if the stream is still alive."""
        return (time.time() - self.last_frame_time) < self.alive_timeout

def get_video_source(priority_sources):
    """
    Attempt to open a video source from a list of priority sources.
    """
    for source in priority_sources:
        logger.info(f"Attempting to open video source: {source}")

        if isinstance(source, str) and source.startswith("http"):
            try:
                stream_wrapper = ESP32StreamWrapper(source)
                return stream_wrapper
            except ConnectionError:
                continue
            except Exception as e:
                logger.error(f"Failed to initialize ESP32 stream {source}: {e}")
                continue

        elif isinstance(source, int):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                logger.info(f"Successfully opened webcam index: {source}")
                return cap
            else:
                logger.error(f"Failed to open webcam index: {source}")
                cap.release()
    
    logger.error("No valid video source found from priority list.")
    return None

def get_config_sources():
    """
    Load video sources from environment variables.
    """
    sources = []
    stream_url = os.getenv("VIDEO_STREAM_URL")
    if stream_url:
        sources.append(stream_url)
    
    webcam_index_str = os.getenv("VIDEO_WEBCAM_INDEX")
    if webcam_index_str:
        try:
            sources.append(int(webcam_index_str))
        except ValueError:
            pass # Ignore if it's not a valid number
    
    # Add a fallback for default webcam
    if not sources:
        sources.append(0)

    return sources
