
#!/usr/bin/env python3
# esp32_stream.py
"""
Production-ready ESP32-CAM MJPEG stream wrapper.

Features:
- Robust parsing of MJPEG stream (handles chunked JPEGs)
- Keeps a persistent requests.Session and iterator for stream.iter_content()
- Automatic reconnect with exponential backoff
- Buffer overflow protection (max_buffer)
- Alive watchdog (alive_timeout)
- Optional synchronous use and async compatibility via asyncio.to_thread()
- Context manager support (with ...)
- Helpful logging

Usage (sync):
    from esp32_stream import find_and_connect_source, get_config_sources
    srcs = get_config_sources()
    cap = find_and_connect_source(srcs)
    if cap:
        if hasattr(cap, "read"):
            ok, frame = cap.read()
            ...
        else:
            ret, frame = cap.read()
            ...

Usage (async):
    wrapper = await asyncio.to_thread(ESP32StreamWrapper, url)
    ok, frame = await asyncio.to_thread(wrapper.read)
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional, Tuple, Union, List
import requests
import numpy as np
import cv2
from urllib.parse import urlparse

# Setup logger (module-level)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("esp32_stream")

# Exceptions
class StreamError(Exception):
    pass


class ESP32StreamWrapper:
    """
    Wrapper that emulates cv2.VideoCapture for an ESP32-CAM MJPEG HTTP stream.

    Primary methods:
      - read() -> (bool, np.ndarray | None)
      - release()
      - is_alive() -> bool

    Use context manager:
      with ESP32StreamWrapper(url) as w:
          ok, frame = w.read()
    """

    def __init__(
        self,
        url: str,
        alive_timeout: float = 10.0,
        connect_timeout: float = 10.0,
        read_chunk_size: int = 8192,
        max_buffer: int = 2 * 1024 * 1024,
        reconnect_backoff_base: float = 0.5,
        reconnect_backoff_max: float = 8.0,
    ):
        self.url = url
        self.alive_timeout = float(alive_timeout)
        self.connect_timeout = float(connect_timeout)
        self.read_chunk_size = int(read_chunk_size)
        self.max_buffer = int(max_buffer)
        self._backoff_base = float(reconnect_backoff_base)
        self._backoff_max = float(reconnect_backoff_max)

        self.session: Optional[requests.Session] = None
        self.response: Optional[requests.Response] = None
        self._iter = None  # generator from response.iter_content(...)
        self.bytes_buffer = b''
        self.last_frame_time = 0.0
        self._connected_at = 0.0
        self._connect_attempts = 0

        # Validate URL scheme for HTTP
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        # Try initial connect
        if not self.connect():
            raise StreamError(f"Failed to connect to ESP32 stream: {self.url}")

    # --------- Connection handling ---------

    def _reset_iter(self) -> None:
        """Reset the internal iterator for iter_content."""
        self._iter = None

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        # Optional: tune session headers to look like a browser or VLC
        s.headers.update({
            "User-Agent": "esp32-mjpeg-client/1.0",
            "Accept": "multipart/x-mixed-replace, image/jpeg, */*",
            "Connection": "keep-alive",
        })
        return s

    def connect(self) -> bool:
        """
        Connect (or reconnect) to the ESP32 MJPEG stream.
        Uses exponential backoff between repeated failures.
        """
        self._connect_attempts += 1
        backoff = min(self._backoff_base * (2 ** (self._connect_attempts - 1)), self._backoff_max)
        if self.session is None:
            self.session = self._make_session()

        # Close previous response if any
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass
            self.response = None

        try:
            logger.info(f"[ESP32] Connecting to {self.url} (attempt {self._connect_attempts})")
            # stream=True to keep connection open
            self.response = self.session.get(self.url, stream=True, timeout=self.connect_timeout)
            self.response.raise_for_status()

            # Basic header verification
            ctype = self.response.headers.get("Content-Type", "")
            if "multipart" not in ctype and "jpeg" not in ctype:
                logger.warning(f"[ESP32] Unexpected Content-Type: {ctype} (continuing anyway)")

            self._iter = self.response.iter_content(chunk_size=self.read_chunk_size)
            self.bytes_buffer = b''
            self.last_frame_time = time.time()
            self._connected_at = time.time()
            self._connect_attempts = 0  # reset attempts on success
            logger.info(f"[ESP32] Connected to stream: {self.url}")
            return True
        except requests.RequestException as e:
            logger.error(f"[ESP32] Connection failed ({self.url}): {e}. Backoff {backoff:.1f}s")
            # Close and cleanup session/response on failure
            if self.response is not None:
                try:
                    self.response.close()
                except Exception:
                    pass
                self.response = None
            # Sleep a bit to avoid hammering the ESP32
            time.sleep(backoff)
            return False

    def release(self) -> None:
        """Close and cleanup network resources."""
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass
            self.response = None
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
        self._reset_iter()
        self.bytes_buffer = b''
        logger.info(f"[ESP32] Stream released: {self.url}")

    # --------- Frame parsing / reading ---------

    def is_alive(self) -> bool:
        """Return True if we received a frame recently (within alive_timeout)."""
        return (time.time() - self.last_frame_time) < float(self.alive_timeout)

    def _ensure_connected(self) -> bool:
        """Ensure we have a live connection; try reconnect if needed."""
        if self.response is None or self._iter is None:
            return self.connect()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the MJPEG stream.

        Returns:
            (True, frame) on success, (False, None) otherwise.
        """
        # If we've not seen a frame for a while, try to reconnect proactively
        if not self.is_alive():
            logger.debug("[ESP32] Alive timeout expired; reconnecting.")
            if not self.connect():
                return False, None

        if not self._ensure_connected():
            return False, None

        try:
            # Ensure we have an iterator (it persists across reads)
            if self._iter is None and self.response is not None:
                self._iter = self.response.iter_content(chunk_size=self.read_chunk_size)

            chunk = next(self._iter, None)
            if chunk is None:
                # Iterator ended or empty - reconnect
                logger.warning("[ESP32] Stream iterator ended or returned empty chunk. Reconnecting.")
                self._reset_iter()
                self.connect()
                return False, None

            if len(chunk) == 0:
                logger.warning("[ESP32] Received zero-length chunk. Ignoring.")
                return False, None

            self.bytes_buffer += chunk

            # Protect against runaway buffer
            if len(self.bytes_buffer) > self.max_buffer:
                logger.warning("[ESP32] Buffer overflow (>%d bytes). Resetting buffer.", self.max_buffer)
                self.bytes_buffer = b''

            # Extract full JPEGs from buffer. There may be multiple complete frames in buffer.
            # We'll return the first complete JPEG we find.
            start_marker = b'\xff\xd8'
            end_marker = b'\xff\xd9'

            while True:
                start = self.bytes_buffer.find(start_marker)
                end = self.bytes_buffer.find(end_marker)
                if start != -1 and end != -1 and end > start:
                    jpg = self.bytes_buffer[start:end + 2]
                    # consume up to end
                    self.bytes_buffer = self.bytes_buffer[end + 2:]
                    # decode
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        # fallback to grayscale decode
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # convert to 3-channel BGR for compatibility
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    if img is not None:
                        self.last_frame_time = time.time()
                        return True, img
                    else:
                        logger.warning("[ESP32] Failed to decode JPEG; skipping.")
                        # continue loop to try next JPG in buffer (if any)
                        continue
                else:
                    # No full JPEG available yet
                    break

        except StopIteration:
            logger.warning("[ESP32] Stream iterator exhausted (StopIteration). Reconnecting.")
            self._reset_iter()
            self.connect()
        except requests.exceptions.ChunkedEncodingError as e:
            logger.warning(f"[ESP32] Chunked encoding error: {e}. Reconnecting.")
            self._reset_iter()
            self.connect()
        except requests.exceptions.Timeout:
            logger.warning("[ESP32] Read timeout. Reconnecting.")
            self._reset_iter()
            self.connect()
        except requests.exceptions.RequestException as e:
            logger.error(f"[ESP32] Requests error while reading: {e}. Reconnecting.")
            self._reset_iter()
            self.connect()
        except Exception as e:
            logger.exception(f"[ESP32] Unexpected error while reading stream: {e}. Releasing.")
            self.release()

        return False, None

    # --------- Context manager support ---------

    def __enter__(self) -> "ESP32StreamWrapper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


# ---------------- Helper functions ----------------

def find_and_connect_source(priority_sources: List[Union[str, int]]) -> Optional[Union[ESP32StreamWrapper, "cv2.VideoCapture"]]:
    """
    Attempt to open a video source from a list of priority sources.
    Returns either an ESP32StreamWrapper or cv2.VideoCapture object, or None if none succeeded.
    Sources:
      - HTTP URL string -> ESP32StreamWrapper
      - int or string (file path) -> cv2.VideoCapture
    """
    for source in priority_sources:
        logger.info(f"[VIDEO] Trying source: {source}")
        if isinstance(source, str) and source.startswith("http"):
            try:
                wrapper = ESP32StreamWrapper(source)
                return wrapper
            except Exception as e:
                logger.error(f"[VIDEO] Failed to initialize ESP32 stream {source}: {e}")
                continue
        else:
            # Try cv2.VideoCapture (webcam index or filepath)
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    logger.info(f"[VIDEO] Opened webcam/file: {source}")
                    return cap
                else:
                    logger.error(f"[VIDEO] Cannot open webcam/file: {source}")
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[VIDEO] Unexpected error opening webcam/file {source}: {e}")
                continue

    logger.error("[VIDEO] No valid video source found")
    return None


def get_config_sources() -> List[Union[str, int]]:
    """
    Load video sources from environment variables:
      - VIDEO_STREAM_URL (string)
      - VIDEO_WEBCAM_INDEX (int or string path)
    If none provided, defaults to [0] (first webcam).
    """
    sources: List[Union[str, int]] = []
    stream_url = os.getenv("VIDEO_STREAM_URL")
    webcam_index_str = os.getenv("VIDEO_WEBCAM_INDEX")

    if stream_url:
        sources.append(stream_url.strip())

    if webcam_index_str:
        # If it's digits -> webcam index, else treat as path
        if webcam_index_str.strip().isdigit():
            try:
                sources.append(int(webcam_index_str.strip()))
            except ValueError:
                logger.warning(f"[VIDEO] Invalid VIDEO_WEBCAM_INDEX value: {webcam_index_str}")
        else:
            sources.append(webcam_index_str.strip())

    if not sources:
        sources.append(0)

    return sources


# ---------------- Example quick-run (only when executed directly) ----------------
if __name__ == "__main__":
    import sys

    url = os.getenv("VIDEO_STREAM_URL", "http://192.168.1.80/stream")
    sources = [url, 0]

    cap = find_and_connect_source(sources)
    if cap is None:
        logger.error("No video source available. Exiting.")
        sys.exit(1)

    try:
        # If cap is our wrapper:
        if isinstance(cap, ESP32StreamWrapper):
            logger.info("Using ESP32StreamWrapper. Press ESC to exit.")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # small sleep to avoid busy-loop if stream down
                    time.sleep(0.1)
                    continue
                cv2.imshow("ESP32 Stream", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        else:
            logger.info("Using cv2.VideoCapture. Press ESC to exit.")
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        try:
            if isinstance(cap, ESP32StreamWrapper):
                cap.release()
            else:
                cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
