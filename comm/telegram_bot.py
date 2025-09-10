
import io
import cv2
import numpy as np
import asyncio
from telegram import Bot
from telegram.error import TelegramError

class TelegramBot:
    """
    Async Telegram bot wrapper compatible with python-telegram-bot v22+.
    Sends messages and OpenCV frames safely with retry and fallback.
    """

    def __init__(self, token: str, chat_id: str, send_test_message: bool = True):
        self.bot = Bot(token=token)  # v22+ không dùng request_kwargs ở đây
        self.chat_id = chat_id

        if send_test_message:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            loop.create_task(self._send_test_message())

    async def _send_test_message(self):
        try:
            await self.send_message("✅ Telegram bot initialized successfully.")
            print("[TELEGRAM] Test message sent successfully.")
        except Exception as e:
            print(f"[TELEGRAM] ⚠️ Failed to send test message during init: {e}")

    async def send_message(self, text: str):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except TelegramError as e:
            print(f"[TELEGRAM] ❌ TelegramError in send_message: {e}")
        except Exception as e:
            print(f"[TELEGRAM] ❌ Unexpected error in send_message: {e}")

    async def send_photo(self, frame: np.ndarray, caption: str = "", retries: int = 3, delay: float = 1.0):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("[TELEGRAM] ⚠️ Invalid frame, sending text only")
            return await self.send_message(caption)

        frame_safe = self._prepare_frame(frame)

        for attempt in range(retries):
            try:
                success, img_encoded = cv2.imencode(".jpg", frame_safe)
                if not success:
                    print("[TELEGRAM] ⚠️ imencode failed, sending text")
                    return await self.send_message(caption)

                with io.BytesIO(img_encoded.tobytes()) as photo_bytes:
                    photo_bytes.name = "image.jpg"
                    photo_bytes.seek(0)
                    await self.bot.send_photo(chat_id=self.chat_id, photo=photo_bytes, caption=caption)
                return
            except Exception as e:
                print(f"[TELEGRAM] ❌ Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(delay)

        try:
            await self.send_message(caption)
        except Exception as e:
            print(f"[TELEGRAM] ❌ Failed sending fallback text: {e}")

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        frame_safe = cv2.convertScaleAbs(frame) if frame.dtype != np.uint8 else frame.copy()

        if frame_safe.ndim == 2:
            frame_safe = cv2.cvtColor(frame_safe, cv2.COLOR_GRAY2BGR)
        elif frame_safe.ndim == 3 and frame_safe.shape[2] == 4:
            frame_safe = cv2.cvtColor(frame_safe, cv2.COLOR_BGRA2BGR)

        return frame_safe
