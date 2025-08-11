import telegram
from telegram.error import TelegramError
import io
import cv2

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id

    async def send_photo(self, photo: bytes, caption: str):
        try:
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo,
                caption=caption
            )
        except TelegramError as e:
            print(f"❌ Telegram Error: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while sending Telegram message: {e}")
