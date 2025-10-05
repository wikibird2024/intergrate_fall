
import asyncio
from panoramisk.manager import Manager
import logging
from typing import List

# Import configuration từ file config
from config.config import EXTENSIONS, ALERT_MESSAGE, CALLER_ID

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class AMITrigger:
    def __init__(self, host: str, port: int, username: str, secret: str):
        """Initialize AMI manager with given credentials."""
        self.manager = Manager(host=host, port=port, username=username, secret=secret)
        self.host = host
        self.port = port
        self.username = username
        self.secret = secret

        self.extensions = EXTENSIONS
        self.alert_message = ALERT_MESSAGE
        self.caller_id = CALLER_ID
        self.is_connected = False

    async def connect(self):
        """Connect to AMI asynchronously."""
        try:
            logger.info(f"[AMI] Connecting to {self.host}:{self.port}...")
            await self.manager.connect()
            self.is_connected = True
            logger.info("[AMI] ✅ Connected to AMI server.")
        except Exception as e:
            self.is_connected = False
            logger.error(f"[AMI] ❌ Failed to connect: {e}")
            raise

    async def _ensure_connected(self) -> bool:
        """Check and reconnect if disconnected."""
        if self.is_connected:
            return True
        logger.warning("[AMI] ⚠️ Connection lost. Attempting reconnect...")
        try:
            await self.connect()
            return True
        except Exception:
            return False

    async def alert_devices(self, message: str = None):
        """Send alert to all configured extensions."""
        message = message or self.alert_message

        if not await self._ensure_connected():
            logger.warning("[AMI] Alert aborted due to connection failure.")
            return

        logger.info(f"[AMI] Triggering alert to {len(self.extensions)} device(s)...")

        # Gửi alert song song
        tasks = [self._handle_extension(ext, message) for ext in self.extensions]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Đợi 3s cho channel cleanup
        CLEANUP_DELAY_SECONDS = 3
        await asyncio.sleep(CLEANUP_DELAY_SECONDS)
        logger.info("[AMI] Alert process finished.")

    async def _handle_extension(self, extension: str, message: str):
        """Send call and message to a single extension concurrently."""
        await asyncio.gather(
            self._originate_call(extension, message),
            self._send_message(extension, message)
        )

    async def _originate_call(self, extension: str, message: str):
        """Originate a call to the hangup point 9999."""
        try:
            response = await self.manager.send_action({
                "Action": "Originate",
                "Channel": f"PJSIP/{extension}",
                "Context": "internal",
                "Exten": "9999",           # Hangup point trong extensions.ael
                "Priority": 1,
                "CallerID": self.caller_id,
                "Variable": f"ALERT_MSG={message}",
                "Async": "true",
            })
            if isinstance(response, dict):
                status = response.get("Response", "Unknown")
                msg = response.get("Message", "")
                logger.info(f"[📞 CALL] → {extension} | Status: {status} - {msg}")
            else:
                logger.error(f"[📞 CALL] → {extension} | ❌ Invalid AMI response: {response}")
        except Exception as e:
            logger.error(f"[📞 CALL] → {extension} | ❌ Error: {e}")
            self.is_connected = False

    async def _send_message(self, extension: str, message: str):
        """Send SIP MESSAGE to the extension via server endpoint."""
        try:
            response = await self.manager.send_action({
                "Action": "MessageSend",
                "To": f"pjsip:{extension}",
                "From": "server",
                "Body": message
            })
            status = response.get("Response", "Unknown")
            msg = response.get("Message", "")
            logger.info(f"[📨 SMS] → {extension} | Status: {status} - {msg}")
        except Exception as e:
            logger.error(f"[📨 SMS] → {extension} | ❌ Error: {e}")
            self.is_connected = False

    async def close(self):
        """Disconnect safely from AMI."""
        if self.is_connected:
            self.manager.close()
            self.is_connected = False
            logger.info("[AMI] 🔌 Disconnected from AMI server.")


# === Example usage ===
# async def main():
#     ami = AMITrigger(host="127.0.0.1", port=5038, username="hx", secret="123")
#     await ami.connect()
#     await ami.alert_devices("⚠️ Cảnh báo: Ngã phát hiện tại vị trí!")
#     await ami.close()
#
# asyncio.run(main())
