import asyncio
from panoramisk.manager import Manager
import logging
from typing import Dict, Any

# Import configuration from a centralized file.
from config.config import EXTENSIONS, ALERT_MESSAGE, CALLER_ID

# Setup logging for the module
logger = logging.getLogger(__name__)

class AMITrigger:
    def __init__(self, host, port, username, secret):
        """Initializes the AMI Manager instance with credentials passed as arguments."""
        self.manager = Manager(
            host=host, port=port, username=username, secret=secret
        )
        self.extensions = EXTENSIONS
        self.alert_message = ALERT_MESSAGE
        self.caller_id = CALLER_ID
        self.is_connected = False
        
        # Store host, port, etc., as direct attributes of the AMITrigger class
        # to make them accessible for logging
        self.host = host
        self.port = port
        self.username = username
        self.secret = secret

    async def connect(self):
        """Asynchronously connect to the AMI server."""
        try:
            logger.info(f"[AMI] Connecting to AMI at {self.host}:{self.port}...")
            await self.manager.connect()
            self.is_connected = True
            logger.info("[AMI] ✅ Connected to AMI server.")
        except Exception as e:
            logger.error(f"[AMI] ❌ Failed to connect to AMI: {e}")
            self.is_connected = False
            raise

    async def alert_devices(self, message: str):
        """
        Triggers a multi-channel alert by initiating calls and sending messages to a list of extensions.
        This method now combines both call and message actions using asyncio.gather.
        """
        if not self.is_connected:
            logger.warning("[AMI] ⚠️ Not connected to AMI. Alert not sent.")
            return

        logger.info(f"[AMI] Triggering alert to {len(self.extensions)} device(s)...")

        tasks = [self._handle_extension(extension) for extension in self.extensions]
        await asyncio.gather(*tasks)

    async def _handle_extension(self, extension: str):
        """Handles both call and message actions for a single extension in parallel."""
        await asyncio.gather(
            self._originate_call(extension),
            self._send_message(extension)
        )

    async def _originate_call(self, extension: str):
        """Helper to originate a single call to an extension."""
        try:
            response = await self.manager.send_action(
                {
                    "Action": "Originate",
                    "Channel": f"PJSIP/{extension}",
                    "Context": "internal",
                    "Exten": extension,
                    "Priority": 1,
                    "CallerID": self.caller_id,
                    "Variable": f"ALERT_MSG={self.alert_message}",
                    "Async": "true",
                }
            )
            if isinstance(response, dict):
                status = response.get("Response", "Unknown")
                message = response.get("Message", "")
                logger.info(f"[📞 CALL] → {extension} | Status: {status} - {message}")
            else:
                logger.error(f"[📞 CALL] → {extension} | ❌ Invalid AMI response: {response}")

        except Exception as e:
            logger.error(f"[📞 CALL] → {extension} | ❌ Error: {e}")

    async def _send_message(self, extension: str):
        """Helper to send a single message to an extension."""
        try:
            response = await self.manager.send_action(
                {
                    'Action': 'MessageSend',
                    'To': f'pjsip:{extension}',
                    'From': 'pjsip:server',
                    'Body': self.alert_message
                }
            )
            status = response.get("Response", "Unknown")
            message = response.get("Message", "")
            logger.info(f"[📨 SMS] → {extension} | Status: {status} - {message}")
        except Exception as e:
            logger.error(f"[📨 SMS] → {extension} | ❌ Error: {e}")

    async def close(self):
        """Disconnect safely from the AMI server."""
        if self.is_connected:
            self.manager.close()
            self.is_connected = False
            logger.info("[AMI] 🔌 Disconnected from AMI server.")
