import asyncio
from panoramisk.manager import Manager
from config.config import EXTENSIONS, ALERT_MESSAGE, CALLER_ID

class AMITrigger:
    def __init__(self, host, port, username, secret):
        # The Manager object stores config internally.
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
            # Access the host and port from the AMITrigger object, not the Manager object
            print(f"[AMI] Connecting to AMI at {self.host}:{self.port}...")
            await self.manager.connect()
            self.is_connected = True
            print("[AMI] ✅ Connected to AMI server.")
        except Exception as e:
            print(f"[AMI] ❌ Failed to connect to AMI: {e}")
            self.is_connected = False
            raise

    async def alert_devices(self, message: str):
        """Trigger an alert by initiating calls to a list of extensions."""
        if not self.is_connected:
            print("[AMI] ⚠️ Not connected to AMI. Alert not sent.")
            return

        print(f"[AMI] Triggering alert to {len(self.extensions)} device(s)...")

        tasks = [self._originate_call(extension) for extension in self.extensions]
        await asyncio.gather(*tasks)

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
                print(f"[📞 CALL] → {extension} | Status: {status} - {message}")
            else:
                print(f"[📞 CALL] → {extension} | ❌ Invalid AMI response: {response}")

        except Exception as e:
            print(f"[📞 CALL] → {extension} | ❌ Error: {e}")

    async def close(self):
        """Disconnect safely from the AMI server."""
        if self.is_connected:
            self.manager.close()
            self.is_connected = False
            print("[AMI] 🔌 Disconnected from AMI server.")
