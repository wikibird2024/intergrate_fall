import asyncio
from panoramisk.manager import Manager
from config.config import AMI_HOST, AMI_PORT, AMI_USERNAME, AMI_SECRET
from config.config import EXTENSIONS, ALERT_MESSAGE, CALLER_ID


class AMITrigger:
    def __init__(self):
        self.manager = Manager(
            host=AMI_HOST, port=AMI_PORT, username=AMI_USERNAME, secret=AMI_SECRET
        )
        self.devices = EXTENSIONS
        self.alert_message = ALERT_MESSAGE
        self.caller_id = CALLER_ID

    async def send_alert_calls(self):
        print(f"[AMI] Triggering alert to {len(self.devices)} device(s)...")

        tasks = [self._originate_call(extension) for extension in self.devices]
        await asyncio.gather(*tasks)

    async def _originate_call(self, extension):
        try:
            response = await self.manager.send_action(
                {
                    "Action": "Originate",
                    "Channel": f"PJSIP/{extension}",
                    "Context": "internal",
                    "Exten": extension,
                    "Priority": 1,
                    "CallerID": f"{self.caller_id} <{extension}>",
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

    async def trigger(self):
        await self.manager.connect()
        await self.send_alert_calls()
        await asyncio.sleep(2)
        self.manager.close()
        print("[AMI] Done triggering and disconnected.")


if __name__ == "__main__":
    asyncio.run(AMITrigger().trigger())
