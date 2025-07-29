from panoramisk import Manager
import asyncio

class AMITrigger:
    def __init__(self, host="localhost", port=5038, username="admin", secret="secret"):
        """Initialize Asterisk AMI connection."""
        self.manager = Manager(
            host=host,
            port=port,
            username=username,
            secret=secret
        )
        # Danh sách các thiết bị SIP cần cảnh báo
        self.devices = ["6001", "6002", "6003"]

    async def connect(self):
        """Establish connection to Asterisk AMI."""
        await self.manager.connect()

    async def trigger_call(self, number: str):
        """
        Trigger a SIP call to the specified number/device.
        Parameters:
            number (str): SIP extension (e.g., '6001')
        """
        action = {
            'Action': 'Originate',
            'Channel': f'SIP/{number}',
            'Context': 'default',
            'Exten': number,
            'Priority': '1',
            'CallerID': 'FallAlert <1000>',
            'Timeout': '30000',  # 30 seconds
        }
        await self.manager.send_action(action)

    async def send_sms(self, number: str, message: str):
        """
        Send a message to the specified SIP device.
        Parameters:
            number (str): SIP extension
            message (str): Message content
        """
        action = {
            'Action': 'MessageSend',
            'To': f'SIP/{number}',
            'From': 'FallAlert <1000>',
            'Body': message,
        }
        await self.manager.send_action(action)

    async def alert_devices(self, message: str = "Fall detected!"):
        """
        Trigger both call and SMS alerts to all configured devices.
        Parameters:
            message (str): Alert message content
        """
        for device in self.devices:
            await self.trigger_call(device)
            await self.send_sms(device, message)

    async def close(self):
        """Close the AMI connection gracefully."""
        self.manager.close()

