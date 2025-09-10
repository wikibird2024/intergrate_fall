
import asyncio
import aiomqtt


class MQTTClient:
    """
    Asynchronous MQTT client for connecting to a broker and handling messages.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "fall/detection",
        username: str | None = None,
        password: str | None = None,
        qos: int = 0,
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.qos = qos
        self.message_queue = asyncio.Queue()

    async def run_forever(self):
        """
        Main loop: connect to broker, subscribe and process incoming messages.
        Will retry on failure.
        """
        while True:
            try:
                async with aiomqtt.Client(
                    hostname=self.broker,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                ) as client:
                    print(f"[MQTT] ✅ Connected to {self.broker}:{self.port}")
                    await client.subscribe(self.topic, qos=self.qos)
                    print(f"[MQTT] 📡 Subscribed to {self.topic} (QoS={self.qos})")

                    async for message in client.messages:
                        try:
                            payload = message.payload.decode()
                            print(f"[MQTT] 📥 Received: {payload}")
                            await self.message_queue.put(payload)
                        except Exception as e:
                            print(f"[MQTT] ⚠️ Error parsing message: {e}")

            except Exception as e:
                print(f"[MQTT] ❌ Unexpected error: {e}, retrying in 5s...")
                await asyncio.sleep(5)

    async def get_message(self):
        """Retrieve message from the internal queue."""
        return await self.message_queue.get()
