
import asyncio
import json
import aiomqtt


class MQTTClient:
    """
    Asynchronous MQTT client for connecting to a broker and handling messages.
    Parses incoming MQTT payloads as JSON and normalizes them.
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
                            data = json.loads(payload)

                            # Chuẩn hóa kiểu dữ liệu
                            normalized = {
                                "device_id": data.get("device_id", "unknown"),
                                "fall_detected": bool(data.get("fall_detected", False)),
                                "latitude": self._to_float(data.get("latitude")),
                                "longitude": self._to_float(data.get("longitude")),
                                "has_gps_fix": self._to_bool(data.get("has_gps_fix")),
                            }

                            print(f"[MQTT] 📥 Parsed: {normalized}")
                            await self.message_queue.put(normalized)

                        except json.JSONDecodeError:
                            print(f"[MQTT] ⚠️ Invalid JSON: {payload}")
                        except Exception as e:
                            print(f"[MQTT] ⚠️ Error parsing message: {e}")

            except Exception as e:
                print(f"[MQTT] ❌ Unexpected error: {e}, retrying in 5s...")
                await asyncio.sleep(5)

    async def get_message(self):
        """Retrieve message from the internal queue (already normalized dict)."""
        return await self.message_queue.get()

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "y")
        return bool(value)
