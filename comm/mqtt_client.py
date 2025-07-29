import asyncio
import json
import paho.mqtt.client as mqtt


class MQTTClient:
    def __init__(self, broker="localhost", port=1883, topic="fall/detection"):
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker = broker
        self.port = port
        self.topic = topic
        self.latest_message = None

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            self.latest_message = json.loads(payload)
            print(f"[MQTT] Message received: {self.latest_message}")
        except json.JSONDecodeError as e:
            print(f"[MQTT] JSON decode error: {e}")

    def start(self):
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    async def wait_for_message(self, timeout=5.0):
        """Wait async for a new MQTT message (default 5s timeout)."""
        old = self.latest_message
        elapsed = 0
        while elapsed < timeout:
            if self.latest_message != old:
                return self.latest_message
            await asyncio.sleep(0.1)
            elapsed += 0.1
        return None  # timeout
