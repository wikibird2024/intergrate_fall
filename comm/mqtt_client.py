import asyncio
import json
import paho.mqtt.client as mqtt


class MQTTClient:
    def __init__(
        self,
        broker="localhost",
        port=1883,
        topic="fall/detection",
        qos=0,
        keepalive=60,
        loop_start=True,
        enable_log=True,
    ):
        # Initialize MQTT client and callbacks
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # Configuration parameters
        self.broker = broker
        self.port = port
        self.topic = topic
        self.qos = qos
        self.keepalive = keepalive
        self.loop_start_enabled = loop_start
        self.enable_log = enable_log

        # Internal state
        self.latest_message = None
        self.connected = False
        self.subscribed = False

    def log(self, msg):
        """Print log messages if enabled"""
        if self.enable_log:
            print(f"[MQTT] {msg}")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback on successful connection"""
        self.connected = True
        if rc == 0:
            self.log(f"✅ Connected to {self.broker}:{self.port}")
            client.subscribe(self.topic, qos=self.qos)
            self.subscribed = True
            self.log(f"📡 Subscribed to topic: {self.topic} (QoS={self.qos})")
        else:
            self.log(f"❌ Failed to connect (code {rc})")

    def _on_message(self, client, userdata, msg):
        """Callback on incoming message"""
        try:
            payload = msg.payload.decode()
            self.latest_message = json.loads(payload)
            self.log(f"📥 Message received: {self.latest_message}")
        except json.JSONDecodeError as e:
            self.log(f"⚠️ JSON decode error: {e}")

    async def run_forever(self, retry_interval=5):
        """
        Asynchronously connect to broker and start loop.
        Required to be called once to activate MQTT communication.
        """
        while not self.connected:
            try:
                self.client.connect(self.broker, self.port, keepalive=self.keepalive)
                if self.loop_start_enabled:
                    self.client.loop_start()
                await asyncio.sleep(1)
                if self.connected:
                    self.log("✅ MQTT loop started and connected")
                    break
            except Exception as e:
                self.log(f"🔁 Retry in {retry_interval}s due to error: {e}")
                await asyncio.sleep(retry_interval)

    async def wait_for_message(self, timeout=5.0):
        """
        Wait until a new message is received or timeout.
        Returns: JSON message or None
        """
        old = self.latest_message
        elapsed = 0.0
        while elapsed < timeout:
            if self.latest_message != old:
                return self.latest_message
            await asyncio.sleep(0.1)
            elapsed += 0.1
        return None

    def get_latest_message(self):
        """Return the latest received message (may be None)"""
        return self.latest_message

    async def stop(self):
        """Stop MQTT loop and disconnect safely"""
        if self.loop_start_enabled:
            self.client.loop_stop()
        self.client.disconnect()
        self.log("🔌 Disconnected from broker")
