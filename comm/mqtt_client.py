import asyncio
import json
import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(
        self,
        broker="localhost",
        port=1883,
        topic="fall/detection",
        username=None,  # Added for authentication
        password=None,  # Added for authentication
        qos=0,
        keepalive=60,
        enable_log=True,
        use_tls=False,  # Added for secure connections
    ):
        # Initialize MQTT client with a blank client ID for broker to assign one
        self.client = mqtt.Client(client_id="")

        # Set username and password if provided
        if username and password:
            self.client.username_pw_set(username=username, password=password)

        # Set up TLS/SSL for secure connection if enabled
        if use_tls:
            self.client.tls_set()

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        # Configuration parameters
        self.broker = broker
        self.port = port
        self.topic = topic
        self.qos = qos
        self.keepalive = keepalive
        self.enable_log = enable_log
        self.use_tls = use_tls

        # Internal state
        self.latest_message = None
        self.connected = False
        self.subscribed = False
        
        # Use an asyncio.Queue to store incoming messages
        self.message_queue = asyncio.Queue()

    def log(self, msg):
        """Print log messages if enabled"""
        if self.enable_log:
            print(f"[MQTT] {msg}")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback on successful connection"""
        if rc == 0:
            self.connected = True
            self.log(f"✅ Connected to {self.broker}:{self.port}")
            client.subscribe(self.topic, qos=self.qos)
            self.subscribed = True
            self.log(f"📡 Subscribed to topic: {self.topic} (QoS={self.qos})")
        else:
            self.log(f"❌ Failed to connect (code {rc})")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc):
        """Callback on disconnection"""
        self.connected = False
        self.subscribed = False
        self.log(f"🔌 Disconnected from broker (code {rc})")

    def _on_message(self, client, userdata, msg):
        """
        Callback on incoming message. This runs in a separate thread.
        We use run_coroutine_threadsafe to safely pass the message to the main event loop.
        """
        try:
            payload = msg.payload.decode()
            message_data = json.loads(payload)
            self.latest_message = message_data
            self.log(f"📥 Message received: {self.latest_message}")
            
            # Get the running event loop and put the message in the queue
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(message_data),
                loop
            )
        except json.JSONDecodeError as e:
            self.log(f"⚠️ JSON decode error: {e}")
        except Exception as e:
            self.log(f"⚠️ An error occurred in on_message: {e}")

    async def run_forever(self, retry_interval=5):
        """
        Asynchronously connect to broker and start the network loop.
        This handles retries until a successful connection.
        """
        while not self.connected:
            try:
                self.log(f"Connecting to MQTT broker at {self.broker}:{self.port}...")
                self.client.connect(self.broker, self.port, keepalive=self.keepalive)
                self.client.loop_start()  # Start the network thread
                await asyncio.sleep(1) # Wait briefly for connection to establish
            except Exception as e:
                self.log(f"🔁 Retry in {retry_interval}s due to error: {e}")
                self.connected = False
                await asyncio.sleep(retry_interval)

    async def stop(self):
        """Stop MQTT loop and disconnect safely"""
        self.client.loop_stop()
        self.client.disconnect()
        self.log("🔌 Disconnected from broker")
        await self.message_queue.join()  # Wait for any messages in queue to be processed

    def get_latest_message(self):
        """Return the latest received message (may be None)"""
        return self.latest_message

    def __aiter__(self):
        """Make this class an asynchronous iterable"""
        return self

    async def __anext__(self):
        """
        Return the next message from the queue.
        This allows `async for message in mqtt_client:` to work.
        """
        return await self.message_queue.get()
