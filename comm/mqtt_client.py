import asyncio
import json
import logging
from typing import Dict, Any, Optional
import aiomqtt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 0,
        max_queue_size: int = 1000,
        max_retries: int = 10,
        retry_delay: int = 5
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.qos = qos
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._retry_count = 0

    async def start(self):
        """Start the MQTT client"""
        self._running = True
        await self.run_forever()

    async def stop(self):
        """Stop the MQTT client gracefully"""
        self._running = False
        logger.info("MQTT client stopping...")

    async def run_forever(self):
        """
        Main loop: connect to broker, subscribe and process incoming messages.
        Will retry on failure with limit.
        """
        while self._running and self._retry_count < self.max_retries:
            try:
                async with aiomqtt.Client(
                    hostname=self.broker,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                ) as client:
                    logger.info(f"Connected to {self.broker}:{self.port}")
                    await client.subscribe(self.topic, qos=self.qos)
                    logger.info(f"Subscribed to {self.topic} (QoS={self.qos})")
                    
                    # Reset retry count on successful connection
                    self._retry_count = 0
                    
                    async for message in client.messages:
                        if not self._running:
                            break
                            
                        await self._process_message(message)
                        
            except asyncio.CancelledError:
                logger.info("MQTT client cancelled")
                break
            except Exception as e:
                self._retry_count += 1
                logger.error(f"Connection error (attempt {self._retry_count}/{self.max_retries}): {e}")
                
                if self._retry_count < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached, stopping client")
                    break

    async def _process_message(self, message):
        """Process individual MQTT message"""
        try:
            raw_payload = message.payload.decode(errors="ignore")
            logger.debug(f"Raw message on {message.topic}: {raw_payload}")
            
            try:
                data = json.loads(raw_payload)
                normalized = self._normalize_data(data)
                logger.debug(f"Parsed: {normalized}")
                
                # Non-blocking put with timeout
                try:
                    await asyncio.wait_for(
                        self.message_queue.put(normalized), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Message queue full, dropping message")
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON: {raw_payload}")
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize incoming data"""
        return {
            "device_id": data.get("device_id", "unknown"),
            "fall_detected": bool(data.get("fall_detected", False)),
            "latitude": self._to_float(data.get("latitude")),
            "longitude": self._to_float(data.get("longitude")),
            "has_gps_fix": self._to_bool(data.get("has_gps_fix")),
            "timestamp": data.get("timestamp")  # Preserve original timestamp if exists
        }

    async def get_message(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve message from the internal queue (already normalized dict).
        
        Args:
            timeout: Maximum time to wait for a message (None = wait forever)
            
        Returns:
            Normalized message dictionary
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        if timeout is not None:
            return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
        return await self.message_queue.get()

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.message_queue.qsize()

    def is_running(self) -> bool:
        """Check if client is running"""
        return self._running

    @staticmethod
    def _to_float(value) -> float:
        """Convert value to float with fallback"""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0

    @staticmethod
    def _to_bool(value) -> bool:
        """Convert value to boolean with smart parsing"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower().strip() in ("true", "1", "yes", "y", "on")
        try:
            return bool(int(value))
        except (ValueError, TypeError):
            return bool(value)

# Example usage
async def main():
    client = MQTTClient(
        broker="localhost",
        port=1883,
        topic="fall/detection",
        max_queue_size=500
    )
    
    # Start client in background
    client_task = asyncio.create_task(client.start())
    
    try:
        # Process messages
        while client.is_running():
            try:
                message = await client.get_message(timeout=5.0)
                print(f"Received: {message}")
            except asyncio.TimeoutError:
                print("No message received in 5 seconds")
            except Exception as e:
                logger.error(f"Error getting message: {e}")
                
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.stop()
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())
