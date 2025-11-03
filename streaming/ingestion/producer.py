import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

# Kafka configuration
KAFKA_TOPIC = "weather-data"
KAFKA_BROKER = "localhost:9092"

# Create Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_sensor_data():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "temperature": round(random.uniform(20.0, 40.0), 2),
        "humidity": round(random.uniform(30.0, 90.0), 2),
        "rainfall": round(random.uniform(0.0, 20.0), 2)
    }

if __name__ == "__main__":
    print(f"Sending data to Kafka topic '{KAFKA_TOPIC}'... (Press CTRL+C to stop)")
    try:
        while True:
            message = generate_sensor_data()
            print("Sending:", message)
            producer.send(KAFKA_TOPIC, message)
            time.sleep(3)  # simulate every 3 seconds
    except KeyboardInterrupt:
        print("\n Stopped streaming.")
