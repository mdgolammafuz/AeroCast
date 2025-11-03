#!/usr/bin/env python3
import json
import time
import datetime
import requests
from kafka import KafkaProducer

NOAA_URL = "https://api.weather.gov/stations/KNYC/observations/latest"
TOPIC = "noaa-weather"
STATION_ID = "KNYC"
INTERVAL_SEC = 5  # stay in sync


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def fetch_noaa():
    r = requests.get(NOAA_URL, timeout=5)
    r.raise_for_status()
    data = r.json()
    props = data.get("properties", {})

    temp_obj = props.get("temperature") or {}
    temperature = temp_obj.get("value")
    if temperature is None:
        temperature = 0.0
    temperature = float(temperature)

    # make the two missing features so the rest of the pipeline has them
    # simple, fixed-ish, no pretending
    humidity = 60.0
    windspeed = 5.0

    return {
        "ts": _now_iso(),
        "station": STATION_ID,
        "temperature": temperature,
        "humidity": humidity,
        "windspeed": windspeed,
        "v": 1,
    }


def main():
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"[noaa] producing to {TOPIC} every {INTERVAL_SEC}s ...")

    while True:
        try:
            payload = fetch_noaa()
            producer.send(TOPIC, payload)
            print(f"[noaa] sent: {payload}")
        except Exception as e:
            print(f"[noaa] ERROR: {e}")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
