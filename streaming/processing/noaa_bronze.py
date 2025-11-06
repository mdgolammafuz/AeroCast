import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)
from prometheus_client import CollectorRegistry, Counter, push_to_gateway

PUSHGATEWAY_HOST = os.environ.get("PUSHGATEWAY_HOST", "localhost:9091")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRONZE_DIR = os.path.join(ROOT, "data", "processed", "noaa")
BAD_DIR = os.path.join(ROOT, "data", "bad", "noaa")

os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

registry = CollectorRegistry()
INGEST_OK = Counter(
    "ingest_records_total",
    "Total good records ingested from NOAA Kafka",
    registry=registry,
)
INGEST_BAD = Counter(
    "ingest_bad_records_total",
    "Total bad records from NOAA Kafka",
    registry=registry,
)


def push_metrics(good_cnt: int, bad_cnt: int):
    if good_cnt:
        INGEST_OK.inc(good_cnt)
    if bad_cnt:
        INGEST_BAD.inc(bad_cnt)
    try:
        push_to_gateway(PUSHGATEWAY_HOST, job="aerocast_ingest", registry=registry)
    except Exception:
        pass


spark = (
    SparkSession.builder.appName("noaa-kafka-to-bronze")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

kafka_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "noaa-weather")
    .option("startingOffsets", "latest")
    .load()
)

payload_schema = StructType(
    [
        StructField("ts", StringType(), True),
        StructField("station", StringType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("windspeed", DoubleType(), True),
        StructField("pressure", DoubleType(), True),
        StructField("v", IntegerType(), True),
    ]
)

parsed = (
    kafka_df.select(F.col("value").cast("string").alias("json_str"))
    .select(F.from_json("json_str", payload_schema).alias("data"))
    .select("data.*")
)

good = (
    parsed.filter(
        (F.col("v") == 1)
        & F.col("ts").isNotNull()
        & F.col("temperature").isNotNull()
        & F.col("windspeed").isNotNull()
        & F.col("pressure").isNotNull()
    )
)

bad = parsed.filter(
    (F.col("v").isNull())
    | (F.col("v") != 1)
    | F.col("ts").isNull()
    | F.col("temperature").isNull()
    | F.col("windspeed").isNull()
    | F.col("pressure").isNull()
)


def write_nonempty_batch(df, batch_id):
    cnt = df.count()
    if cnt == 0:
        return
    (
        df.write
        .mode("append")
        .parquet(BRONZE_DIR)
    )
    push_metrics(cnt, 0)
    print(f"[noaa_bronze] batch={batch_id} rows={cnt}")


bronze_query = (
    good.writeStream
    .foreachBatch(write_nonempty_batch)
    .option("checkpointLocation", os.path.join(BRONZE_DIR, "_chk"))
    .start()
)

bad_query = (
    bad.writeStream.outputMode("append")
    .format("parquet")
    .option("path", BAD_DIR)
    .option("checkpointLocation", os.path.join(BAD_DIR, "_chk"))
    .start()
)

console_query = (
    good.select("ts", "station", "temperature", "windspeed", "pressure", "v")
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .start()
)

spark.streams.awaitAnyTermination()
