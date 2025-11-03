# streaming/processing/noaa_spark.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

NOAA_TOPIC = "noaa-weather"
PARQUET_PATH = "data/processed/noaa"
CHECKPOINT_PATH = "data/checkpoints/noaa"

spark = (
    SparkSession.builder
    .appName("AeroCast-NOAA-Stream")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

schema = StructType(
    [
        StructField("ts", StringType(), True),
        StructField("station", StringType(), True),
        StructField("temperature", DoubleType(), True),
        # producer currently doesn't send these, we default them
        StructField("humidity", DoubleType(), True),
        StructField("rainfall", DoubleType(), True),
        # producer sends this but we don't need it downstream
        StructField("v", IntegerType(), True),
    ]
)

# 1) read from kafka
raw_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", NOAA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

# 2) parse json payload
json_df = raw_df.select(from_json(col("value").cast("string"), schema).alias("data"))

parsed_df = (
    json_df.select(
        col("data.ts").alias("ts"),
        col("data.station").alias("station"),
        col("data.temperature").alias("temperature"),
        col("data.humidity").alias("humidity"),
        col("data.rainfall").alias("rainfall"),
        # we drop v here on purpose
    )
    .fillna({"humidity": 50.0, "rainfall": 0.0})
)

# tiny progress logger (what you saw earlier)
def log_progress(batch_df, batch_id):
    rows = batch_df.count()
    print(f'[noaa_stream] progress: {{"epoch": {batch_id}, "rows": {rows}}}')

progress_q = (
    parsed_df.writeStream
    .foreachBatch(log_progress)
    .outputMode("update")
    .start()
)

# 3) write to parquet (for training script)
parquet_q = (
    parsed_df.writeStream
    .format("parquet")
    .option("path", PARQUET_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

# 4) show nice table in terminal again
console_q = (
    parsed_df.writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", "false")
    .trigger(processingTime="5 seconds")
    .start()
)

# wait for any of them
spark.streams.awaitAnyTermination()
