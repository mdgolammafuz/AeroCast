from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType,
    StringType,
    DoubleType,
    IntegerType,
)

spark = (
    SparkSession.builder
    .appName("NOAA-Streaming")
    .getOrCreate()
)

# reduce spam
spark.sparkContext.setLogLevel("WARN")

schema = (
    StructType()
    .add("ts", StringType())
    .add("station", StringType())
    .add("temperature", DoubleType())
    .add("humidity", DoubleType())
    .add("windspeed", DoubleType())
    .add("v", IntegerType())
)

df_raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "noaa-weather")
    .load()
)

df_parsed = (
    df_raw
    .selectExpr("CAST(value AS STRING) AS json")
    .select(from_json(col("json"), schema).alias("data"))
    .select("data.*")
)

# prettier console: 1 row per batch, full columns, every 5s
console_q = (
    df_parsed
    .writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", False)
    .option("numRows", 20)
    .trigger(processingTime="5 seconds")
    .start()
)

parquet_q = (
    df_parsed
    .writeStream
    .format("parquet")
    .option("path", "data/processed/noaa/")
    .option("checkpointLocation", "data/processed/noaa/_checkpoints")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

console_q.awaitTermination()
parquet_q.awaitTermination()
