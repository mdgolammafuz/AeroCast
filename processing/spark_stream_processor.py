from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType

# Define the JSON schema for incoming sensor data
schema = StructType() \
    .add("timestamp", StringType()) \
    .add("temperature", DoubleType()) \
    .add("humidity", DoubleType()) \
    .add("rainfall", DoubleType())

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("KafkaSensorConsumer") \
    .getOrCreate()

# Step 2: Read from Kafka topic 'weather-data'
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "weather-data") \
    .load()

# Step 3: Parse the Kafka 'value' field (bytes) into JSON using schema
df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# Step 4a: Write to console (for dev/debugging)
console_query = df_parsed.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

# Step 4b: Persist to Parquet (production-ready sink)
parquet_query = df_parsed.writeStream \
    .format("parquet") \
    .option("path", "data/processed/") \
    .option("checkpointLocation", "data/checkpoints/") \
    .outputMode("append") \
    .start()

# Await termination of both sinks
console_query.awaitTermination()
parquet_query.awaitTermination()
