# streaming/processing/noaa_silver_5min.py
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRONZE_DIR = os.path.join(ROOT, "data", "processed", "noaa")
SILVER_DIR = os.path.join(ROOT, "data", "processed", "noaa_silver_5min")

os.makedirs(SILVER_DIR, exist_ok=True)

# optional: reset this job's checkpoint if schema/window changes
if os.environ.get("CLEAR_SILVER_CHECKPOINT") == "1":
    chk = os.path.join(SILVER_DIR, "_chk")
    if os.path.isdir(chk):
        shutil.rmtree(chk)
        print("[noaa_silver] cleared silver checkpoint")

spark = (
    SparkSession.builder.appName("noaa-bronze-to-silver-5min")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# match bronze schema (temp, windspeed, pressure)
bronze_stream = (
    spark.readStream.format("parquet")
    .schema(
        "ts string, station string, temperature double, windspeed double, pressure double, v int"
    )
    .load(BRONZE_DIR)
)

# timestamp for watermark/window
bronze_ts = bronze_stream.withColumn("ts_ts", F.to_timestamp("ts"))

silver_src = bronze_ts.withWatermark("ts_ts", "10 minutes")

silver_agg = (
    silver_src.groupBy(F.window("ts_ts", "5 minutes").alias("w"))
    .agg(
        F.avg("temperature").alias("avg_temp"),
        F.avg("windspeed").alias("avg_windspeed"),
        F.avg("pressure").alias("avg_pressure"),
        F.count("*").alias("rows"),
    )
    .select(
        F.col("w.start").alias("window_start"),
        F.col("w.end").alias("window_end"),
        "avg_temp",
        "avg_windspeed",
        "avg_pressure",
        "rows",
    )
)

def log_batch(df, batch_id: int):
    cnt = df.count()
    print(f"[noaa_silver] batch={batch_id} rows={cnt}")

silver_q = (
    silver_agg.writeStream.outputMode("append")
    .format("parquet")
    .option("path", SILVER_DIR)
    .option("checkpointLocation", os.path.join(SILVER_DIR, "_chk"))
    .foreachBatch(log_batch)
    .start()
)

spark.streams.awaitAnyTermination()
