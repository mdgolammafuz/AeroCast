# 📦 STEP 1: Start Zookeeper (Required for Kafka)
brew services start zookeeper

# 📦 STEP 2: Start Kafka Broker
brew services start kafka

# 🧪 STEP 3: Confirm Kafka is running at localhost:9092
lsof -i :9092   # Should show Kafka listening

# 📍 STEP 4: (Optional but recommended) Create the Kafka topic manually
kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 1 \
  --topic weather-data

# ✅ STEP 5: Confirm topic exists
kafka-topics --list --bootstrap-server localhost:9092
