import json

from kafka import KafkaConsumer


# Kafka configuration
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "financial-news"


# Create Kafka consumer
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_SERVER,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="quantformer-news-consumer",
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)


print("QuantFormer Financial News Consumer Started...")
print("Listening for financial news...\n")


try:
    for message in consumer:

        news = message.value

        print(
            f"Time: {news['timestamp']} | "
            f"Symbol: {news['symbol']}"
        )

        print(
            f"Category: {news['category']}"
        )

        print(
            f"Headline: {news['headline']}"
        )

        print("-" * 80)


except KeyboardInterrupt:
    print("\nFinancial News Consumer stopped.")

finally:
    consumer.close()