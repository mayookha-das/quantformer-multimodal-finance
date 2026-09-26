import json

from kafka import KafkaConsumer

from backend.preprocessing.orderbook_features import extract_orderbook_features


# Kafka configuration
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "orderbook"


# Create Kafka consumer
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_SERVER,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="quantformer-feature-consumer",
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)


print("QuantFormer Order Book Feature Consumer Started...")
print("Listening for order-book data...\n")


try:
    for message in consumer:

        order_book = message.value

        # Extract quantitative features
        features = extract_orderbook_features(order_book)

        print(
            f"Time: {features['timestamp']} | "
            f"Mid Price: {features['mid_price']}"
        )

        print(
            f"Spread: {features['spread']} | "
            f"Best Bid: {features['best_bid']} | "
            f"Best Ask: {features['best_ask']}"
        )

        print(
            f"Total Bid Volume: {features['total_bid_volume']} | "
            f"Total Ask Volume: {features['total_ask_volume']}"
        )

        print(
            f"Order Book Imbalance: "
            f"{features['orderbook_imbalance']}"
        )

        print("-" * 80)


except KeyboardInterrupt:
    print("\nOrder Book Feature Consumer stopped.")

finally:
    consumer.close()