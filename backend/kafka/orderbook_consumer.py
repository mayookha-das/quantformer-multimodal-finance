from pathlib import Path
import json

from kafka import KafkaConsumer

from backend.preprocessing.orderbook_features import extract_orderbook_features


# Kafka configuration
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "orderbook"

OUTPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed_orderbook.jsonl"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


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

        processed_orderbook = {
            "timestamp": features["timestamp"],
            "symbol": features["symbol"],
            "mid_price": features["mid_price"],
            "spread": features["spread"],
            "best_bid": features["best_bid"],
            "best_ask": features["best_ask"],
            "total_bid_volume": features["total_bid_volume"],
            "total_ask_volume": features["total_ask_volume"],
            "orderbook_imbalance": features["orderbook_imbalance"],
        }

        with OUTPUT_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(processed_orderbook) + "\n")

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