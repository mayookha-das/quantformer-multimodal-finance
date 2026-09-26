import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


# Kafka configuration
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "orderbook"

# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


# Initial market price
mid_price = 100.00


def generate_order_book():
    """
    Generate a mock Level-2 order book with 5 bid levels
    and 5 ask levels.
    """

    global mid_price

    # Small random price movement
    mid_price += random.uniform(-0.05, 0.05)

    bids = []
    asks = []

    # Generate 5 bid levels and 5 ask levels
    for level in range(1, 6):

        bid_price = round(mid_price - level * 0.01, 2)
        ask_price = round(mid_price + level * 0.01, 2)

        bid_volume = random.randint(50, 500)
        ask_volume = random.randint(50, 500)

        bids.append({
            "price": bid_price,
            "volume": bid_volume
        })

        asks.append({
            "price": ask_price,
            "volume": ask_volume
        })

    order_book = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "MOCK",
        "mid_price": round(mid_price, 2),
        "bids": bids,
        "asks": asks
    }

    return order_book


print("QuantFormer Order Book Producer Started...")
print("Sending mock L2 order-book data to Kafka...")
print("Press Ctrl+C to stop.\n")


try:
    while True:

        order_book = generate_order_book()

        # Send data to Kafka
        producer.send(TOPIC_NAME, value=order_book)

        # Display the generated data
        print(
            f"Time: {order_book['timestamp']} | "
            f"Mid Price: {order_book['mid_price']} | "
            f"Bid: {order_book['bids'][0]['price']} | "
            f"Ask: {order_book['asks'][0]['price']}"
        )

        # Approximately 10 updates per second
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nOrder Book Producer stopped.")

finally:
    producer.flush()
    producer.close()