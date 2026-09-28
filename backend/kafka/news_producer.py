import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


# Kafka configuration
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "financial-news"


# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


# Mock financial news headlines
NEWS_HEADLINES = [
    {
        "headline": "Company reports stronger than expected quarterly earnings",
        "source": "Mock Financial News",
        "category": "earnings"
    },
    {
        "headline": "Technology company announces major product expansion",
        "source": "Mock Financial News",
        "category": "business"
    },
    {
        "headline": "Company warns of weaker demand in the coming quarter",
        "source": "Mock Financial News",
        "category": "earnings"
    },
    {
        "headline": "Markets react to unexpected economic data",
        "source": "Mock Financial News",
        "category": "economy"
    },
    {
        "headline": "Company announces strategic partnership with global firm",
        "source": "Mock Financial News",
        "category": "business"
    },
    {
        "headline": "Investors remain cautious amid rising market uncertainty",
        "source": "Mock Financial News",
        "category": "markets"
    },
    {
        "headline": "Company raises revenue forecast for the current year",
        "source": "Mock Financial News",
        "category": "earnings"
    },
    {
        "headline": "Shares fall after company reports disappointing results",
        "source": "Mock Financial News",
        "category": "earnings"
    }
]


print("QuantFormer Financial News Producer Started...")
print("Sending mock financial news to Kafka...")
print("Press Ctrl+C to stop.\n")


try:
    while True:

        news = random.choice(NEWS_HEADLINES)

        message = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "MOCK",
            "headline": news["headline"],
            "source": news["source"],
            "category": news["category"]
        }

        # Send news message to Kafka
        producer.send(TOPIC_NAME, value=message)

        print(
            f"Time: {message['timestamp']} | "
            f"Symbol: {message['symbol']} | "
            f"Headline: {message['headline']}"
        )

        # Generate one news event every 2 seconds
        time.sleep(2)


except KeyboardInterrupt:
    print("\nFinancial News Producer stopped.")

finally:
    producer.flush()
    producer.close()