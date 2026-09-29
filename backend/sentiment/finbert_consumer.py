import json

from confluent_kafka import Consumer
from transformers import pipeline

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "financial-news"

print("Loading FinBERT model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT loaded successfully!")
print("Starting Kafka Financial News + FinBERT Consumer...\n")

consumer = Consumer({
    "bootstrap.servers": KAFKA_SERVER,
    "group.id": "quantformer-finbert-consumer",
    "auto.offset.reset": "latest",
})

consumer.subscribe([TOPIC_NAME])

print("Listening for financial news...\n")

try:
    while True:
        message = consumer.poll(1.0)

        if message is None:
            continue

        if message.error():
            print(f"Kafka error: {message.error()}")
            continue

        news = json.loads(message.value().decode("utf-8"))

        headline = news["headline"]

        result = sentiment_pipeline(headline)[0]

        sentiment = result["label"]
        confidence = result["score"]

        print(f"Time: {news['timestamp']}")
        print(f"Symbol: {news['symbol']}")
        print(f"Category: {news['category']}")
        print(f"Headline: {headline}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 80)

except KeyboardInterrupt:
    print("\nFinBERT Consumer stopped.")

finally:
    consumer.close()