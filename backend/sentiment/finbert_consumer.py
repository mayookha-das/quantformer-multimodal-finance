import json
from pathlib import Path
import torch

from confluent_kafka import Consumer
from transformers import pipeline, AutoTokenizer, AutoModel

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "financial-news"

OUTPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed_news.jsonl"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading FinBERT model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)
print("Loading FinBERT embedding model...")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
embedding_model = AutoModel.from_pretrained("ProsusAI/finbert")

embedding_model.eval()

print("FinBERT embedding model loaded successfully!")

print("FinBERT loaded successfully!")
print("Starting Kafka Financial News + FinBERT Consumer...\n")

def calculate_sentiment_score(label, confidence):
    """Convert FinBERT sentiment into a signed score from -1 to +1."""

    label = label.lower()

    if label == "positive":
        return round(confidence, 4)

    if label == "negative":
        return round(-confidence, 4)

    return 0.0
def generate_embedding(text):
    """Generate a dense FinBERT embedding for a financial headline."""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = embedding_model(**inputs)

    # Mean pooling across the token dimension
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.tolist()
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

        sentiment_score = calculate_sentiment_score(
            sentiment,
            confidence
        )
        embedding = generate_embedding(headline)

        print(f"Time: {news['timestamp']}")
        print(f"Symbol: {news['symbol']}")
        print(f"Category: {news['category']}")
        print(f"Headline: {headline}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Sentiment Score: {sentiment_score:+.4f}")
        print(f"Embedding Dimensions: {len(embedding)}")
        print(f"Embedding Preview: {embedding[:5]}")
        processed_news = {
            "timestamp": news["timestamp"],
            "symbol": news["symbol"],
            "category": news["category"],
            "headline": headline,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "sentiment_score": sentiment_score,
            "embedding": embedding,
        }

        with OUTPUT_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(processed_news) + "\n")
        print("-" * 80)

except KeyboardInterrupt:
    print("\nFinBERT Consumer stopped.")

finally:
    consumer.close()