
import json
from pathlib import Path

import torch
from confluent_kafka import Consumer
from transformers import AutoModel, AutoTokenizer, pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "financial-news"
CONSUMER_GROUP = "quantformer-finbert-week3"

# This file is located in:
# backend/preprocessing/sentiment/finbert_consumer.py
# Output will be:
# backend/data/processed_news.jsonl

BACKEND_DIR = Path(__file__).resolve().parents[3]
OUTPUT_FILE = BACKEND_DIR / "backend" / "data" / "processed_news.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD FINBERT
# ============================================================

print("Loading FinBERT sentiment model...")

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="ProsusAI/finbert",
)

print("Loading FinBERT embedding model...")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
embedding_model = AutoModel.from_pretrained("ProsusAI/finbert")
embedding_model.eval()

print("FinBERT models loaded successfully!")
print(f"Processed news output: {OUTPUT_FILE}")


# ============================================================
# 3. SENTIMENT SCORING
# ============================================================

def calculate_sentiment_score(label, confidence):
    """Return a signed sentiment score between -1 and +1."""

    label = str(label).lower()

    if label == "positive":
        return round(float(confidence), 4)

    if label == "negative":
        return round(-float(confidence), 4)

    return 0.0


# ============================================================
# 4. GENERATE FINBERT EMBEDDINGS
# ============================================================

def generate_embedding(text):
    """Generate a 768-dimensional mean-pooled FinBERT embedding."""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = embedding_model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    # Exclude padding tokens when calculating the mean.
    masked_embeddings = token_embeddings * attention_mask
    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = attention_mask.sum(dim=1).clamp(min=1)

    embedding = (summed_embeddings / token_counts).squeeze(0)

    return embedding.cpu().tolist()


# ============================================================
# 5. KAFKA CONSUMER
# ============================================================

consumer = Consumer({
    "bootstrap.servers": KAFKA_SERVER,
    "group.id": CONSUMER_GROUP,
    "auto.offset.reset": "latest",
    "enable.auto.commit": True,
})

consumer.subscribe([TOPIC_NAME])

print("\nQuantFormer FinBERT Kafka consumer started.")
print(f"Listening to topic: {TOPIC_NAME}")
print("Waiting for incoming financial news...\n")


# ============================================================
# 6. PROCESS NEWS STREAM
# ============================================================

try:
    while True:
        message = consumer.poll(timeout=1.0)

        if message is None:
            continue

        if message.error():
            print(f"Kafka error: {message.error()}")
            continue

        try:
            news = json.loads(message.value().decode("utf-8"))

            headline = str(news.get("headline", "")).strip()

            if not headline:
                print("Skipping message: headline is missing.")
                continue

            timestamp = news.get("timestamp")
            symbol = news.get("symbol", "UNKNOWN")
            category = news.get("category", "UNKNOWN")

            # Predict financial sentiment.
            result = sentiment_pipeline(
                headline,
                truncation=True,
                max_length=512,
            )[0]

            sentiment = result["label"].lower()
            confidence = float(result["score"])

            sentiment_score = calculate_sentiment_score(
                sentiment,
                confidence,
            )

            # Generate semantic representation of the headline.
            embedding = generate_embedding(headline)

            processed_news = {
                "timestamp": timestamp,
                "symbol": symbol,
                "category": category,
                "headline": headline,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "sentiment_score": sentiment_score,
                "embedding": embedding,
            }

            # Append one JSON object per line.
            with OUTPUT_FILE.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(processed_news, ensure_ascii=False) + "\n"
                )

            print(f"Time: {timestamp}")
            print(f"Symbol: {symbol}")
            print(f"Category: {category}")
            print(f"Headline: {headline}")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Sentiment score: {sentiment_score:+.4f}")
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"Saved to: {OUTPUT_FILE}")
            print("-" * 80)

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"Skipping malformed Kafka message: {exc}")

        except Exception as exc:
            print(f"Error processing news message: {exc}")

except KeyboardInterrupt:
    print("\nFinBERT consumer stopped by user.")

finally:
    consumer.close()
    print("Kafka consumer closed.")
