
from transformers import pipeline

print("Loading FinBERT model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
)

print("FinBERT loaded successfully!")


def analyze_sentiment(headline: str) -> dict:
    """Analyze a financial headline using FinBERT."""

    if not isinstance(headline, str) or not headline.strip():
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.0,
        }

    result = sentiment_pipeline(headline.strip())[0]

    label = result["label"].lower()
    confidence = float(result["score"])

    # Convert sentiment into a signed score:
    # positive = bullish, negative = bearish.
    if label == "positive":
        signed_score = confidence
    elif label == "negative":
        signed_score = -confidence
    else:
        signed_score = 0.0

    return {
        "sentiment": label,
        "sentiment_score": signed_score,
        "confidence": confidence,
    }


if __name__ == "__main__":
    test_headlines = [
        "Company reports stronger than expected quarterly earnings",
        "Company warns of weaker demand in the coming quarter",
        "Investors remain cautious amid rising market uncertainty",
    ]

    for headline in test_headlines:
        result = analyze_sentiment(headline)

        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Sentiment score: {result['sentiment_score']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 60)
