from transformers import pipeline

print("Loading FinBERT model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT loaded successfully!\n")

headlines = [
    "Company reports stronger than expected quarterly earnings",
    "Company warns of weaker demand in the coming quarter",
    "Investors remain cautious amid rising market uncertainty"
]

for headline in headlines:
    result = sentiment_pipeline(headline)[0]

    print(f"Headline: {headline}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['score']:.4f}")
    print("-" * 80)