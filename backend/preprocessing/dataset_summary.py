import json
from pathlib import Path


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "aligned_multimodal.jsonl"
)


def create_summary():
    records = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    summary = {
        "dataset_name": "QuantFormer Aligned Multimodal Dataset",
        "records": len(records),
        "modalities": {
            "orderbook": True,
            "financial_news": True,
            "finbert_sentiment": True,
            "finbert_embedding": True,
        },
        "orderbook_feature_count": 7,
        "sentiment_feature_count": 1,
        "finbert_embedding_dimensions": len(
            records[0]["embedding"]
        ),
        "alignment_threshold_seconds": 30,
        "fields": [
            "news_timestamp",
            "orderbook_timestamp",
            "timestamp_difference_seconds",
            "symbol",
            "headline",
            "sentiment",
            "sentiment_score",
            "embedding",
            "mid_price",
            "spread",
            "best_bid",
            "best_ask",
            "total_bid_volume",
            "total_ask_volume",
            "orderbook_imbalance",
        ],
    }

    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    create_summary()