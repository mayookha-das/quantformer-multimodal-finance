import json
from pathlib import Path
from datetime import datetime
from bisect import bisect_left


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

ORDERBOOK_FILE = DATA_DIR / "processed_orderbook.jsonl"
NEWS_FILE = DATA_DIR / "processed_news.jsonl"
OUTPUT_FILE = DATA_DIR / "aligned_multimodal.jsonl"


def parse_timestamp(timestamp):
    """Convert ISO-8601 timestamp into a datetime object."""
    return datetime.fromisoformat(timestamp)


def load_jsonl(file_path):
    """Load JSON Lines data into a list."""
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def align_news_with_orderbook(orderbook_records, news_records):
    """Match each news event with the nearest order-book timestamp."""

    orderbook_records.sort(
        key=lambda record: parse_timestamp(record["timestamp"])
    )

    orderbook_times = [
        parse_timestamp(record["timestamp"])
        for record in orderbook_records
    ]

    aligned_records = []

    for news in news_records:
        news_time = parse_timestamp(news["timestamp"])

        position = bisect_left(orderbook_times, news_time)

        candidates = []

        if position > 0:
            candidates.append(orderbook_records[position - 1])

        if position < len(orderbook_records):
            candidates.append(orderbook_records[position])

        if not candidates:
            continue

        nearest_orderbook = min(
            candidates,
            key=lambda record: abs(
                parse_timestamp(record["timestamp"]) - news_time
            )
        )

        orderbook_time = parse_timestamp(
            nearest_orderbook["timestamp"]
        )

        time_difference = abs(orderbook_time - news_time)

        # Only align events that are reasonably close in time.
        if time_difference.total_seconds() > 30:
            continue

        aligned_record = {
            "news_timestamp": news["timestamp"],
            "orderbook_timestamp": nearest_orderbook["timestamp"],
            "timestamp_difference_seconds": round(
                time_difference.total_seconds(),
                6
            ),
            "symbol": news["symbol"],
            "headline": news["headline"],
            "sentiment": news["sentiment"],
            "sentiment_score": news["sentiment_score"],
            "embedding": news["embedding"],
            "mid_price": nearest_orderbook["mid_price"],
            "spread": nearest_orderbook["spread"],
            "best_bid": nearest_orderbook["best_bid"],
            "best_ask": nearest_orderbook["best_ask"],
            "total_bid_volume": nearest_orderbook["total_bid_volume"],
            "total_ask_volume": nearest_orderbook["total_ask_volume"],
            "orderbook_imbalance": nearest_orderbook[
                "orderbook_imbalance"
            ],
        }

        aligned_records.append(aligned_record)

    return aligned_records


def main():
    print("Loading order-book data...")
    orderbook_records = load_jsonl(ORDERBOOK_FILE)

    print("Loading financial news data...")
    news_records = load_jsonl(NEWS_FILE)

    print(f"Order-book records: {len(orderbook_records)}")
    print(f"News records: {len(news_records)}")

    aligned_records = align_news_with_orderbook(
        orderbook_records,
        news_records
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in aligned_records:
            file.write(json.dumps(record) + "\n")

    print(f"Aligned records: {len(aligned_records)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()