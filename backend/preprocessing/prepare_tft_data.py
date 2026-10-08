import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

ORDERBOOK_FILE = BASE_DIR / "data" / "processed_orderbook.jsonl"
NEWS_FILE = BASE_DIR / "data" / "processed_news.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "tft_minute_data.csv"


# ---------------------------------------------------------
# Load JSONL files
# ---------------------------------------------------------

def load_jsonl(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# Load today's data

TARGET_DATE = "2026-10-08"

orderbook_records = load_jsonl(ORDERBOOK_FILE)
news_records = load_jsonl(NEWS_FILE)

orderbook_records = [
    record
    for record in orderbook_records
    if record["timestamp"].startswith(TARGET_DATE)
]

news_records = [
    record
    for record in news_records
    if record["timestamp"].startswith(TARGET_DATE)
]

print("October 8 order-book records:", len(orderbook_records))
print("October 8 news records:", len(news_records))

# ---------------------------------------------------------
# Convert to DataFrames
# ---------------------------------------------------------

orderbook_df = pd.DataFrame(orderbook_records)
news_df = pd.DataFrame(news_records)

orderbook_df["timestamp"] = pd.to_datetime(
    orderbook_df["timestamp"],
    utc=True
)

news_df["timestamp"] = pd.to_datetime(
    news_df["timestamp"],
    utc=True
)


# ---------------------------------------------------------
# Aggregate high-frequency order book to 1-minute data
# ---------------------------------------------------------

orderbook_df = orderbook_df.set_index("timestamp")

minute_orderbook = orderbook_df.resample("1min").agg(
    {
        "mid_price": "last",
        "spread": "mean",
        "best_bid": "last",
        "best_ask": "last",
        "total_bid_volume": "mean",
        "total_ask_volume": "mean",
        "orderbook_imbalance": "mean",
    }
)

minute_orderbook = minute_orderbook.dropna().reset_index()


# ---------------------------------------------------------
# Aggregate financial news to 1-minute data
# ---------------------------------------------------------

news_df = news_df.set_index("timestamp")

minute_news = news_df.resample("1min").agg(
    {
        "sentiment_score": "mean",
    }
)

minute_news = minute_news.reset_index()


# ---------------------------------------------------------
# Merge order book + news
# ---------------------------------------------------------

minute_data = pd.merge_asof(
    minute_orderbook.sort_values("timestamp"),
    minute_news.sort_values("timestamp"),
    on="timestamp",
    direction="backward",
    tolerance=pd.Timedelta("1min"),
)

minute_data["sentiment_score"] = (
    minute_data["sentiment_score"].fillna(0.0)
)


# ---------------------------------------------------------
# Create 5-minute future target
# ---------------------------------------------------------

minute_data["target_price_5min"] = (
    minute_data["mid_price"].shift(-5)
)


# ---------------------------------------------------------
# Remove rows without a future target
# ---------------------------------------------------------

minute_data = minute_data.dropna(
    subset=["target_price_5min"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Add time index for TFT
# ---------------------------------------------------------

minute_data["time_idx"] = range(len(minute_data))

minute_data["symbol"] = "MOCK"

minute_data = minute_data[
    [
        "time_idx",
        "timestamp",
        "symbol",
        "mid_price",
        "spread",
        "best_bid",
        "best_ask",
        "total_bid_volume",
        "total_ask_volume",
        "orderbook_imbalance",
        "sentiment_score",
        "target_price_5min",
    ]
]


# ---------------------------------------------------------
# Save TFT-ready dataset
# ---------------------------------------------------------

minute_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nTFT preprocessing completed.")
print("Minute-level records:", len(minute_data))
print("Output file:", OUTPUT_FILE)

print("\nColumns:")
for column in minute_data.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(minute_data.head())

print("\nLast 5 rows:")
print(minute_data.tail())