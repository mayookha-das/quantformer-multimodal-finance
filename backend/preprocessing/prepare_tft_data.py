
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORDERBOOK_FILE = (
    PROJECT_ROOT / "backend" / "data" / "processed_orderbook.jsonl"
)
NEWS_FILE = (
    PROJECT_ROOT / "backend" / "data" / "processed_news.jsonl"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "backend" / "data" / "tft_minute_data.csv"
)

TARGET_HORIZON_MINUTES = 5
SESSION_FREQUENCY = pd.Timedelta(minutes=1)
READ_CHUNK_SIZE = 50_000


# ---------------------------------------------------------
# Load JSONL data
# ---------------------------------------------------------

def load_jsonl(file_path):
    """Read JSONL records in chunks to limit memory usage."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    chunks = []
    records = []
    total_records = 0

    print(f"Reading {file_path.name}...")

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON at "
                    f"{file_path.name}:{line_number}"
                )
                continue

            if len(records) >= READ_CHUNK_SIZE:
                chunks.append(pd.DataFrame(records))
                total_records += len(records)
                records = []

    if records:
        chunks.append(pd.DataFrame(records))
        total_records += len(records)

    if not chunks:
        raise ValueError(
            f"No valid records found in {file_path.name}"
        )

    print(f"Loaded records: {total_records:,}")

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------
# Prepare order-book data
# ---------------------------------------------------------

def prepare_orderbook():
    df = load_jsonl(ORDERBOOK_FILE)

    required_columns = [
        "timestamp",
        "mid_price",
        "spread",
        "best_bid",
        "best_ask",
        "total_bid_volume",
        "total_ask_volume",
        "orderbook_imbalance",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing order-book columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = required_columns[1:]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df = df[
        (df["mid_price"] > 0)
        & (df["best_bid"] > 0)
        & (df["best_ask"] > 0)
        & (df["total_bid_volume"] >= 0)
        & (df["total_ask_volume"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid order-book records remain."
        )

    # Sort before aggregation.
    df = df.sort_values("timestamp")

    print(
        "Order-book timestamp range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    # Use the last observed prices and mean volume/imbalance
    # for each minute containing order-book updates.
    df = df.set_index("timestamp")

    minute_orderbook = df.resample("1min").agg({
        "mid_price": "last",
        "spread": "mean",
        "best_bid": "last",
        "best_ask": "last",
        "total_bid_volume": "mean",
        "total_ask_volume": "mean",
        "orderbook_imbalance": "mean",
    })

    minute_orderbook = (
        minute_orderbook
        .dropna(subset=["mid_price"])
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "Order-book minute rows:",
        len(minute_orderbook),
    )

    return minute_orderbook


# ---------------------------------------------------------
# Prepare financial-news sentiment
# ---------------------------------------------------------

def prepare_news():
    df = load_jsonl(NEWS_FILE)

    required_columns = [
        "timestamp",
        "sentiment_score",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing news columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce",
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid financial-news sentiment records remain."
        )

    df = df.sort_values("timestamp")

    print(
        "News timestamp range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    # Average news sentiment within each minute.
    df = df.set_index("timestamp")

    minute_news = (
        df.resample("1min")
        .agg({"sentiment_score": "mean"})
        .dropna()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "News sentiment minute rows:",
        len(minute_news),
    )

    return minute_news


# ---------------------------------------------------------
# Align data, identify sessions, and create target
# ---------------------------------------------------------

def build_dataset(orderbook, news):
    print("\nAligning news sentiment to order-book minutes...")

    # Use only news available at or before each order-book
    # timestamp. Limit matching to recent news.
    minute_data = pd.merge_asof(
        orderbook.sort_values("timestamp"),
        news.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=SESSION_FREQUENCY,
    )

    missing_sentiment = int(
        minute_data["sentiment_score"].isna().sum()
    )

    # Zero is a neutral fallback, but does not mean news
    # was actually observed during that minute.
    minute_data["sentiment_score"] = (
        minute_data["sentiment_score"].fillna(0.0)
    )

    print(
        "Minutes without a matched recent news score:",
        missing_sentiment,
    )

    minute_data = minute_data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Start a new session when timestamps are not exactly
    # one minute apart. Never bridge a recording gap.
    time_gap = minute_data["timestamp"].diff()

    new_session = (
        time_gap.isna()
        | time_gap.ne(SESSION_FREQUENCY)
    )

    minute_data["session_id"] = (
        new_session.cumsum().astype(int)
    )

    print("\nCreating five-minute-ahead price targets...")

    minute_data["target_price_5min"] = (
        minute_data.groupby("session_id")["mid_price"]
        .shift(-TARGET_HORIZON_MINUTES)
    )

    # Last five rows of each session lack future targets.
    minute_data = minute_data.dropna(
        subset=["target_price_5min"]
    ).copy()

    # Rebuild IDs and indices after removing target rows.
    time_gap = minute_data["timestamp"].diff()

    new_session = (
        time_gap.isna()
        | time_gap.ne(SESSION_FREQUENCY)
    )

    minute_data["session_id"] = (
        new_session.cumsum().astype(int)
    )

    minute_data["time_idx"] = (
        minute_data.groupby("session_id").cumcount()
    )

    minute_data["symbol"] = "MOCK"

    output_columns = [
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
        "session_id",
    ]

    minute_data = minute_data[output_columns]

    if minute_data.empty:
        raise ValueError(
            "No rows remain after target creation."
        )

    return minute_data


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("Starting full-history TFT preprocessing...")

    orderbook = prepare_orderbook()
    news = prepare_news()

    minute_data = build_dataset(
        orderbook,
        news,
    )

    session_lengths = (
        minute_data.groupby("session_id").size()
    )

    print("\nSession lengths after target creation:")
    print(session_lengths.to_string())

    print("\nData-quality summary")
    print("Minute-level rows:", len(minute_data))
    print("Continuous sessions:", len(session_lengths))
    print(
        "Longest session:",
        int(session_lengths.max()),
        "minutes",
    )
    print(
        "Sessions with at least 25 rows:",
        int((session_lengths >= 25).sum()),
    )
    print(
        "Sessions with at least 100 rows:",
        int((session_lengths >= 100).sum()),
    )
    print(
        "Sessions with at least 500 rows:",
        int((session_lengths >= 500).sum()),
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    minute_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nPreprocessing completed successfully.")
    print("Output file:", OUTPUT_FILE)
    print("Saved rows:", len(minute_data))
    print("\nFirst five rows:")
    print(minute_data.head().to_string(index=False))


if __name__ == "__main__":
    main()
