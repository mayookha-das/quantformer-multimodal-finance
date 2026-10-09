import json
from pathlib import Path

import torch
import pandas as pd

DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "aligned_multimodal.jsonl"
)


def validate_data():
    records = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    print(f"Total records: {len(records)}")

    missing_values = 0
    invalid_embeddings = 0
    invalid_sentiment = 0
    invalid_orderbook_features = 0

    for record in records:

        # Check required fields
        required_fields = [
            "sentiment_score",
            "embedding",
            "mid_price",
            "spread",
            "best_bid",
            "best_ask",
            "total_bid_volume",
            "total_ask_volume",
            "orderbook_imbalance",
        ]

        if any(
            field not in record or record[field] is None
            for field in required_fields
        ):
            missing_values += 1

        # Check FinBERT embedding
        if len(record["embedding"]) != 768:
            invalid_embeddings += 1

        # Check sentiment score
        sentiment_score = record["sentiment_score"]

        if not -1 <= sentiment_score <= 1:
            invalid_sentiment += 1

        # Check order-book features
        orderbook_features = [
            record["mid_price"],
            record["spread"],
            record["best_bid"],
            record["best_ask"],
            record["total_bid_volume"],
            record["total_ask_volume"],
            record["orderbook_imbalance"],
        ]

        if len(orderbook_features) != 7:
            invalid_orderbook_features += 1

    print("\nValidation Results")
    print("------------------")
    print(f"Missing values: {missing_values}")
    print(f"Invalid embeddings: {invalid_embeddings}")
    print(f"Invalid sentiment scores: {invalid_sentiment}")
    print(
        f"Invalid order-book feature vectors: "
        f"{invalid_orderbook_features}"
    )

    # Tensor validation
    if records:
        embedding_tensor = torch.tensor(
            records[0]["embedding"],
            dtype=torch.float32,
        )

        orderbook_tensor = torch.tensor(
            [
                records[0]["mid_price"],
                records[0]["spread"],
                records[0]["best_bid"],
                records[0]["best_ask"],
                records[0]["total_bid_volume"],
                records[0]["total_ask_volume"],
                records[0]["orderbook_imbalance"],
            ],
            dtype=torch.float32,
        )

        sentiment_tensor = torch.tensor(
            [records[0]["sentiment_score"]],
            dtype=torch.float32,
        )

        print("\nTensor Validation")
        print("-----------------")
        print("Order-book tensor:", orderbook_tensor.shape)
        print("Sentiment tensor:", sentiment_tensor.shape)
        print("Embedding tensor:", embedding_tensor.shape)

    if (
        missing_values == 0
        and invalid_embeddings == 0
        and invalid_sentiment == 0
        and invalid_orderbook_features == 0
    ):
        print("\nValidation PASSED!")
    else:
        print("\nValidation FAILED!")


def validate_tft_dataset():
    tft_file = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "tft_minute_data.csv"
    )

    if not tft_file.exists():
        print("\nTFT dataset not found.")
        return

    df = pd.read_csv(tft_file)

    required_columns = [
        "time_idx",
        "timestamp",
        "mid_price",
        "sentiment_score",
        "target_price_5min",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    print("\nTFT Dataset Validation")
    print("----------------------")
    print("Records:", len(df))
    print("Missing columns:", missing_columns)
    print("Missing values:", int(df[required_columns].isna().sum().sum())
          if not missing_columns else "Not checked")

    if (
        len(df) > 5
        and not missing_columns
        and not df[required_columns].isna().any().any()
        and df["time_idx"].is_unique
        and df["time_idx"].tolist() == list(range(len(df)))
    ):
        print("TFT dataset validation PASSED!")
    else:
        print("TFT dataset validation FAILED!")



if __name__ == "__main__":
    validate_data()
    validate_tft_dataset()