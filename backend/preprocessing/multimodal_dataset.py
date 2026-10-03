import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "aligned_multimodal.jsonl"
)


class MultimodalDataset(Dataset):
    """PyTorch Dataset for aligned order-book and news data."""

    def __init__(self, file_path=DATA_FILE):
        self.file_path = Path(file_path)

        self.records = []

        with self.file_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        orderbook_features = torch.tensor(
            [
                record["mid_price"],
                record["spread"],
                record["best_bid"],
                record["best_ask"],
                record["total_bid_volume"],
                record["total_ask_volume"],
                record["orderbook_imbalance"],
            ],
            dtype=torch.float32,
        )

        sentiment_score = torch.tensor(
            [record["sentiment_score"]],
            dtype=torch.float32,
        )

        embedding = torch.tensor(
            record["embedding"],
            dtype=torch.float32,
        )

        return {
            "orderbook_features": orderbook_features,
            "sentiment_score": sentiment_score,
            "embedding": embedding,
        }


def create_dataloader(batch_size=16):
    """Create a PyTorch DataLoader for multimodal data."""

    dataset = MultimodalDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader


if __name__ == "__main__":
    dataset = MultimodalDataset()

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print(
        "Order-book feature shape:",
        sample["orderbook_features"].shape,
    )

    print(
        "Sentiment score shape:",
        sample["sentiment_score"].shape,
    )

    print(
        "FinBERT embedding shape:",
        sample["embedding"].shape,
    )

    dataloader = create_dataloader(batch_size=16)

    batch = next(iter(dataloader))

    print("\nDataLoader batch shapes:")
    print(
        "Order-book features:",
        batch["orderbook_features"].shape,
    )
    print(
        "Sentiment score:",
        batch["sentiment_score"].shape,
    )
    print(
        "FinBERT embedding:",
        batch["embedding"].shape,
    )