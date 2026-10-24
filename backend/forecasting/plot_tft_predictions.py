
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "tft_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "tft_forecast_plot.png"
)


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "forecast_step",
        "predicted_mid_price",
    ]

    if not all(column in df.columns for column in required_columns):
        raise ValueError(
            f"CSV must contain these columns: {required_columns}"
        )

    df = df.sort_values("forecast_step")

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["forecast_step"],
        df["predicted_mid_price"],
        marker="o",
        linewidth=2,
        label="TFT predicted mid-price",
    )

    plt.title("QuantFormer - TFT Five-Step Forecast")
    plt.xlabel("Forecast step")
    plt.ylabel("Predicted mid-price")
    plt.xticks(df["forecast_step"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"Forecast graph saved to: {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()
