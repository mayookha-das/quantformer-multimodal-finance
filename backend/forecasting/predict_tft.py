
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)

from backend.forecasting.train_tft import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    load_and_prepare_data,
    create_tft_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
MODEL_PATH = MODEL_DIR / "quantformer_tft_state.pt"


def main():
    print("Loading latest forecasting data...")

    df = load_and_prepare_data()

    print("\nPreparing dataset configuration...")
    training_dataset, _ = create_tft_datasets(df)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}\n"
            "Train the updated model first."
        )

    print("\nLoading trained checkpoint...")

    # Load the best full checkpoint saved during training.
    checkpoint_path = MODEL_DIR / "best_checkpoint_path.txt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint path file not found. "
            "Please retrain the updated model first."
        )

    best_checkpoint = checkpoint_path.read_text(
        encoding="utf-8"
    ).strip()

    if not Path(best_checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {best_checkpoint}"
        )

    model = TemporalFusionTransformer.load_from_checkpoint(
        best_checkpoint,
        map_location="cpu",
    )
    model.eval()

    # Select the same longest recording session used for training.
    session_lengths = df.groupby("session_id").size()
    selected_session = session_lengths.idxmax()

    session_df = (
        df[df["session_id"] == selected_session]
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )

    required_rows = ENCODER_LENGTH + PREDICTION_LENGTH

    if len(session_df) < required_rows:
        raise ValueError(
            f"Need at least {required_rows} rows, "
            f"but found {len(session_df)}."
        )

    # Reserve the final five rows as the prediction window.
    # Their actual prices will be available in the CSV, so this
    # is primarily a pipeline check, not a true future forecast.
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        session_df,
        predict=True,
        stop_randomization=True,
    )

    if len(prediction_dataset) == 0:
        raise ValueError(
            "Could not create a prediction sample."
        )

    prediction_loader = prediction_dataset.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
    )

    print("\nGenerating five-step forecast...")

    with torch.no_grad():
        predictions = model.predict(
            prediction_loader,
            mode="prediction",
        )

    values = predictions[0].detach().cpu().numpy().reshape(-1)

    print("\nForecast results")
    print("-" * 45)

    for step, price in enumerate(values, start=1):
        print(
            f"Step {step} "
            f"(future time index +{step}): {price:.6f}"
        )

    output_path = MODEL_DIR / "tft_predictions.csv"

    pd.DataFrame({
        "forecast_step": np.arange(1, len(values) + 1),
        "predicted_mid_price": values,
    }).to_csv(output_path, index=False)

    print(f"\nForecast saved to: {output_path}")
    print(
        "\nImportant: this is an initial pipeline test. "
        "Validate against unseen future observations before "
        "using forecasts for financial decisions."
    )


if __name__ == "__main__":
    main()

