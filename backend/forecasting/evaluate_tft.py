
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from backend.forecasting.train_tft import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    load_and_prepare_data,
    create_tft_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

CHECKPOINT_FILE = (
    MODEL_DIR
    / "multisession_experiment"
    / "best_checkpoint_path.txt"
)


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return mae, rmse


def main():
    print("Loading data...")
    df = load_and_prepare_data()

    print("\nPreparing multi-session datasets...")
    _, validation_dataset = create_tft_datasets(df)

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint pointer not found: {CHECKPOINT_FILE}"
        )

    checkpoint_text = CHECKPOINT_FILE.read_text(
        encoding="utf-8"
    ).strip()

    checkpoint_path = Path(checkpoint_text)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    print("\nLoading experimental TFT checkpoint...")
    print(checkpoint_path)

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )
    model.eval()

    loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=8,
        num_workers=0,
    )

    print("\nGenerating held-out-session predictions...")

    with torch.no_grad():
        result = model.predict(
            loader,
            mode="prediction",
            return_y=True,
        )

    predictions = result.output.detach().cpu().numpy()
    actual = result.y[0].detach().cpu().numpy()

    predictions = np.asarray(predictions).reshape(
        -1, PREDICTION_LENGTH
    )
    actual = np.asarray(actual).reshape(
        -1, PREDICTION_LENGTH
    )

    if len(predictions) != len(validation_dataset):
        raise RuntimeError(
            "Prediction count does not match validation dataset: "
            f"{len(predictions)} predictions vs "
            f"{len(validation_dataset)} validation samples."
        )

    # The validation dataset is built from the held-out session.
    # Use its raw prices for the persistence baseline.
    validation_session_id = (
        df.sort_values("timestamp").iloc[-1]["session_id"]
    )

    validation_df = (
        df[df["session_id"] == validation_session_id]
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )

    rows = []

    for forecast_idx in range(len(predictions)):
        sample = validation_dataset[forecast_idx]
        sample_inputs = sample[0]

        encoder_start = int(
            sample_inputs["encoder_time_idx_start"].item()
        )
        encoder_length = int(
            sample_inputs["encoder_length"].item()
        )

        # The validation session's time_idx starts at zero.
        # Find the last observed raw-price row for this window.
        last_observed_idx = (
            encoder_start + encoder_length - 1
        )

        for step in range(PREDICTION_LENGTH):
            actual_idx = last_observed_idx + step + 1

            if actual_idx >= len(validation_df):
                raise IndexError(
                    "Forecast window exceeds the validation "
                    "session's raw price data. Check dataset "
                    "index alignment."
                )

            last_observed_price = float(
                validation_df.iloc[last_observed_idx]["mid_price"]
            )
            actual_price = float(
                validation_df.iloc[actual_idx]["mid_price"]
            )

            rows.append(
                {
                    "forecast_id": forecast_idx,
                    "forecast_step_minutes": step + 1,
                    "actual_mid_price": actual_price,
                    "tft_predicted_mid_price": float(
                        predictions[forecast_idx, step]
                    ),
                    "baseline_predicted_mid_price": (
                        last_observed_price
                    ),
                }
            )

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        raise RuntimeError(
            "No validation predictions were generated."
        )

    five_minute_df = results_df[
        results_df["forecast_step_minutes"] == PREDICTION_LENGTH
    ].copy()

    overall_tft_mae, overall_tft_rmse = calculate_metrics(
        results_df["actual_mid_price"],
        results_df["tft_predicted_mid_price"],
    )

    overall_base_mae, overall_base_rmse = calculate_metrics(
        results_df["actual_mid_price"],
        results_df["baseline_predicted_mid_price"],
    )

    five_tft_mae, five_tft_rmse = calculate_metrics(
        five_minute_df["actual_mid_price"],
        five_minute_df["tft_predicted_mid_price"],
    )

    five_base_mae, five_base_rmse = calculate_metrics(
        five_minute_df["actual_mid_price"],
        five_minute_df["baseline_predicted_mid_price"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        MODEL_DIR / "tft_evaluation_predictions.csv"
    )
    five_minute_path = (
        MODEL_DIR / "tft_five_minute_evaluation.csv"
    )

    results_df.to_csv(predictions_path, index=False)
    five_minute_df.to_csv(five_minute_path, index=False)

    print("\n========== HELD-OUT SESSION EVALUATION ==========")
    print(
        f"Forecast windows: "
        f"{results_df['forecast_id'].nunique()}"
    )
    print(f"Evaluated steps: {len(results_df)}")
    print(f"Overall TFT MAE: {overall_tft_mae:.6f}")
    print(f"Overall TFT RMSE: {overall_tft_rmse:.6f}")
    print(f"Overall baseline MAE: {overall_base_mae:.6f}")
    print(f"Overall baseline RMSE: {overall_base_rmse:.6f}")

    print("\n========== FIVE-MINUTE-AHEAD EVALUATION ==========")
    print(f"Forecast windows: {len(five_minute_df)}")
    print(f"TFT MAE: {five_tft_mae:.6f}")
    print(f"TFT RMSE: {five_tft_rmse:.6f}")
    print(f"Baseline MAE: {five_base_mae:.6f}")
    print(f"Baseline RMSE: {five_base_rmse:.6f}")

    print("\nSaved evaluation files:")
    print(predictions_path)
    print(five_minute_path)

    print(
        "\nCaution: validation windows overlap, so overall metrics "
        "are not independent forecast observations. These results "
        "are preliminary and do not establish reliable financial "
        "forecasting performance."
    )


if __name__ == "__main__":
    main()
