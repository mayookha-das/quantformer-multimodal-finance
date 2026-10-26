
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
CHECKPOINT_FILE = MODEL_DIR / "best_checkpoint_path.txt"

FORECAST_INTERVAL = PREDICTION_LENGTH


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return mae, rmse


def main():
    print("Loading forecasting data...")
    df = load_and_prepare_data()

    print("\nPreparing dataset configuration...")
    training_dataset, validation_dataset = create_tft_datasets(df)

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint path file not found: {CHECKPOINT_FILE}"
        )

    checkpoint_path = Path(
        CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}"
        )

    print(f"\nLoading model: {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )
    model.eval()

    # Evaluate only the same longest continuous session used for training.
    session_lengths = df.groupby("session_id").size()
    selected_session = session_lengths.idxmax()

    session_df = (
        df[df["session_id"] == selected_session]
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )

    split_idx = int(
        len(session_df) * (1 - 0.20)
    )

    print(f"Selected session: {selected_session}")
    print(f"Session rows: {len(session_df)}")
    print(f"Validation starts at row: {split_idx}")

    results = []

    # Each origin uses only observations before that origin.
    # Origins are spaced five minutes apart to reduce overlap.
    first_origin = max(split_idx, ENCODER_LENGTH)
    last_origin = len(session_df) - PREDICTION_LENGTH

    for origin in range(
        first_origin,
        last_origin + 1,
        FORECAST_INTERVAL,
    ):
        window_start = origin - ENCODER_LENGTH
        window_end = origin + PREDICTION_LENGTH

        window_df = session_df.iloc[
            window_start:window_end
        ].copy()

        expected_length = ENCODER_LENGTH + PREDICTION_LENGTH
        if len(window_df) != expected_length:
            continue

        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            window_df,
            predict=True,
            stop_randomization=True,
        )

        if len(prediction_dataset) == 0:
            continue

        prediction_loader = prediction_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
        )

        with torch.no_grad():
            prediction = model.predict(
                prediction_loader,
                mode="prediction",
            )

        predicted_prices = (
            prediction[0].detach().cpu().numpy().reshape(-1)
        )

        actual_prices = (
            window_df["mid_price"]
            .iloc[-PREDICTION_LENGTH:]
            .to_numpy(dtype=float)
        )

        # Baseline predicts that the price remains unchanged.
        last_observed_price = float(
            window_df["mid_price"].iloc[ENCODER_LENGTH - 1]
        )

        baseline_prices = np.full(
            PREDICTION_LENGTH,
            last_observed_price,
        )

        for step in range(PREDICTION_LENGTH):
            results.append(
                {
                    "forecast_origin_row": origin,
                    "forecast_step_minutes": step + 1,
                    "actual_mid_price": actual_prices[step],
                    "tft_predicted_mid_price": predicted_prices[step],
                    "baseline_predicted_mid_price": baseline_prices[step],
                }
            )

    if not results:
        raise RuntimeError(
            "No forecasts were generated. Check the session length "
            "and dataset configuration."
        )

    results_df = pd.DataFrame(results)

    # Overall metrics across all five decoder steps.
    overall_tft_mae, overall_tft_rmse = calculate_metrics(
        results_df["actual_mid_price"],
        results_df["tft_predicted_mid_price"],
    )

    overall_base_mae, overall_base_rmse = calculate_metrics(
        results_df["actual_mid_price"],
        results_df["baseline_predicted_mid_price"],
    )

    # The project's principal target: the fifth output step.
    five_minute_df = results_df[
        results_df["forecast_step_minutes"] == PREDICTION_LENGTH
    ].copy()

    five_tft_mae, five_tft_rmse = calculate_metrics(
        five_minute_df["actual_mid_price"],
        five_minute_df["tft_predicted_mid_price"],
    )

    five_base_mae, five_base_rmse = calculate_metrics(
        five_minute_df["actual_mid_price"],
        five_minute_df["baseline_predicted_mid_price"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = MODEL_DIR / "tft_evaluation_predictions.csv"
    results_df.to_csv(predictions_path, index=False)

    five_minute_path = MODEL_DIR / "tft_five_minute_evaluation.csv"
    five_minute_df.to_csv(five_minute_path, index=False)

    print("\n========== OVERALL FIVE-STEP EVALUATION ==========")
    print(f"Forecast rows: {len(results_df)}")
    print(f"TFT MAE:      {overall_tft_mae:.6f}")
    print(f"TFT RMSE:     {overall_tft_rmse:.6f}")
    print(f"Baseline MAE: {overall_base_mae:.6f}")
    print(f"Baseline RMSE:{overall_base_rmse:.6f}")

    print("\n========== FIVE-MINUTE-AHEAD EVALUATION ==========")
    print(f"Forecasts evaluated: {len(five_minute_df)}")
    print(f"TFT MAE:      {five_tft_mae:.6f}")
    print(f"TFT RMSE:     {five_tft_rmse:.6f}")
    print(f"Baseline MAE: {five_base_mae:.6f}")
    print(f"Baseline RMSE:{five_base_rmse:.6f}")

    if five_tft_mae < five_base_mae:
        print("TFT has lower five-minute MAE than the baseline.")
    elif five_tft_mae > five_base_mae:
        print("Baseline has lower five-minute MAE than TFT.")
    else:
        print("TFT and baseline have equal five-minute MAE.")

    print(f"\nAll predictions saved to: {predictions_path}")
    print(f"Five-minute results saved to: {five_minute_path}")

    print(
        "\nCaution: the checkpoint was selected using validation loss "
        "from this same validation period. Treat these metrics as "
        "preliminary, not as an independent final test."
    )


if __name__ == "__main__":
    main()
