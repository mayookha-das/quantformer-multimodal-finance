
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from backend.forecasting.train_tft import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    create_tft_datasets,
)
from backend.preprocessing.prepare_tft_data import (
    prepare_orderbook,
    prepare_news,
    build_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
DATA_DIR = PROJECT_ROOT / "backend" / "data"

CHECKPOINT_FILE = MODEL_DIR / "best_checkpoint_path.txt"
ORDERBOOK_FILE = DATA_DIR / "processed_orderbook.jsonl"

app = FastAPI(title="QuantFormer Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "QuantFormer Forecast API",
    }


def get_checkpoint_path():
    """Resolve and validate the saved TFT checkpoint."""
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            "Checkpoint path file is missing. Train the TFT model first."
        )

    checkpoint_path = Path(
        CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path


def get_latest_live_price():
    """Read the latest valid mid-price from processed order-book data."""
    if not ORDERBOOK_FILE.exists():
        return None

    try:
        with ORDERBOOK_FILE.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(0, size - 200_000))
            chunk = file.read().decode("utf-8", errors="ignore")

        for line in reversed(chunk.splitlines()):
            try:
                row = json.loads(line)
                price = float(
                    row.get("mid_price", row.get("midPrice"))
                )
                if np.isfinite(price) and price > 0:
                    return price
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

    except OSError:
        return None

    return None


def load_model_and_data():
    """Load the saved model and build the current historical dataset."""
    checkpoint_path = get_checkpoint_path()

    orderbook = prepare_orderbook()
    news = prepare_news()
    df = build_dataset(orderbook, news)

    print("\n--- FORECAST INPUT CHECK ---")
    print("Dataset rows:", len(df))
    print("Latest timestamp:", df["timestamp"].max())
    print("Latest historical mid-price:", df["mid_price"].iloc[-1])
    print("Latest sentiment score:", df["sentiment_score"].iloc[-1])
    print("Sentiment nonzero rows:", int((df["sentiment_score"].fillna(0) != 0).sum()))
    print("Session lengths:")
    print(df.groupby("session_id").size().to_string())

    if df.empty:
        raise ValueError("The combined order-book/news dataset is empty.")

    training_dataset, _ = create_tft_datasets(df)

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )
    model.eval()

    return df, training_dataset, model


def create_latest_prediction_loader(df, training_dataset):
    """Build the latest encoder window and five future decoder rows."""
    session_id = df.groupby("session_id").size().idxmax()

    session_df = (
        df[df["session_id"] == session_id]
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )

    required_rows = ENCODER_LENGTH + PREDICTION_LENGTH
    if len(session_df) < required_rows:
        raise ValueError(
            f"Need at least {required_rows} rows in the selected session."
        )

    encoder_df = session_df.tail(ENCODER_LENGTH).copy()
    last_row = encoder_df.iloc[-1].copy()
    last_historical_price = float(last_row["mid_price"])

    future_rows = []

    for step in range(1, PREDICTION_LENGTH + 1):
        row = last_row.copy()
        row["time_idx"] = int(last_row["time_idx"]) + step
        row["timestamp"] = (
            pd.Timestamp(last_row["timestamp"])
            + pd.Timedelta(minutes=step)
        )

        # Placeholder target for the future decoder rows.
        # These are not observed future prices.
        row["mid_price"] = last_historical_price
        future_rows.append(row)

    window_df = pd.concat(
        [encoder_df, pd.DataFrame(future_rows)],
        ignore_index=True,
    )

    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        window_df,
        predict=True,
        stop_randomization=True,
    )

    if len(prediction_dataset) == 0:
        raise ValueError("Could not create a prediction window.")

    loader = prediction_dataset.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
    )

    return loader, last_row, last_historical_price


@app.get("/forecast")
def forecast():
    try:
        df, training_dataset, model = load_model_and_data()

        loader, last_row, historical_price = create_latest_prediction_loader(
            df,
            training_dataset,
        )

        with torch.no_grad():
            prediction = model.predict(
                loader,
                mode="prediction",
            )

        values = prediction[0].detach().cpu().numpy().reshape(-1)
        live_price = get_latest_live_price()

        return {
            "symbol": str(last_row["symbol"]),
            "last_observed_price": (
                live_price if live_price is not None else historical_price
            ),
            "live_price_available": live_price is not None,
            "forecast_steps": [
                {
                    "minutes_ahead": step,
                    "predicted_mid_price": float(price),
                }
                for step, price in enumerate(values, start=1)
            ],
            "model": "Temporal Fusion Transformer",
            "note": (
                "Experimental forecast. Current live price is reported "
                "separately. The forecast uses the combined historical "
                "dataset; it is not yet a verified live-input forecast. "
                "Not trading advice."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


def importance_to_list(values, feature_names):
    """Convert TFT importance tensors/arrays into named JSON values."""
    if values is None:
        return []

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    array = np.asarray(values).reshape(-1)
    results = []

    for index, value in enumerate(array):
        name = (
            str(feature_names[index])
            if index < len(feature_names)
            else f"feature_{index}"
        )
        results.append(
            {
                "feature": name,
                "importance": float(value),
            }
        )

    results.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )
    return results


@app.get("/interpretability")
def interpretability():
    try:
        df, training_dataset, model = load_model_and_data()

        loader, last_row, _ = create_latest_prediction_loader(
            df,
            training_dataset,
        )

        with torch.no_grad():
            raw_prediction = model.predict(
                loader,
                mode="raw",
                return_x=True,
            )

        interpretation = model.interpret_output(
            raw_prediction.output,
            reduction="mean",
        )

        # Use the variable names supplied by the model, rather than
        # assuming every variable is a real-valued feature.
        encoder_names = list(
            getattr(model, "encoder_variables", [])
        )
        decoder_names = list(
            getattr(model, "decoder_variables", [])
        )

        encoder_importance = importance_to_list(
            interpretation.get("encoder_variables"),
            encoder_names,
        )
        decoder_importance = importance_to_list(
            interpretation.get("decoder_variables"),
            decoder_names,
        )

        attention = interpretation.get("attention")
        if attention is not None:
            if hasattr(attention, "detach"):
                attention = attention.detach().cpu().numpy()
            attention_values = np.asarray(attention).reshape(-1).tolist()
        else:
            attention_values = []

        return {
            "model": "Temporal Fusion Transformer",
            "symbol": str(last_row["symbol"]),
            "encoder_feature_importance": encoder_importance,
            "decoder_feature_importance": decoder_importance,
            "attention_weights": [
                float(value) for value in attention_values
            ],
            "note": (
                "Experimental interpretation of a historical input window. "
                "Feature importance and attention do not establish causation. "
                "Live news and order-book fusion must be independently "
                "verified. Not trading advice."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
