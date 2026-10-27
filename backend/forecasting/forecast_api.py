
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
    """Return the saved checkpoint path."""
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
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    return checkpoint_path


def get_latest_live_orderbook():
    """
    Read the newest valid order-book record.

    This is reported separately from the TFT input window.
    """
    if not ORDERBOOK_FILE.exists():
        return None

    try:
        with ORDERBOOK_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            # Process line-by-line to avoid loading the entire file.
            latest = None

            for line in file:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                price = record.get(
                    "mid_price",
                    record.get("midPrice"),
                )

                if price is None:
                    continue

                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(price) or price <= 0:
                    continue

                timestamp = record.get("timestamp")

                if timestamp is None:
                    continue

                latest = {
                    "price": price,
                    "timestamp": str(timestamp),
                    "symbol": str(record.get("symbol", "MOCK")),
                }

            return latest

    except OSError:
        return None

    return None


def load_model_and_data():
    """
    Build the historical dataset and load the trained TFT checkpoint.

    The preprocessing and dataset configuration must remain compatible
    with the configuration used during training.
    """
    checkpoint_path = get_checkpoint_path()

    orderbook = prepare_orderbook()
    news = prepare_news()
    df = build_dataset(orderbook, news)

    if df.empty:
        raise ValueError(
            "The combined order-book/news dataset is empty."
        )

    required_columns = {
        "timestamp",
        "time_idx",
        "symbol",
        "session_id",
        "mid_price",
        "sentiment_score",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )
    df["mid_price"] = pd.to_numeric(
        df["mid_price"],
        errors="coerce",
    )
    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce",
    )
    df = df.dropna(
        subset=["timestamp", "mid_price", "time_idx", "session_id"]
    )

    if df.empty:
        raise ValueError(
            "No valid historical rows remain after data cleaning."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("\n--- QUANTFORMER FORECAST INPUT CHECK ---")
    print("Dataset rows:", len(df))
    print("Latest timestamp:", df["timestamp"].max())
    print("Latest historical mid-price:", df.iloc[-1]["mid_price"])

    print("\n--- SESSION DIAGNOSTICS ---")

    session_summary = (
        df.groupby("session_id")
        .agg(
            rows=("mid_price", "size"),
            first_price=("mid_price", "first"),
            last_price=("mid_price", "last"),
            latest_time=("timestamp", "max"),
        )
        .sort_values("latest_time")
    )

    print(session_summary.to_string())

    print("\nLongest session ID:", df.groupby("session_id").size().idxmax())
    print("Latest session ID:", df.sort_values("timestamp").iloc[-1]["session_id"])



    print(
        "Nonzero sentiment rows:",
        int((df["sentiment_score"].fillna(0) != 0).sum()),
    )

    training_dataset, _ = create_tft_datasets(df)

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )
    model.eval()

    return df, training_dataset, model


def create_latest_prediction_loader(df, training_dataset):
    """
    Build the most recent historical encoder window and decoder rows.

    Future mid_price values are placeholders required by the dataset
    interface. They are NOT observed future prices.

    Other future covariates are also assumptions because their real
    future values are unknown.
    """
    # Find the most recent session, not merely the longest session.
    latest_session_id = (
        df.sort_values("timestamp")
        .iloc[-1]["session_id"]
    )

    session_df = (
        df[df["session_id"] == latest_session_id]
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )

    required_rows = ENCODER_LENGTH + PREDICTION_LENGTH

    if len(session_df) < required_rows:
        raise ValueError(
            f"The latest session has {len(session_df)} rows. "
            f"At least {required_rows} rows are required. "
            "Collect more continuous order-book data."
        )

    encoder_df = session_df.tail(ENCODER_LENGTH).copy()
    last_row = encoder_df.iloc[-1].copy()
    historical_price = float(last_row["mid_price"])

    # Use the training dataset's feature configuration.
    # The copied values below are explicit placeholder assumptions,
    # not real future order-book/news observations.
    future_rows = []

    for step in range(1, PREDICTION_LENGTH + 1):
        row = last_row.copy()
        row["time_idx"] = int(last_row["time_idx"]) + step
        row["timestamp"] = (
            pd.Timestamp(last_row["timestamp"])
            + pd.Timedelta(minutes=step)
        )

        # Required placeholder for the target column.
        # It must not be interpreted as an actual future price.
        row["mid_price"] = historical_price

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
        raise ValueError(
            "Could not create a prediction window. "
            "Check the latest session and training dataset configuration."
        )

    loader = prediction_dataset.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
    )

    return loader, last_row, historical_price


@app.get("/forecast")
def forecast():
    try:
        df, training_dataset, model = load_model_and_data()

        loader, last_row, historical_price = (
            create_latest_prediction_loader(
                df,
                training_dataset,
            )
        )

        print("DEBUG 1: Starting model prediction...", flush=True)

        with torch.no_grad():
            prediction = model.predict(
                loader,
                mode="prediction",
            )

        print("DEBUG 2: Model prediction completed.", flush=True)

        values = (
            prediction[0]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if len(values) != PREDICTION_LENGTH:
            raise ValueError(
                f"Expected {PREDICTION_LENGTH} predictions, "
                f"but received {len(values)}."
            )

        live = get_latest_live_orderbook()

        return {
            "symbol": str(last_row["symbol"]),
            "last_observed_price": (
                live["price"] if live else historical_price
            ),
            "live_price_available": live is not None,
            "live_price_timestamp": (
                live["timestamp"] if live else None
            ),
            "historical_input_timestamp": str(
                pd.Timestamp(last_row["timestamp"])
            ),
            "forecast_steps": [
                {
                    "minutes_ahead": step,
                    "predicted_mid_price": float(price),
                }
                for step, price in enumerate(values, start=1)
            ],
            "model": "Temporal Fusion Transformer",
            "experimental": True,
            "future_features_are_assumed": True,
            "note": (
                "Experimental forecast from a historical TFT input window. "
                "The current live order-book price is reported separately. "
                "Future decoder features are placeholders, not observed "
                "future data. The forecast is not yet validated as a "
                "live-input prediction or trading signal."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/interpretability")
def interpretability():
    try:
        df, training_dataset, model = load_model_and_data()

        loader, last_row, _ = create_latest_prediction_loader(
            df,
            training_dataset,
        )

        with torch.no_grad():
            interpretation = model.interpret_output(
                model.predict(
                    loader,
                    mode="raw",
                    return_x=True,
                )[0],
                reduction="sum",
            )

        def importance_list(values):
            if values is None:
                return []

            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()

            values = np.asarray(values).reshape(-1)

            names = list(
                getattr(model, "encoder_variables", [])
            )

            result = []

            for index, value in enumerate(values):
                name = (
                    str(names[index])
                    if index < len(names)
                    else f"feature_{index + 1}"
                )

                result.append({
                    "feature": name,
                    "importance": float(value),
                })

            return sorted(
                result,
                key=lambda item: item["importance"],
                reverse=True,
            )

        encoder_importance = importance_list(
            interpretation.get("encoder_variables")
        )

        decoder_importance = importance_list(
            interpretation.get("decoder_variables")
        )

        attention = interpretation.get("attention")

        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()

        if attention is None:
            attention_values = []
        else:
            attention_values = (
                np.asarray(attention)
                .reshape(-1)
                .astype(float)
                .tolist()
            )

        return {
            "model": "Temporal Fusion Transformer",
            "symbol": str(last_row["symbol"]),
            "historical_input_timestamp": str(
                pd.Timestamp(last_row["timestamp"])
            ),
            "encoder_feature_importance": encoder_importance,
            "decoder_feature_importance": decoder_importance,
            "attention_weights": attention_values,
            "note": (
                "Experimental interpretation of a historical input window. "
                "Feature importance and attention do not establish causation. "
                "Future decoder features are assumed placeholders. "
                "Not trading advice."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )