
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE


# ---------------------------------------------------------
# Project paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "backend" / "data" / "tft_minute_data.csv"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

ENCODER_LENGTH = 20
PREDICTION_LENGTH = 5
BATCH_SIZE = 8
MAX_EPOCHS = 10
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

def load_and_prepare_data():
    """Load data while preserving recording-session boundaries."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"TFT dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required_columns = [
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
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    numeric_columns = [
        "mid_price",
        "spread",
        "best_bid",
        "best_ask",
        "total_bid_volume",
        "total_ask_volume",
        "orderbook_imbalance",
        "sentiment_score",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    # Reconstruct sessions if session_id is unavailable.
    if "session_id" not in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        gaps = df["timestamp"].diff()
        starts_session = gaps.isna() | gaps.ne(pd.Timedelta(minutes=1))
        df["session_id"] = starts_session.cumsum()

    df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce")
    df = df.dropna(subset=["session_id"]).copy()
    df["session_id"] = df["session_id"].astype(int)

    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    # Remove invalid numeric values and non-positive prices.
    finite_mask = np.isfinite(df[numeric_columns]).all(axis=1)
    df = df[(df["mid_price"] > 0) & finite_mask].copy()

    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    # Rebuild the time index separately inside every session.
    df["time_idx"] = df.groupby("session_id").cumcount()

    if df.empty:
        raise ValueError("No usable rows remain after validation.")

    print(f"Prepared rows: {len(df)}")
    print(f"Recording sessions: {df['session_id'].nunique()}")
    print("\nSession lengths:")
    print(df.groupby("session_id").size().to_string())

    return df


# ---------------------------------------------------------
# Create chronological training and validation datasets
# ---------------------------------------------------------



def create_tft_datasets(df):
    """Train on multiple sessions and validate on the latest session."""

    df = df.copy()

    minimum_rows = ENCODER_LENGTH + PREDICTION_LENGTH + 2

    session_lengths = df.groupby("session_id").size()
    usable_sessions = session_lengths[
        session_lengths >= minimum_rows
    ].index.tolist()

    if len(usable_sessions) < 2:
        raise ValueError(
            "At least two usable recording sessions are required."
        )

    session_order = (
        df[df["session_id"].isin(usable_sessions)]
        .groupby("session_id")["timestamp"]
        .max()
        .sort_values()
        .index.tolist()
    )

    validation_session = session_order[-1]
    training_sessions = session_order[:-1]

    df["series_id"] = (
        df["symbol"].astype(str)
        + "_session_"
        + df["session_id"].astype(str)
    )

    training_df = df[
        df["session_id"].isin(training_sessions)
    ].copy()

    validation_df = df[
        df["session_id"] == validation_session
    ].copy()

    training_df["time_idx"] = (
        training_df.groupby("series_id").cumcount()
    )
    validation_df["time_idx"] = (
        validation_df.groupby("series_id").cumcount()
    )

    dataset_kwargs = {
        "time_idx": "time_idx",
        "target": "mid_price",
        "group_ids": ["series_id"],
        "static_categoricals": ["symbol", "series_id"],
        "time_varying_known_reals": ["time_idx"],
        "time_varying_unknown_reals": [
            "mid_price",
            "spread",
            "best_bid",
            "best_ask",
            "total_bid_volume",
            "total_ask_volume",
            "orderbook_imbalance",
            "sentiment_score",
        ],
        "max_encoder_length": ENCODER_LENGTH,
        "min_encoder_length": ENCODER_LENGTH,
        "max_prediction_length": PREDICTION_LENGTH,
        "min_prediction_length": PREDICTION_LENGTH,
        "target_normalizer": GroupNormalizer(groups=[]),
        "categorical_encoders": {
            "series_id": NaNLabelEncoder(add_nan=True),
            "symbol": NaNLabelEncoder(add_nan=True),
        },
        "add_relative_time_idx": True,
        "add_target_scales": True,
        "add_encoder_length": True,
        "allow_missing_timesteps": False,
    }

    training_dataset = TimeSeriesDataSet(
        training_df,
        **dataset_kwargs,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_df,
        predict=False,
        stop_randomization=True,
    )

    print("\nMulti-session dataset configuration")
    print("Training sessions:", training_sessions)
    print("Validation session:", validation_session)
    print("Training rows:", len(training_df))
    print("Validation rows:", len(validation_df))
    print("Training samples:", len(training_dataset))
    print("Validation samples:", len(validation_dataset))

    if len(training_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError(
            "Training or validation dataset is empty. "
            "Check session lengths and dataset configuration."
        )

    return training_dataset, validation_dataset




# ---------------------------------------------------------
# Configure the TFT model
# ---------------------------------------------------------

def build_tft_model(training_dataset):
    """Build the Temporal Fusion Transformer."""

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=8,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        loss=MAE(),
        optimizer="adam",
    )

    print("\nTFT model configured successfully.")
    print(f"Model parameters: {model.size() / 1000:.1f}K")

    return model


# ---------------------------------------------------------
# Train and save the best model
# ---------------------------------------------------------

def train_model(model, training_dataset, validation_dataset):
    """Train TFT and save the checkpoint with the best validation loss."""

    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    validation_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(MODEL_DIR / "multisession_experiment"),
        filename="quantformer-tft-multi-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
    )

    print("\nStarting TFT training...")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    best_checkpoint = checkpoint_callback.best_model_path

    if not best_checkpoint:
        raise RuntimeError("Training finished without a best checkpoint.")

    best_score = checkpoint_callback.best_model_score

    print(f"\nBest checkpoint: {best_checkpoint}")
    experiment_pointer = (
        MODEL_DIR
        / "multisession_experiment"
        / "best_checkpoint_path.txt"
    )

    experiment_pointer.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_pointer.write_text(
        best_checkpoint,
        encoding="utf-8",
    )

    print(f"Experiment checkpoint pointer saved to: {experiment_pointer}")
    print(f"Best validation loss: {best_score}")

    best_model = TemporalFusionTransformer.load_from_checkpoint(
        best_checkpoint,
        map_location="cpu",
    )

    weights_path = (
        MODEL_DIR
        / "multisession_experiment"
        / "quantformer_tft_state.pt"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "hyper_parameters": dict(best_model.hparams),
            "encoder_length": ENCODER_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "target": "mid_price",
            "selected_checkpoint": best_checkpoint,
        },
        weights_path,
    )

    print(f"Best model weights saved to: {weights_path}")

    return trainer, best_checkpoint


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

if __name__ == "__main__":
    pl.seed_everything(RANDOM_SEED, workers=True)

    print("Loading TFT forecasting data...")
    df = load_and_prepare_data()

    print("\nCreating training and validation datasets...")
    training_dataset, validation_dataset = create_tft_datasets(df)

    print("\nConfiguring TFT model...")
    model = build_tft_model(training_dataset)

    print("\nForecast configuration:")
    print("Target: mid_price")
    print(f"Encoder length: {ENCODER_LENGTH} time steps")
    print(f"Prediction length: {PREDICTION_LENGTH} time steps")
    print("The fifth decoder output is the five-minute-ahead forecast")
    print("provided observations are sampled one minute apart.")

    print(
        "\nNote: this model is experimental and is not a validated "
        "financial forecasting system."
    )

    trainer, best_checkpoint = train_model(
        model,
        training_dataset,
        validation_dataset,
    )

    print("\nTFT training completed successfully.")
    print(f"Best checkpoint: {best_checkpoint}")

