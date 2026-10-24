
from pathlib import Path

import pandas as pd
import torch
import lightning.pytorch as pl

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
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
    """Load data while preserving recording sessions."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"TFT dataset not found: {DATA_FILE}"
        )

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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
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
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid rows remain after cleaning the dataset."
        )

    # Reconstruct recording sessions if session_id is absent.
    if "session_id" not in df.columns:
        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        time_gap = df["timestamp"].diff()

        starts_session = (
            time_gap.isna()
            | time_gap.ne(pd.Timedelta(minutes=1))
        )

        df["session_id"] = starts_session.cumsum()

    df["session_id"] = pd.to_numeric(
        df["session_id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["session_id"]
    ).copy()

    df["session_id"] = df["session_id"].astype(int)

    df = df.sort_values(
        ["session_id", "timestamp"]
    ).reset_index(drop=True)

    # Each recording session gets its own zero-based time index.
    df["time_idx"] = (
        df.groupby("session_id").cumcount()
    )

    # Reject invalid prices and non-finite numeric values.
    df = df[
        (df["mid_price"] > 0)
        & df[numeric_columns].apply(
            lambda column: column.map(
                lambda value: pd.notna(value)
                and abs(value) != float("inf")
            )
        ).all(axis=1)
    ].copy()

    df = df.sort_values(
        ["session_id", "timestamp"]
    ).reset_index(drop=True)

    # Rebuild time indices after cleaning.
    df["time_idx"] = (
        df.groupby("session_id").cumcount()
    )

    if df.empty:
        raise ValueError(
            "No usable data remains after numeric validation."
        )

    print(f"Prepared rows: {len(df)}")
    print(
        f"Recording sessions: "
        f"{df['session_id'].nunique()}"
    )
    print("\nSession lengths:")
    print(
        df.groupby("session_id").size().to_string()
    )

    return df


# ---------------------------------------------------------
# Create training and validation datasets
# ---------------------------------------------------------

def create_tft_datasets(df):
    """Create chronological datasets from the longest session."""

    session_lengths = df.groupby("session_id").size()

    minimum_rows = (
        ENCODER_LENGTH + PREDICTION_LENGTH + 2
    )

    usable_sessions = session_lengths[
        session_lengths >= minimum_rows
    ]

    if usable_sessions.empty:
        raise ValueError(
            f"No session has at least {minimum_rows} rows. "
            "Collect a longer continuous recording."
        )

    # Avoid creating sequences across recording gaps.
    selected_session = usable_sessions.idxmax()

    session_df = (
        df[df["session_id"] == selected_session]
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )

    split_idx = int(
        len(session_df) * (1 - VALIDATION_FRACTION)
    )

    if split_idx < ENCODER_LENGTH + PREDICTION_LENGTH:
        raise ValueError(
            "The training segment is too short for the "
            "chosen encoder and prediction lengths."
        )

    training_df = session_df.iloc[:split_idx].copy()

    # Include previous observations for the validation encoder.
    validation_df = session_df.iloc[
        split_idx - ENCODER_LENGTH:
    ].copy()

    dataset_kwargs = {
        "time_idx": "time_idx",
        "target": "mid_price",
        "group_ids": ["symbol"],
        "max_encoder_length": ENCODER_LENGTH,
        "max_prediction_length": PREDICTION_LENGTH,
        "static_categoricals": ["symbol"],
        "time_varying_known_reals": [
            "time_idx",
        ],
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
        "target_normalizer": GroupNormalizer(
            groups=["symbol"]
        ),
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
        predict=True,
        stop_randomization=True,
    )

    print(f"\nSelected session: {selected_session}")
    print(f"Selected session rows: {len(session_df)}")
    print(f"Training rows: {len(training_df)}")
    print(
        "Validation rows including encoder context: "
        f"{len(validation_df)}"
    )
    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    if len(training_dataset) == 0:
        raise ValueError(
            "No training samples were created."
        )

    if len(validation_dataset) == 0:
        raise ValueError(
            "No validation samples were created."
        )

    return training_dataset, validation_dataset


# ---------------------------------------------------------
# Configure the TFT model
# ---------------------------------------------------------

def build_tft_model(training_dataset):
    """Build a small TFT model for the prototype."""

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
    print(
        f"Model parameters: {model.size() / 1000:.1f}K"
    )

    return model


# ---------------------------------------------------------
# Train and save the best model
# ---------------------------------------------------------

def train_model(
    model,
    training_dataset,
    validation_dataset,
):
    """Train TFT and save the best validation checkpoint."""

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

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(MODEL_DIR),
        filename="quantformer-tft-{epoch:02d}-{val_loss:.4f}",
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
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
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
        raise RuntimeError(
            "Training finished without a best checkpoint."
        )

    best_score = checkpoint_callback.best_model_score

    print(f"\nBest checkpoint: {best_checkpoint}")
    (MODEL_DIR / "best_checkpoint_path.txt").write_text(
    best_checkpoint,
    encoding="utf-8",
    )
    print(f"Best validation loss: {best_score}")

    # Load the best checkpoint before saving the state dictionary.
    best_model = TemporalFusionTransformer.load_from_checkpoint(
        best_checkpoint,
        map_location="cpu",
    )

    weights_path = (
        MODEL_DIR / "quantformer_tft_state.pt"
    )

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
    pl.seed_everything(
        RANDOM_SEED,
        workers=True,
    )

    print("Loading TFT forecasting data...")
    df = load_and_prepare_data()

    print("\nCreating training and validation datasets...")
    training_dataset, validation_dataset = (
        create_tft_datasets(df)
    )

    print("\nConfiguring TFT model...")
    model = build_tft_model(training_dataset)

    print("\nDataset and model setup completed.")
    print("Forecast target: mid_price")
    print(f"Prediction length: {PREDICTION_LENGTH} time steps")
    print(
        "Note: each step represents one minute only if "
        "the data is sampled at one-minute intervals."
    )
    print(
        "This is an experimental model, not a validated "
        "financial forecasting system."
    )

    trainer, best_checkpoint = train_model(
        model,
        training_dataset,
        validation_dataset,
    )

    print("\nTFT training completed successfully.")
    print(f"Best checkpoint: {best_checkpoint}")



