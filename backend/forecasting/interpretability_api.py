
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pytorch_forecasting import TemporalFusionTransformer

from backend.forecasting.train_tft import (
    load_and_prepare_data,
    create_tft_datasets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
CHECKPOINT_FILE = MODEL_DIR / "best_checkpoint_path.txt"

app = FastAPI(title="QuantFormer Interpretability API")

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
    return {"status": "ok"}


@app.get("/interpretability")
def interpretability():
    try:
        if not CHECKPOINT_FILE.exists():
            raise FileNotFoundError(
                "Train the TFT model first; checkpoint path is missing."
            )

        checkpoint = Path(
            CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
        )

        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        df = load_and_prepare_data()
        training_dataset, _ = create_tft_datasets(df)

        session_id = df.groupby("session_id").size().idxmax()
        session = (
            df[df["session_id"] == session_id]
            .sort_values("time_idx")
            .reset_index(drop=True)
        )

        encoder_length = training_dataset.max_encoder_length
        prediction_length = training_dataset.max_prediction_length
        needed = encoder_length + prediction_length

        if len(session) < needed:
            raise ValueError(
                f"Need at least {needed} rows in the selected session."
            )

        window = session.tail(needed).copy()

        dataset = type(training_dataset).from_dataset(
            training_dataset,
            window,
            predict=True,
            stop_randomization=True,
        )

        if len(dataset) == 0:
            raise ValueError("Could not create an interpretation window.")

        loader = dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
        )

        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint),
            map_location="cpu",
        )
        model.eval()

        with torch.no_grad():
            batch = next(iter(loader))
            x, _ = batch
            interpretation = model.interpret_output(
                model(x),
                reduction="sum",
            )

        attention = interpretation.get("attention")
        variables = interpretation.get("encoder_variables")
        decoder_variables = interpretation.get("decoder_variables")

        def to_list(values):
            if values is None:
                return []
            return (
                values.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(float)
                .tolist()
            )

        encoder_importance = to_list(variables)
        decoder_importance = to_list(decoder_variables)

        feature_names = [
            "mid_price",
            "spread",
            "best_bid",
            "best_ask",
            "total_bid_volume",
            "total_ask_volume",
            "orderbook_imbalance",
            "sentiment_score",
        ]

        def rank_features(values):
            names = feature_names[:len(values)]
            ranked = sorted(
                zip(names, values),
                key=lambda item: item[1],
                reverse=True,
            )
            return [
                {"feature": name, "importance": value}
                for name, value in ranked
            ]

        attention_values = to_list(attention)

        return {
            "model": "Temporal Fusion Transformer",
            "symbol": str(session.iloc[-1]["symbol"]),
            "encoder_feature_importance": rank_features(
                encoder_importance
            ),
            "decoder_feature_importance": rank_features(
                decoder_importance
            ),
            "attention_weights": attention_values,
            "note": (
                "Experimental model interpretation for a historical "
                "window. Importance scores are not causal explanations "
                "or trading advice."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
