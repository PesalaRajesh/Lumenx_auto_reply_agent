"""
training/train_mlp.py
Trains the Confidence Net MLP on labelled draft data from the database.

Usage:
    python -m training.train_mlp [--epochs 100] [--lr 0.001]
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import typer

from agent.confidence_net import ConfidenceNetModel
from data.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()
_MODEL_PATH = Path(__file__).parent / "confidence_net.pt"
_DEFAULT_MIN_EXAMPLES = 30


async def load_training_data(db: Database, min_examples: int) -> tuple[np.ndarray, np.ndarray]:
    """Load labelled drafts and return X, y arrays."""
    rows = await db.get_labelled_dataset()

    if len(rows) < min_examples:
        raise ValueError(
            f"Only {len(rows)} labelled examples found. Need at least {min_examples}."
        )

    X, y = [], []
    for row in rows:
        features = json.loads(row["features_json"])
        feature_vec = [
            features.get("intent_id", 7),
            features.get("reply_length", 0.5),
            features.get("contains_pricing", 0.0),
            features.get("contains_refund", 0.0),
            features.get("product_mentioned", 0.0),
            features.get("thread_depth", 0.0),
            features.get("wiki_hit_count", 0.0),
            features.get("feedback_log_match_score", 0.0),
            features.get("draft_length_ratio", 0.5),
            features.get("customer_sentiment", 0.0),
        ]
        X.append(feature_vec)
        y.append(float(row["label"]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


@app.command()
def train(
    db_path: str = "data/agent.db",
    epochs: int = 150,
    lr: float = 0.001,
    batch_size: int = 8,
    min_examples: int = _DEFAULT_MIN_EXAMPLES,
) -> None:
    """Train the Confidence Net and save to disk."""

    async def _run() -> None:
        db = Database(db_path)
        X, y = await load_training_data(db, min_examples)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Training on %d examples, validating on %d", len(X_train), len(X_val))

        model = ConfidenceNetModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_train_t = torch.tensor(X_train)
        y_train_t = torch.tensor(y_train).unsqueeze(1)
        X_val_t   = torch.tensor(X_val)
        y_val_t   = torch.tensor(y_val).unsqueeze(1)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            # Mini-batch training
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            batches = 0
            for i in range(0, len(perm), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_train_t[idx], y_train_t[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t).numpy()
                val_mae = mean_absolute_error(y_val, val_pred)
                val_loss = criterion(model(X_val_t), y_val_t).item()
                logger.info(
                    "Epoch %3d — train_loss=%.4f  val_loss=%.4f  val_mae=%.4f",
                    epoch + 1, epoch_loss / batches, val_loss, val_mae,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), _MODEL_PATH)
                    logger.info("  ✓ Best model saved (val_loss=%.4f)", best_val_loss)

        logger.info("Training complete. Model saved to %s", _MODEL_PATH)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
