"""
agent/confidence_net.py
Loads the trained MLP and runs inference on draft features.
Falls back to a neutral score (0.5) if no model has been trained yet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from data.models import ConfidenceFeatures

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent / "training" / "confidence_net.pt"
_N_FEATURES  = 10


class ConfidenceNetModel(nn.Module):
    """Tiny MLP: 10 → 32 → 16 → 1 (sigmoid)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_N_FEATURES, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConfidenceNet:
    """
    Inference wrapper for ConfidenceNetModel.
    Loads from disk if available; otherwise uses untrained fallback (score=0.5).
    """

    def __init__(self) -> None:
        self._model: ConfidenceNetModel | None = None
        self._trained = False
        self._load()

    def _load(self) -> None:
        if _MODEL_PATH.exists():
            try:
                self._model = ConfidenceNetModel()
                self._model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu"))
                self._model.eval()
                self._trained = True
                logger.info("Confidence Net loaded from %s", _MODEL_PATH)
            except Exception as exc:
                logger.warning("Failed to load Confidence Net: %s — using fallback", exc)
                self._model = None
                self._trained = False
        else:
            logger.info("No trained Confidence Net found — using fallback score 0.5")

    def predict(self, features: ConfidenceFeatures) -> float:
        """Return confidence score in [0.0, 1.0]."""
        if not self._trained or self._model is None:
            return 0.5  # Neutral fallback until trained

        feature_list = features.to_list()
        tensor = torch.tensor([feature_list], dtype=torch.float32)
        with torch.no_grad():
            score = self._model(tensor).item()
        return float(score)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def reload(self) -> None:
        """Reload model from disk (call after retraining)."""
        self._load()
