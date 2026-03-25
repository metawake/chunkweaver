"""MLOCRHeadingDetector — BoundaryDetector backed by trained model.

Loads a joblib model trained on OCR-damaged heading patterns and uses it
to emit SplitPoints at lines classified as section headings.
"""

from __future__ import annotations
import os
from pathlib import Path

import joblib
import numpy as np

from chunkweaver.detectors import Annotation, BoundaryDetector, SplitPoint
from features import extract_features, FEATURE_NAMES


class MLOCRHeadingDetector(BoundaryDetector):
    """Detects section headings in OCR-damaged text using a trained ML model."""

    def __init__(self, model_path: str | None = None, threshold: float = 0.5):
        self.threshold = threshold
        if model_path is None:
            model_path = str(Path(__file__).parent / "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train.py first."
            )
        self.model = joblib.load(model_path)

    def detect(self, text: str) -> list[Annotation]:
        lines = text.split("\n")
        annotations: list[Annotation] = []

        char_offset = 0
        for i, line in enumerate(lines):
            if line.strip():
                feats = extract_features(lines, i)
                X = np.array([[feats[f] for f in FEATURE_NAMES]])
                proba = self.model.predict_proba(X)[0, 1]

                if proba >= self.threshold:
                    annotations.append(SplitPoint(
                        position=char_offset,
                        line_number=i,
                        label=f"ocr_heading (p={proba:.2f})",
                    ))
            char_offset += len(line) + 1

        return annotations
