"""ML-based clinical section boundary detector.

Drop-in replacement for HeadingDetector on clinical notes.
Uses a trained gradient boosting model to identify section transitions
in dictated/transcribed clinical text where explicit headers are missing.

Usage::

    from chunkweaver import Chunker
    from detector import MLClinicalSectionDetector

    chunker = Chunker(
        target_size=1024,
        detectors=[MLClinicalSectionDetector()],
    )
    chunks = chunker.chunk(clinical_note_text)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np

from chunkweaver.detectors import Annotation, BoundaryDetector, SplitPoint

from features import extract_line_features


class MLClinicalSectionDetector(BoundaryDetector):
    """Detect clinical note section boundaries using a trained ML model.

    Args:
        model_path: Path to the trained joblib model. Defaults to
            model.joblib in the same directory as this file.
        threshold: Prediction probability threshold. Lower values
            detect more boundaries (higher recall, lower precision).
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        if model_path is None:
            model_path = str(Path(__file__).parent / "model.joblib")
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def detect(self, text: str) -> List[Annotation]:
        lines = text.split("\n")
        results: List[Annotation] = []

        feature_rows: list[tuple[int, list[float]]] = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            feats = extract_line_features(lines, i)
            feature_rows.append((i, feats))

        if not feature_rows:
            return results

        indices, feat_matrix = zip(*feature_rows)
        X = np.array(feat_matrix)
        probas = self.model.predict_proba(X)[:, 1]

        offset = 0
        line_offsets = []
        for line in lines:
            line_offsets.append(offset)
            offset += len(line) + 1

        for idx, (line_idx, prob) in enumerate(zip(indices, probas)):
            if prob >= self.threshold:
                line_text = lines[line_idx].strip()
                results.append(SplitPoint(
                    position=line_offsets[line_idx],
                    line_number=line_idx,
                    label=f"clinical-section ({prob:.2f}): {line_text[:50]}",
                ))

        return results
