"""Train the clinical section boundary detector.

Trains a gradient boosting classifier on labeled clinical notes,
evaluates with leave-one-note-out cross-validation, and saves the model.

Usage:
    python train.py              # Train and save model.joblib
    python train.py --evaluate   # Run cross-validation only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

from features import FEATURE_NAMES, extract_line_features
from sample_notes import LABELED_NOTES


def prepare_dataset(
    notes: list[list[tuple[str, str]]],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Convert labeled notes to feature matrix and label vector.

    Returns (X, y, note_ids) where note_ids tracks which note each
    sample belongs to (for cross-validation).
    """
    all_features: list[list[float]] = []
    all_labels: list[int] = []
    note_ids: list[int] = []

    for note_idx, note in enumerate(notes):
        full_text = "\n\n".join(text for _, text in note)
        lines = full_text.split("\n")

        section_start_lines: set[int] = set()
        line_offset = 0
        for _, section_text in note:
            section_lines = section_text.split("\n")
            first_content = None
            for j, sl in enumerate(section_lines):
                if sl.strip():
                    first_content = line_offset + j
                    break
            if first_content is not None:
                section_start_lines.add(first_content)
            line_offset += len(section_lines) + 1  # +1 for blank separator

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            feats = extract_line_features(lines, i)
            is_start = 1 if i in section_start_lines else 0
            all_features.append(feats)
            all_labels.append(is_start)
            note_ids.append(note_idx)

    return np.array(all_features), np.array(all_labels), note_ids


def leave_one_out_cv(
    X: np.ndarray, y: np.ndarray, note_ids: list[int]
) -> dict:
    """Leave-one-note-out cross-validation."""
    unique_notes = sorted(set(note_ids))
    ids_arr = np.array(note_ids)

    all_true = []
    all_pred = []

    for held_out in unique_notes:
        train_mask = ids_arr != held_out
        test_mask = ids_arr == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        all_true_arr, all_pred_arr, pos_label=1, average="binary"
    )

    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "total_lines": len(all_true),
        "total_boundaries": int(all_true_arr.sum()),
        "predicted_boundaries": int(all_pred_arr.sum()),
        "report": classification_report(
            all_true_arr, all_pred_arr,
            target_names=["continuation", "section_start"],
        ),
    }


def train_and_save(X: np.ndarray, y: np.ndarray, path: str = "model.joblib"):
    """Train on full dataset and save."""
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=3,
        random_state=42,
    )
    clf.fit(X, y)

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nFeature importances:")
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:25s}  {imp:.3f}  {bar}")

    joblib.dump(clf, path)
    print(f"\nModel saved to {path}")
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="Run cross-validation only, don't save model")
    args = parser.parse_args()

    print(f"Preparing dataset from {len(LABELED_NOTES)} labeled notes...")
    X, y, note_ids = prepare_dataset(LABELED_NOTES)
    print(f"  {X.shape[0]} lines, {y.sum()} section boundaries, "
          f"{X.shape[1]} features")

    print("\n=== Leave-one-note-out cross-validation ===")
    results = leave_one_out_cv(X, y, note_ids)
    print(f"\nSection boundary detection (positive class):")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall:    {results['recall']}")
    print(f"  F1:        {results['f1']}")
    print(f"\n{results['report']}")

    if not args.evaluate:
        print("=== Training final model on all data ===")
        train_and_save(X, y)


if __name__ == "__main__":
    main()
