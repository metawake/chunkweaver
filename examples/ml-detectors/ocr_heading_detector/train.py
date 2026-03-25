"""Train an OCR heading detector model.

Uses GradientBoostingClassifier with leave-one-doc-out cross-validation.
Prints per-fold and aggregate metrics, plus feature importances.
Saves trained model to model.joblib.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

from sample_docs import LABELED_DOCS
from features import extract_features, FEATURE_NAMES


def doc_to_features_labels(doc: list[tuple[str, bool]]):
    """Convert a labeled doc to feature matrix + label vector."""
    lines = [text for text, _ in doc]
    X, y = [], []
    for i, (text, is_heading) in enumerate(doc):
        if not text.strip():
            continue
        feats = extract_features(lines, i)
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(1 if is_heading else 0)
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("OCR Heading Detector — Training")
    print("=" * 60)

    all_X, all_y, doc_indices = [], [], []
    for doc_idx, doc in enumerate(LABELED_DOCS):
        X, y = doc_to_features_labels(doc)
        all_X.append(X)
        all_y.append(y)
        doc_indices.extend([doc_idx] * len(X))

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    doc_indices = np.array(doc_indices)

    print(f"\nTotal samples: {len(all_y)}")
    print(f"  Headings: {all_y.sum()}")
    print(f"  Body:     {(1 - all_y).sum()}")
    print(f"  Docs:     {len(LABELED_DOCS)}")

    # Leave-one-doc-out cross-validation
    print("\n--- Leave-One-Doc-Out Cross-Validation ---\n")
    fold_f1s = []
    fold_reports = []

    for held_out in range(len(LABELED_DOCS)):
        train_mask = doc_indices != held_out
        test_mask = doc_indices == held_out

        X_train, y_train = all_X[train_mask], all_y[train_mask]
        X_test, y_test = all_X[test_mask], all_y[test_mask]

        if len(np.unique(y_test)) < 2:
            continue

        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        fold_f1s.append(f1)

        doc_type = "clean" if held_out == 5 else "OCR-damaged"
        print(f"Fold {held_out + 1} ({doc_type}):  F1={f1:.3f}  "
              f"(test={len(y_test)} samples, {y_test.sum()} headings)")
        fold_reports.append((held_out, classification_report(y_test, y_pred, zero_division=0)))

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"\nAggregate F1: {mean_f1:.3f} ± {std_f1:.3f}")

    # Train final model on all data
    print("\n--- Training Final Model ---\n")
    final_clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    final_clf.fit(all_X, all_y)

    print("Feature importances:")
    importances = list(zip(FEATURE_NAMES, final_clf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:25s} {imp:.3f} {bar}")

    # Save
    joblib.dump(final_clf, "model.joblib")
    print("\nModel saved to model.joblib")

    # Print detailed report for worst fold
    if fold_reports:
        worst_idx = int(np.argmin(fold_f1s))
        held_out, report = fold_reports[worst_idx]
        print(f"\n--- Worst fold detail (Doc {held_out + 1}) ---")
        print(report)


if __name__ == "__main__":
    main()
