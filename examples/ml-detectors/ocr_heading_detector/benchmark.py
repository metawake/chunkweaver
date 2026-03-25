"""Benchmark: Heuristic HeadingDetector vs ML OCR Heading Detector.

All ML numbers are HELD-OUT only (leave-one-doc-out cross-validation).
Uses hand-written docs + programmatically augmented docs for scale.
Reports 95% confidence intervals via bootstrap.
"""

from __future__ import annotations
import sys
import os
import random

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from sample_docs import LABELED_DOCS
from augment import generate_augmented_docs
from features import extract_features, FEATURE_NAMES

from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.detectors import SplitPoint


def doc_to_features_labels(doc):
    lines = [text for text, _ in doc]
    X, y, line_indices = [], [], []
    for i, (text, is_heading) in enumerate(doc):
        if not text.strip():
            continue
        feats = extract_features(lines, i)
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(1 if is_heading else 0)
        line_indices.append(i)
    return np.array(X), np.array(y), line_indices


def heading_detector_detect(full_text: str) -> set[int]:
    hd = HeadingDetector()
    annotations = hd.detect(full_text)
    return {ann.line_number for ann in annotations if isinstance(ann, SplitPoint)}


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def bootstrap_f1_ci(doc_results: list[dict], n_boot: int = 2000, ci: float = 0.95):
    """Bootstrap 95% CI for micro-averaged F1 over documents."""
    rng = np.random.RandomState(42)
    f1s = []
    n = len(doc_results)
    for _ in range(n_boot):
        sample = [doc_results[i] for i in rng.randint(0, n, size=n)]
        tp = sum(r["tp"] for r in sample)
        fp = sum(r["fp"] for r in sample)
        fn = sum(r["fn"] for r in sample)
        _, _, f1 = metrics(tp, fp, fn)
        f1s.append(f1)
    f1s.sort()
    lo = f1s[int((1 - ci) / 2 * n_boot)]
    hi = f1s[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def main():
    # Combine hand-written + augmented docs
    augmented = generate_augmented_docs(n_docs=20)
    all_docs = list(LABELED_DOCS) + augmented
    n_docs = len(all_docs)

    print("=" * 70)
    print("OCR Heading Detection Benchmark (v3 — augmented)")
    print(f"Docs: {len(LABELED_DOCS)} hand-written + {len(augmented)} augmented = {n_docs}")
    print("All ML numbers: leave-one-doc-out CV (held-out)")
    print("=" * 70)

    # Categorize docs
    hand_categories = {
        0: "OCR", 1: "OCR", 2: "OCR", 3: "OCR", 4: "OCR",
        5: "Clean", 6: "Mixed", 7: "OCR",
        8: "Adversarial", 9: "Adversarial", 10: "Adversarial", 11: "Adversarial",
    }
    doc_cats = []
    for i in range(n_docs):
        if i < len(LABELED_DOCS):
            doc_cats.append(hand_categories.get(i, "Unknown"))
        else:
            aug_i = i - len(LABELED_DOCS)
            if aug_i < 4:
                doc_cats.append("Clean-aug")
            elif aug_i < 8:
                doc_cats.append("Mixed-aug")
            else:
                doc_cats.append("OCR-aug")

    # Extract features for all docs
    all_Xs, all_ys, all_line_idxs = [], [], []
    for doc in all_docs:
        X, y, li = doc_to_features_labels(doc)
        all_Xs.append(X)
        all_ys.append(y)
        all_line_idxs.append(li)

    total_samples = sum(len(y) for y in all_ys)
    total_headings = sum(y.sum() for y in all_ys)
    print(f"Total samples: {total_samples} ({int(total_headings)} headings, "
          f"{int(total_samples - total_headings)} body)")

    # --- Leave-one-doc-out CV ---
    heuristic_results = []
    ml_results = []

    for held_out in range(n_docs):
        doc = all_docs[held_out]
        cat = doc_cats[held_out]
        lines = [text for text, _ in doc]
        full_text = "\n".join(lines)

        gt_heading_lines = {
            i for i, (text, is_heading) in enumerate(doc)
            if is_heading and text.strip()
        }

        # Heuristic
        h_detected = heading_detector_detect(full_text)
        h_tp = len(h_detected & gt_heading_lines)
        h_fp = len(h_detected - gt_heading_lines)
        h_fn = len(gt_heading_lines - h_detected)
        hp, hr, hf = metrics(h_tp, h_fp, h_fn)

        # ML — train on all docs except held_out
        train_X = np.vstack([all_Xs[j] for j in range(n_docs) if j != held_out])
        train_y = np.concatenate([all_ys[j] for j in range(n_docs) if j != held_out])

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )
        clf.fit(train_X, train_y)
        y_pred = clf.predict(all_Xs[held_out])

        ml_detected = {
            all_line_idxs[held_out][j]
            for j, p in enumerate(y_pred) if p == 1
        }
        m_tp = len(ml_detected & gt_heading_lines)
        m_fp = len(ml_detected - gt_heading_lines)
        m_fn = len(gt_heading_lines - ml_detected)
        mp, mr, mf = metrics(m_tp, m_fp, m_fn)

        heuristic_results.append({"cat": cat, "tp": h_tp, "fp": h_fp, "fn": h_fn, "f1": hf, "total": len(gt_heading_lines)})
        ml_results.append({"cat": cat, "tp": m_tp, "fp": m_fp, "fn": m_fn, "f1": mf, "total": len(gt_heading_lines)})

    # --- Aggregate by category ---
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY (micro-averaged F1, 95% bootstrap CI)")
    print("=" * 70)

    all_cat_groups = [
        ("OCR (hand-written)", {"OCR"}),
        ("OCR (augmented)", {"OCR-aug"}),
        ("OCR (all)", {"OCR", "OCR-aug"}),
        ("Clean (all)", {"Clean", "Clean-aug"}),
        ("Mixed (all)", {"Mixed", "Mixed-aug"}),
        ("Adversarial", {"Adversarial"}),
        ("ALL", None),
    ]

    for group_name, cat_filter in all_cat_groups:
        if cat_filter is None:
            h_sub = heuristic_results
            m_sub = ml_results
        else:
            h_sub = [r for r in heuristic_results if r["cat"] in cat_filter]
            m_sub = [r for r in ml_results if r["cat"] in cat_filter]

        if not h_sub:
            continue

        h_tp = sum(r["tp"] for r in h_sub)
        h_fp = sum(r["fp"] for r in h_sub)
        h_fn = sum(r["fn"] for r in h_sub)
        hp, hr, hf = metrics(h_tp, h_fp, h_fn)

        m_tp = sum(r["tp"] for r in m_sub)
        m_fp = sum(r["fp"] for r in m_sub)
        m_fn = sum(r["fn"] for r in m_sub)
        mp, mr, mf = metrics(m_tp, m_fp, m_fn)

        n_headings = sum(r["total"] for r in h_sub)
        n_docs_cat = len(h_sub)

        h_lo, h_hi = bootstrap_f1_ci(h_sub)
        m_lo, m_hi = bootstrap_f1_ci(m_sub)

        print(f"\n  {group_name} ({n_docs_cat} docs, {n_headings} headings):")
        print(f"    Heuristic:  P={hp:.2f}  R={hr:.2f}  F1={hf:.2f}  [{h_lo:.2f}, {h_hi:.2f}]  (TP={h_tp} FP={h_fp} FN={h_fn})")
        print(f"    ML (CV):    P={mp:.2f}  R={mr:.2f}  F1={mf:.2f}  [{m_lo:.2f}, {m_hi:.2f}]  (TP={m_tp} FP={m_fp} FN={m_fn})")

        if mf > hf:
            print(f"    → ML wins by +{mf - hf:.2f} F1")
        elif hf > mf:
            print(f"    → Heuristic wins by +{hf - mf:.2f} F1")
        else:
            print(f"    → Tie")

    # --- Feature importance from full model ---
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES (full model, all data)")
    print("=" * 70)

    full_X = np.vstack(all_Xs)
    full_y = np.concatenate(all_ys)
    full_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
    )
    full_clf.fit(full_X, full_y)
    importances = sorted(zip(FEATURE_NAMES, full_clf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {name:25s} {imp:.3f} {bar}")

    # --- Error analysis: worst ML misses ---
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: ML false negatives (missed headings)")
    print("=" * 70)
    fn_count = 0
    for doc_idx in range(n_docs):
        doc = all_docs[doc_idx]
        lines = [text for text, _ in doc]
        gt = {i for i, (t, h) in enumerate(doc) if h and t.strip()}
        pred = {all_line_idxs[doc_idx][j] for j, p in enumerate(ml_results[doc_idx].get("_preds", [])) if p == 1} if "_preds" in ml_results[doc_idx] else set()

        # Reconstruct from tp/fn
        m_res = ml_results[doc_idx]
        if m_res["fn"] > 0:
            # Re-run prediction for this doc to get specifics
            train_X = np.vstack([all_Xs[j] for j in range(n_docs) if j != doc_idx])
            train_y = np.concatenate([all_ys[j] for j in range(n_docs) if j != doc_idx])
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
            )
            clf.fit(train_X, train_y)
            y_pred = clf.predict(all_Xs[doc_idx])
            ml_det = {all_line_idxs[doc_idx][j] for j, p in enumerate(y_pred) if p == 1}
            missed = gt - ml_det
            for li in sorted(missed):
                fn_count += 1
                if fn_count <= 15:
                    print(f"  Doc {doc_idx+1} [{doc_cats[doc_idx]}] line {li}: {repr(lines[li][:70])}")
    if fn_count > 15:
        print(f"  ... and {fn_count - 15} more")
    print(f"  Total ML false negatives: {fn_count}")

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: ML false positives")
    print("=" * 70)
    fp_count = 0
    for doc_idx in range(n_docs):
        doc = all_docs[doc_idx]
        lines = [text for text, _ in doc]
        gt = {i for i, (t, h) in enumerate(doc) if h and t.strip()}
        m_res = ml_results[doc_idx]
        if m_res["fp"] > 0:
            train_X = np.vstack([all_Xs[j] for j in range(n_docs) if j != doc_idx])
            train_y = np.concatenate([all_ys[j] for j in range(n_docs) if j != doc_idx])
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
            )
            clf.fit(train_X, train_y)
            y_pred = clf.predict(all_Xs[doc_idx])
            ml_det = {all_line_idxs[doc_idx][j] for j, p in enumerate(y_pred) if p == 1}
            false_pos = ml_det - gt
            for li in sorted(false_pos):
                fp_count += 1
                if fp_count <= 15:
                    print(f"  Doc {doc_idx+1} [{doc_cats[doc_idx]}] line {li}: {repr(lines[li][:70])}")
    if fp_count > 15:
        print(f"  ... and {fp_count - 15} more")
    print(f"  Total ML false positives: {fp_count}")

    print()


if __name__ == "__main__":
    main()
