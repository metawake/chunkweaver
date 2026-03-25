# ML Detector Experiments

These examples demonstrate how chunkweaver's `BoundaryDetector` ABC can be extended with small, specialized ML models for cases where heuristics struggle.

The core insight: chunkweaver's architecture doesn't care *how* a detector decides where to split — regex, heuristic scoring, or a trained classifier all produce the same `SplitPoint` / `KeepTogetherRegion` annotations. ML detectors are drop-in replacements.

## Experiments

### [ocr_heading_detector](./ocr_heading_detector/)

Detects section headings in OCR-damaged legal/regulatory documents where letterspacing artifacts (`D E F I N I T I O N S`) break regex patterns. A GradientBoosting classifier achieves 0.97 F1 vs 0.76 for the heuristic HeadingDetector on held-out cross-validation (32 docs, 642 samples).

### [clinical_section_detector](./clinical_section_detector/)

Detects section boundaries in clinical notes (HPI, Assessment, Plan, etc.) where abbreviated styles and implicit transitions make headings hard to catch with patterns alone.

## Requirements

```bash
pip install scikit-learn joblib numpy chunkweaver
```

Each experiment is self-contained: train, evaluate, and benchmark independently.
