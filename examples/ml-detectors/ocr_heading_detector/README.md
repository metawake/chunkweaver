# OCR Heading Detector — ML vs Heuristic Experiment

Can a small ML model beat regex-based heading detection on OCR-damaged text?

**TL;DR:** Yes — on letterspaced / fragmented headings from bad PDF extraction, a 100-tree gradient boosting classifier achieves **0.97 F1** vs the heuristic HeadingDetector's **0.76 F1** (held-out, leave-one-doc-out CV, 32 docs, 642 samples, non-overlapping 95% CIs).

## The problem

PDF extractors and OCR engines often produce letterspaced headings:

```
D E F I N I T I O N S              ← should be "DEFINITIONS"
i n vi i te d  p a rt i es         ← should be "invited parties"
Sect ion 3 Com pen sat ion         ← should be "Section 3 Compensation"
I t e m   1 A .   R i s k          ← should be "Item 1A. Risk"
```

No regex or keyword list can match these. HeadingDetector relies on casing, known prefixes, and whitespace context — it catches clean headings perfectly but misses 20%+ of OCR-damaged ones and produces false positives on short body lines (signatures, list items, captions).

## How it works

The ML detector (`MLOCRHeadingDetector`) is a `BoundaryDetector` subclass that uses a trained `GradientBoostingClassifier` to classify each line as heading/body. It extracts 19 features per line:

**Most important (from feature importance on full dataset):**

| Feature | Importance | What it captures |
|---------|-----------|-----------------|
| `upper_ratio` | 0.525 | Headings tend to be uppercase |
| `punct_ratio` | 0.152 | Headings have less punctuation |
| `space_ratio` | 0.086 | OCR letterspacing inflates spaces |
| `expansion_ratio` | 0.067 | How much the line "expands" from added spaces |

## Results (held-out CV)

| Category | Heuristic F1 | ML F1 | 95% CI (ML) |
|----------|-------------|-------|-------------|
| OCR-damaged (18 docs) | 0.72 | **0.96** | [0.92, 0.99] |
| Clean (5 docs) | 0.80 | **0.98** | [0.95, 1.00] |
| Adversarial (4 docs) | 0.86 | **0.96** | [0.86, 1.00] |
| **Overall (32 docs)** | **0.76** | **0.97** | **[0.94, 0.99]** |

The heuristic's main failure mode is **false positives** (61 total) — it flags short body lines like `"EMPLOYER:"`, `"Name: John Smith"`, `"- W-9 tax form"` as headings. The ML model has 3 false positives.

## Running the experiment

```bash
pip install scikit-learn joblib numpy chunkweaver

# Train the model
python train.py

# Run the benchmark
python benchmark.py
```

## File structure

| File | Purpose |
|------|---------|
| `sample_docs.py` | 12 hand-written labeled documents (OCR, clean, adversarial) |
| `augment.py` | Programmatic OCR damage + adversarial generation (20 more docs) |
| `features.py` | 19-feature extraction per line |
| `train.py` | Train GradientBoosting, leave-one-doc-out CV, save model |
| `detector.py` | `MLOCRHeadingDetector(BoundaryDetector)` — drop-in for chunkweaver |
| `benchmark.py` | Full comparison with bootstrap CIs and error analysis |

## Using in chunkweaver

```python
from chunkweaver import Chunker
from detector import MLOCRHeadingDetector

chunker = Chunker(
    target_size=1024,
    detectors=[MLOCRHeadingDetector()],
)

chunks = chunker.chunk(ocr_damaged_text)
```

## Limitations

- Trained on **synthetic data** — real OCR artifacts may differ
- 32 documents, 142 headings — statistically meaningful but not production-validated
- The model relies heavily on `upper_ratio` — lowercase OCR-damaged headings (like `"g over ni ng l aw"`) are the hardest case
- Not tested on non-Latin scripts or right-to-left text
