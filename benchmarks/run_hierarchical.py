#!/usr/bin/env python3
"""
Hierarchical vs Flat Chunking — Cross-Domain Comparison
========================================================

Runs hierarchical and flat chunking on real documents from four domains
(legal, technical, medical, financial) and prints a comparison table.

No ML or LLM required — just chunkweaver and the corpus files.

Install & run:
    pip install -e .
    python benchmarks/run_hierarchical.py

Corpus: benchmarks/corpus/

Results are printed to stdout and saved to benchmarks/hierarchical_results.json.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chunkweaver import Chunker
from chunkweaver.boundaries import BoundarySpec, detect_boundaries
from chunkweaver.presets import (
    FDA_LABEL,
    FDA_LABEL_LEVELED,
    LEGAL_EU,
    RFC,
    RFC_LEVELED,
    SEC_10K,
    SEC_10K_LEVELED,
)

CORPUS_DIR = PROJECT_ROOT / "benchmarks" / "corpus"
RESULTS_FILE = PROJECT_ROOT / "benchmarks" / "hierarchical_results.json"

# GDPR has leading whitespace on CHAPTER lines — adjusted patterns
LEGAL_EU_REAL: list[str] = [
    r"^\s*CHAPTER\s+[IVX\d]+",
    r"^Article\s+\d+",
    r"^\(\d+\)\s+",
]

LEGAL_EU_REAL_LEVELED: list[BoundarySpec] = [
    (r"^\s*CHAPTER\s+[IVX\d]+", 0),
    (r"^Article\s+\d+", 1),
    (r"^\(\d+\)\s+", 2),
]

CCPA_BOUNDARIES: list[str] = [
    r"^1798\.\d+",
    r"^\([a-z]\)",
    r"^\(\d+\)",
]

CCPA_LEVELED: list[BoundarySpec] = [
    (r"^1798\.\d+", 0),
    (r"^\([a-z]\)", 1),
    (r"^\(\d+\)", 2),
]


@dataclass
class DocResult:
    name: str
    domain: str
    doc_chars: int
    flat_chunks: int
    hier_chunks: int
    reduction_pct: float
    text_preserved: bool
    flat_mean_size: int
    flat_median_size: int
    hier_mean_size: int
    hier_median_size: int
    hier_levels_used: list[int]
    boundary_counts: dict[int, int]


def load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def size_stats(chunks: list[str]) -> Tuple[int, int]:
    sizes = [len(c) for c in chunks]
    return int(statistics.mean(sizes)), int(statistics.median(sizes))


def check_text_preserved(text: str, boundaries: Sequence[BoundarySpec], target: int) -> bool:
    chunker = Chunker(target_size=target, overlap=0, boundaries=list(boundaries), min_size=0)
    chunks = chunker.chunk_with_metadata(text)
    return "".join(c.content_text for c in chunks) == text


def analyze_doc(
    name: str,
    domain: str,
    text: str,
    flat_b: Sequence[BoundarySpec],
    hier_b: Sequence[BoundarySpec],
    target: int = 2048,
) -> DocResult:
    flat = Chunker(target_size=target, overlap=0, boundaries=list(flat_b), min_size=0)
    hier = Chunker(target_size=target, overlap=0, boundaries=list(hier_b), min_size=0)

    flat_chunks = flat.chunk(text)
    hier_chunks = hier.chunk(text)

    flat_mean, flat_med = size_stats(flat_chunks)
    hier_mean, hier_med = size_stats(hier_chunks)

    reduction = (1 - len(hier_chunks) / len(flat_chunks)) * 100 if flat_chunks else 0

    preserved = check_text_preserved(text, hier_b, target)

    hier_meta = hier.chunk_with_metadata(text)
    levels_used = sorted({c.boundary_level for c in hier_meta})

    matches = detect_boundaries(text, list(hier_b))
    level_counts: dict[int, int] = {}
    for m in matches:
        level_counts[m.level] = level_counts.get(m.level, 0) + 1

    return DocResult(
        name=name,
        domain=domain,
        doc_chars=len(text),
        flat_chunks=len(flat_chunks),
        hier_chunks=len(hier_chunks),
        reduction_pct=round(reduction, 1),
        text_preserved=preserved,
        flat_mean_size=flat_mean,
        flat_median_size=flat_med,
        hier_mean_size=hier_mean,
        hier_median_size=hier_med,
        hier_levels_used=levels_used,
        boundary_counts=level_counts,
    )


def print_table(results: list[DocResult]) -> None:
    header = (
        f"{'Document':<28} {'Domain':<10} {'Size':>8} "
        f"{'Flat':>6} {'Hier':>6} {'Δ%':>7} "
        f"{'Flat med':>9} {'Hier med':>9} "
        f"{'Preserved':>10} {'Levels':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        delta = f"{r.reduction_pct:+.1f}%"
        preserved = "YES" if r.text_preserved else "NO"
        levels = ",".join(str(l) for l in r.hier_levels_used)
        print(
            f"{r.name:<28} {r.domain:<10} {r.doc_chars:>7,} "
            f"{r.flat_chunks:>6} {r.hier_chunks:>6} {delta:>7} "
            f"{r.flat_median_size:>9} {r.hier_median_size:>9} "
            f"{preserved:>10} {levels:>8}"
        )

    print(sep)

    total_flat = sum(r.flat_chunks for r in results)
    total_hier = sum(r.hier_chunks for r in results)
    total_reduction = (1 - total_hier / total_flat) * 100 if total_flat else 0
    all_preserved = all(r.text_preserved for r in results)

    print(
        f"{'TOTAL':<28} {'':10} {'':>8} "
        f"{total_flat:>6} {total_hier:>6} {total_reduction:+.1f}% "
        f"{'':>9} {'':>9} "
        f"{'ALL YES' if all_preserved else 'SOME NO':>10}"
    )
    print()


def print_boundary_detail(results: list[DocResult]) -> None:
    print("Boundary detection detail:")
    print(f"  {'Document':<28} {'Level 0':>9} {'Level 1':>9} {'Level 2':>9} {'Total':>7}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")
    for r in results:
        l0 = r.boundary_counts.get(0, 0)
        l1 = r.boundary_counts.get(1, 0)
        l2 = r.boundary_counts.get(2, 0)
        total = sum(r.boundary_counts.values())
        print(f"  {r.name:<28} {l0:>9} {l1:>9} {l2:>9} {total:>7}")
    print()


def main() -> None:
    print("=" * 70)
    print("chunkweaver: Hierarchical vs Flat Chunking — Cross-Domain Comparison")
    print("=" * 70)

    configs: list[Tuple[str, str, Path, Sequence[BoundarySpec], Sequence[BoundarySpec]]] = []

    # Legal
    gdpr = CORPUS_DIR / "eu_gdpr_2016_679.txt"
    if gdpr.exists():
        configs.append(("EU GDPR (2016/679)", "Legal", gdpr, LEGAL_EU_REAL, LEGAL_EU_REAL_LEVELED))

    ccpa = CORPUS_DIR / "ccpa_1798.txt"
    if ccpa.exists():
        configs.append(("CCPA (Cal. Civ. 1798)", "Legal", ccpa, CCPA_BOUNDARIES, CCPA_LEVELED))

    ai_act = CORPUS_DIR / "eu_ai_act_2024_1689.txt"
    if ai_act.exists():
        configs.append(("EU AI Act (2024/1689)", "Legal", ai_act, LEGAL_EU_REAL, LEGAL_EU_REAL_LEVELED))

    # RFCs
    for rfc_file in sorted(CORPUS_DIR.glob("rfc*.txt")):
        label = rfc_file.stem.replace("_", " ").upper()
        configs.append((label, "RFC", rfc_file, RFC, RFC_LEVELED))

    # Medical
    fda = CORPUS_DIR / "fda_metformin_label.txt"
    if fda.exists():
        configs.append(("FDA Metformin Label", "Medical", fda, FDA_LABEL, FDA_LABEL_LEVELED))

    # Financial
    sec = CORPUS_DIR / "sec_enron_10k_2000.txt"
    if sec.exists():
        configs.append(("SEC Enron 10-K (2000)", "Financial", sec, SEC_10K, SEC_10K_LEVELED))

    if not configs:
        print("\nERROR: No corpus documents found. Check paths.")
        sys.exit(1)

    print(f"\n  Documents: {len(configs)}")
    print(f"  Target size: 2048 chars")
    print(f"  Overlap: 0 (disabled for fair comparison)")

    start = time.time()
    results: list[DocResult] = []

    for name, domain, path, flat_b, hier_b in configs:
        text = load(path)
        r = analyze_doc(name, domain, text, flat_b, hier_b, target=2048)
        results.append(r)
        print(f"  {name}: {r.flat_chunks} → {r.hier_chunks} chunks ({r.reduction_pct:+.1f}%)")

    elapsed = time.time() - start

    print_table(results)
    print_boundary_detail(results)

    # Also run at multiple target sizes for key documents
    print("Target size sensitivity (GDPR, RFC-JWT, FDA, SEC 10-K):")
    sensitivity_docs = [
        ("EU GDPR", CORPUS_DIR / "eu_gdpr_2016_679.txt", LEGAL_EU_REAL, LEGAL_EU_REAL_LEVELED),
        ("RFC-JWT", CORPUS_DIR / "rfc7519_jwt.txt", RFC, RFC_LEVELED),
        ("FDA Label", CORPUS_DIR / "fda_metformin_label.txt", FDA_LABEL, FDA_LABEL_LEVELED),
        ("SEC 10-K", CORPUS_DIR / "sec_enron_10k_2000.txt", SEC_10K, SEC_10K_LEVELED),
    ]
    targets = [512, 1024, 2048, 4096, 8192]

    print(f"  {'Document':<12}", end="")
    for t in targets:
        print(f"  {t:>14}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in targets:
        print(f"  {'-'*14}", end="")
    print()

    for name, path, flat_b, hier_b in sensitivity_docs:
        if not path.exists():
            continue
        text = load(path)
        print(f"  {name:<12}", end="")
        for t in targets:
            flat_n = len(Chunker(target_size=t, overlap=0, boundaries=list(flat_b), min_size=0).chunk(text))
            hier_n = len(Chunker(target_size=t, overlap=0, boundaries=list(hier_b), min_size=0).chunk(text))
            reduction = (1 - hier_n / flat_n) * 100 if flat_n else 0
            print(f"  {hier_n:>5}/{flat_n:<5} {reduction:+.0f}%", end="")
        print()

    print(f"\n  Time: {elapsed:.1f}s")

    # Save results
    output = {
        "benchmark": "hierarchical-vs-flat",
        "target_size": 2048,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "documents": len(results),
        "results": [asdict(r) for r in results],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
