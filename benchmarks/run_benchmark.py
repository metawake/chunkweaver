#!/usr/bin/env python3
"""
chunkweaver Benchmark — Self-contained
=======================================

Compares chunkweaver vs baselines on EU GDPR legal text using
NeedleCoverage@5: what fraction of ground-truth text spans appear in
the top-5 retrieved chunks.

Strategies tested:
  1. naive-1024        — fixed-size 1024-char splits, 64-char overlap
  2. naive-1300        — same algorithm, sized to match chunkweaver avg
  3. langchain-RCTS    — RecursiveCharacterTextSplitter (if installed)
  4. chunkweaver       — boundary-aware with LEGAL_EU preset
  5. chunkweaver + ol  — same with 2-sentence overlap

Install & run:
    pip install -e ".[benchmark]"         # or: pip install sentence-transformers
    python benchmarks/run_benchmark.py

Optional (adds LangChain baseline):
    pip install langchain-text-splitters

Results are printed to stdout and saved to benchmarks/results.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent
CORPUS_FILE = BENCHMARK_DIR / "corpus" / "eu_gdpr_2016_679.txt"
NEEDLES_FILE = BENCHMARK_DIR / "needles.json"
RESULTS_FILE = BENCHMARK_DIR / "results.json"

sys.path.insert(0, str(PROJECT_ROOT))

from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
TARGET_SIZE = 1024


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def naive_chunk(text: str, size: int, overlap: int = 64) -> list[str]:
    """Fixed-size character chunker with overlap (baseline)."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            last_space = text[start:end].rfind(" ")
            if last_space > size // 2:
                end = start + last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        step = max((end - start) - overlap, 1)
        start += step
    return chunks


def langchain_chunk(text: str, size: int, overlap: int = 64) -> list[str] | None:
    """RecursiveCharacterTextSplitter baseline. Returns None if not installed."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def avg_chunk_size(chunks: list[str]) -> int:
    return sum(len(c) for c in chunks) // len(chunks) if chunks else 0


def compute_needle_coverage(chunk_texts: list[str], needles: list[dict]) -> list[dict]:
    """Check which needles appear in the concatenated top-k chunks."""
    concat = normalize_ws("\n".join(chunk_texts)).lower()
    results = []
    for needle in needles:
        needle_norm = normalize_ws(needle["text"]).lower()
        results.append({
            "text": needle["text"][:60],
            "source": needle.get("source", ""),
            "difficulty": needle.get("difficulty", ""),
            "found": needle_norm in concat,
        })
    return results


def run():
    from sentence_transformers import SentenceTransformer

    print("=" * 70)
    print("chunkweaver Benchmark: GDPR NeedleCoverage@5")
    print("=" * 70)

    if not CORPUS_FILE.exists():
        print(f"ERROR: Corpus not found at {CORPUS_FILE}")
        print("Run from the chunkweaver project root.")
        sys.exit(1)

    text = CORPUS_FILE.read_text(errors="ignore")
    print(f"  Corpus: {CORPUS_FILE.name} ({len(text):,} chars)")

    with open(NEEDLES_FILE) as f:
        needles_data = json.load(f)
    gdpr_queries = [q for q in needles_data["queries"] if q.get("domain") == "gdpr"]
    total_needles = sum(len(q["needles"]) for q in gdpr_queries)
    print(f"  Queries: {len(gdpr_queries)}, Needles: {total_needles}")

    # --- Build strategies ---
    sc = Chunker(target_size=TARGET_SIZE, boundaries=LEGAL_EU)
    sc_chunks = [c.text for c in sc.chunk_with_metadata(text)]
    sc_avg = avg_chunk_size(sc_chunks)

    sc_ol = Chunker(target_size=TARGET_SIZE, boundaries=LEGAL_EU,
                    overlap=2, overlap_unit="sentence")
    sc_ol_chunks = [c.text for c in sc_ol.chunk_with_metadata(text)]

    strategies: dict[str, list[str]] = {}

    strategies[f"naive-{TARGET_SIZE}"] = naive_chunk(text, size=TARGET_SIZE, overlap=64)

    # Fair-size naive: match chunkweaver's avg chunk size so the comparison
    # isolates boundary quality from chunk length advantage.
    strategies[f"naive-{sc_avg} (size-matched)"] = naive_chunk(text, size=sc_avg, overlap=64)

    lc_chunks = langchain_chunk(text, size=sc_avg, overlap=64)
    if lc_chunks is not None:
        strategies[f"langchain-RCTS-{sc_avg}"] = lc_chunks
    else:
        print("  (langchain-text-splitters not installed, skipping RCTS baseline)")

    strategies["chunkweaver (LEGAL_EU)"] = sc_chunks
    strategies["chunkweaver + 2-sent overlap"] = sc_ol_chunks

    print(f"\n  Strategies:")
    for name, chunks in strategies.items():
        print(f"    {name}: {len(chunks)} chunks, avg {avg_chunk_size(chunks)} chars")

    # --- Embed and evaluate ---
    print(f"\n  Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    results = {}
    start_time = time.time()

    for strat_name, chunks in strategies.items():
        print(f"\n  [{strat_name}]")
        print(f"    Embedding {len(chunks)} chunks...")
        chunk_embeddings = model.encode(chunks, show_progress_bar=True, batch_size=128)

        total_found = 0
        query_results = []

        for query in gdpr_queries:
            query_emb = model.encode(query["text"], show_progress_bar=False)
            sims = np.dot(chunk_embeddings, query_emb) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
            )
            top_indices = np.argsort(-sims)[:TOP_K]
            top_texts = [chunks[i] for i in top_indices]

            needle_results = compute_needle_coverage(top_texts, query["needles"])
            found = sum(1 for r in needle_results if r["found"])
            total_found += found

            query_results.append({
                "id": query["id"],
                "found": found,
                "total": len(query["needles"]),
                "coverage": round(found / len(query["needles"]), 4) if query["needles"] else 0,
                "details": needle_results,
            })

        coverage = total_found / total_needles if total_needles > 0 else 0
        avg_sz = avg_chunk_size(chunks)

        results[strat_name] = {
            "n_chunks": len(chunks),
            "avg_chunk_size": avg_sz,
            "needles_found": total_found,
            "needles_total": total_needles,
            "coverage": round(coverage, 4),
            "per_query": query_results,
        }
        print(f"    NeedleCoverage@{TOP_K}: {total_found}/{total_needles} ({coverage:.1%})")

    elapsed = time.time() - start_time

    # --- Print results table ---
    col_w = max(len(n) for n in results) + 2
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  {'Strategy':<{col_w}} {'Chunks':>7} {'Avg':>5} {'Coverage':>12}")
    print(f"  {'-' * col_w} {'-' * 7} {'-' * 5} {'-' * 12}")

    sorted_r = sorted(results.items(), key=lambda x: x[1]["coverage"], reverse=True)
    for name, r in sorted_r:
        print(f"  {name:<{col_w}} {r['n_chunks']:>7} {r['avg_chunk_size']:>5} "
              f"{r['needles_found']}/{r['needles_total']:>3} ({r['coverage']:.1%})")

    best_name, best = sorted_r[0]
    worst_name, worst = sorted_r[-1]
    delta_pp = (best["coverage"] - worst["coverage"]) * 100
    print(f"\n  Best:  {best_name} ({best['coverage']:.1%})")
    print(f"  Worst: {worst_name} ({worst['coverage']:.1%})")
    print(f"  Delta: {delta_pp:+.1f} pp")

    # Per-query detail
    print(f"\n  Per-query breakdown:")
    short_names = []
    for n, _ in sorted_r:
        short = n[:20]
        short_names.append(short)
    header = f"  {'Query':<14}"
    for s in short_names:
        header += f"  {s:>20}"
    print(header)
    print(f"  {'-' * 14}" + f"  {'-' * 20}" * len(short_names))

    for qi in range(len(gdpr_queries)):
        qid = sorted_r[0][1]["per_query"][qi]["id"]
        row = f"  {qid:<14}"
        for name, r in sorted_r:
            qr = r["per_query"][qi]
            cell = f"{qr['found']}/{qr['total']} ({qr['coverage']:.0%})"
            row += f"  {cell:>20}"
        print(row)

    print(f"\n  Time: {elapsed:.1f}s | Model: {MODEL_NAME} | Top-K: {TOP_K}")

    # --- Save ---
    output = {
        "benchmark": "chunkweaver-vs-baselines",
        "corpus": CORPUS_FILE.name,
        "model": MODEL_NAME,
        "target_size": TARGET_SIZE,
        "top_k": TOP_K,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
