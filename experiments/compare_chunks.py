"""Compare chunk quality: FINANCIAL preset only vs FINANCIAL + HeadingDetector.

For each filing, shows:
- Chunk counts, avg size, size distribution
- How many chunks start at a real heading (structural alignment)
- Example chunks that differ between the two approaches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkweaver import Chunker
from chunkweaver.presets import FINANCIAL
from experiments.heading_detector import detect_headings, headings_to_boundary_patterns


def chunk_stats(chunks, label):
    sizes = [len(c) for c in chunks]
    avg = sum(sizes) / len(sizes) if sizes else 0
    median = sorted(sizes)[len(sizes) // 2] if sizes else 0
    return {
        "label": label,
        "count": len(chunks),
        "avg_size": int(avg),
        "median_size": median,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
    }


def first_line(chunk_text):
    """Return first non-empty line of a chunk."""
    for line in chunk_text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "(empty)"


def main():
    filing_dir = Path(__file__).parent / "10k_filings"
    files = sorted(filing_dir.glob("*.txt"))

    if not files:
        print("No 10-K files. Run download_10k.py first.")
        return

    target_size = 1024

    for fpath in files:
        text = fpath.read_text()

        # Strategy A: FINANCIAL preset only
        chunker_a = Chunker(
            target_size=target_size,
            overlap=2,
            overlap_unit="sentence",
            boundaries=FINANCIAL,
        )
        chunks_a = chunker_a.chunk(text)

        # Strategy B: FINANCIAL + HeadingDetector
        headings = detect_headings(text, min_score=5.0)
        heading_patterns = headings_to_boundary_patterns(headings)
        combined_boundaries = FINANCIAL + heading_patterns

        chunker_b = Chunker(
            target_size=target_size,
            overlap=2,
            overlap_unit="sentence",
            boundaries=combined_boundaries,
        )
        chunks_b = chunker_b.chunk(text)

        # Strategy C: No boundaries at all (pure paragraph fallback)
        chunker_c = Chunker(
            target_size=target_size,
            overlap=2,
            overlap_unit="sentence",
            boundaries=[],
        )
        chunks_c = chunker_c.chunk(text)

        stats_a = chunk_stats(chunks_a, "FINANCIAL only")
        stats_b = chunk_stats(chunks_b, "FINANCIAL + Heading")
        stats_c = chunk_stats(chunks_c, "No boundaries")

        print(f"\n{'='*75}")
        print(f"  {fpath.name}  ({len(text):,} chars)")
        print(f"{'='*75}")

        for s in [stats_c, stats_a, stats_b]:
            print(f"  {s['label']:22s}  {s['count']:4d} chunks  "
                  f"avg={s['avg_size']:4d}  med={s['median_size']:4d}  "
                  f"min={s['min_size']:4d}  max={s['max_size']:5d}")

        print(f"\n  Headings detected: {len(headings)}")
        print(f"  Extra chunks from headings: {stats_b['count'] - stats_a['count']}")

        # Show what chunk starts look like
        print(f"\n  --- FINANCIAL only: first lines of first 10 chunks ---")
        for i, c in enumerate(chunks_a[:10]):
            print(f"    [{i:2d}] {first_line(c)}")

        print(f"\n  --- FINANCIAL + Heading: first lines of first 10 chunks ---")
        for i, c in enumerate(chunks_b[:10]):
            print(f"    [{i:2d}] {first_line(c)}")

        # Find chunks in B that start at a heading (structural alignment)
        heading_texts = {h.text for h in headings}
        aligned_b = sum(1 for c in chunks_b if first_line(c) in heading_texts)
        aligned_a = sum(1 for c in chunks_a if first_line(c) in heading_texts)

        print(f"\n  Chunks starting at a detected heading:")
        print(f"    FINANCIAL only:     {aligned_a:4d} / {stats_a['count']} "
              f"({100*aligned_a/max(stats_a['count'],1):.1f}%)")
        print(f"    FINANCIAL + Heading: {aligned_b:4d} / {stats_b['count']} "
              f"({100*aligned_b/max(stats_b['count'],1):.1f}%)")

        # Show chunks from B that start at headings the FINANCIAL preset missed
        financial_chunker_starts = {first_line(c) for c in chunks_a}
        new_heading_chunks = [
            c for c in chunks_b
            if first_line(c) in heading_texts
            and first_line(c) not in financial_chunker_starts
        ]

        if new_heading_chunks:
            print(f"\n  --- NEW heading-aligned chunks (not in FINANCIAL-only) ---")
            for c in new_heading_chunks[:15]:
                print(f"    -> {first_line(c)}")

        print()


if __name__ == "__main__":
    main()
