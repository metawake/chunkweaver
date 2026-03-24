"""Compare chunk quality with and without TableDetector keep-together regions.

Measures:
1. How many table regions get split across multiple chunks
2. How many chunks are "orphaned numerics" — mostly numbers with no header context
3. How chunk stats change when tables are kept together
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkweaver import Chunker
from chunkweaver.presets import FINANCIAL
from experiments.heading_detector import detect_headings, headings_to_boundary_patterns
from experiments.table_detector import detect_tables, TableRegion

from typing import List, Tuple


def _char_ranges(text: str, chunks: list[str]) -> list[tuple[int, int]]:
    """Return (start, end) char offsets for each chunk in the original text."""
    ranges = []
    pos = 0
    for chunk in chunks:
        idx = text.find(chunk[:80], pos)
        if idx == -1:
            idx = text.find(chunk[:40], max(0, pos - 200))
        if idx == -1:
            ranges.append((pos, pos + len(chunk)))
        else:
            ranges.append((idx, idx + len(chunk)))
            pos = idx + 1
    return ranges


def _line_to_char(lines: list[str], line_no: int) -> int:
    """Convert a line number to a character offset."""
    offset = 0
    for i in range(min(line_no, len(lines))):
        offset += len(lines[i]) + 1
    return offset


def _table_region_char_ranges(text: str, regions: List[TableRegion]) -> List[Tuple[int, int]]:
    """Convert table regions (line-based) to char ranges."""
    lines = text.split("\n")
    result = []
    for r in regions:
        start = _line_to_char(lines, r.start_line)
        end = _line_to_char(lines, r.end_line)
        result.append((start, end))
    return result


def _is_orphan_numeric(chunk: str) -> bool:
    """A chunk that's mostly numbers with no semantic header."""
    stripped = chunk.strip()
    if not stripped:
        return False
    digit_chars = sum(1 for c in stripped if c.isdigit())
    total = len(stripped)
    if total < 50:
        return False

    first_line = ""
    for line in stripped.split("\n"):
        if line.strip():
            first_line = line.strip()
            break

    first_digits = sum(1 for c in first_line if c.isdigit())
    first_ratio = first_digits / max(len(first_line), 1)

    return digit_chars / total > 0.25 and first_ratio > 0.3


def _merge_table_text(text: str, regions: List[TableRegion]) -> str:
    """Protect table regions by removing internal newlines that could trigger splits.

    Strategy: Replace double-newlines inside table regions with single newlines,
    preventing paragraph-break-based splitting within tables.
    """
    if not regions:
        return text

    lines = text.split("\n")
    protected_lines = set()
    for r in regions:
        for i in range(r.start_line, min(r.end_line, len(lines))):
            protected_lines.add(i)

    result = []
    for i, line in enumerate(lines):
        if i in protected_lines:
            if line.strip() == "" and i + 1 < len(lines) and i + 1 in protected_lines:
                result.append(" ")
            else:
                result.append(line)
        else:
            result.append(line)

    return "\n".join(result)


def main():
    filing_dir = Path(__file__).parent / "10k_filings"
    files = sorted(filing_dir.glob("*.txt"))
    # Skip Microsoft (shattered table format)
    files = [f for f in files if "microsoft" not in f.name]

    if not files:
        print("No 10-K files. Run download_10k.py first.")
        return

    target_size = 1024

    total_splits_without = 0
    total_splits_with = 0
    total_orphans_without = 0
    total_orphans_with = 0
    total_tables = 0

    for fpath in files:
        text = fpath.read_text()
        lines = text.split("\n")

        headings = detect_headings(text, min_score=4.0)
        heading_patterns = headings_to_boundary_patterns(headings)
        boundaries = FINANCIAL + heading_patterns

        tables = detect_tables(text)
        table_char_ranges = _table_region_char_ranges(text, tables)
        total_tables += len(tables)

        # Strategy A: FINANCIAL + HeadingDetector (no table protection)
        chunker_a = Chunker(
            target_size=target_size,
            overlap=2,
            overlap_unit="sentence",
            boundaries=boundaries,
        )
        chunks_a = chunker_a.chunk(text)
        ranges_a = _char_ranges(text, chunks_a)

        # Strategy B: Same boundaries, but protect table regions
        protected_text = _merge_table_text(text, tables)
        chunker_b = Chunker(
            target_size=target_size,
            overlap=2,
            overlap_unit="sentence",
            boundaries=boundaries,
        )
        chunks_b = chunker_b.chunk(protected_text)
        ranges_b = _char_ranges(protected_text, chunks_b)

        # Count table splits: how many table regions span 2+ chunks
        def count_table_splits(chunk_ranges, table_ranges):
            splits = 0
            for tstart, tend in table_ranges:
                containing_chunks = 0
                for cstart, cend in chunk_ranges:
                    if cstart < tend and cend > tstart:
                        containing_chunks += 1
                if containing_chunks > 1:
                    splits += 1
            return splits

        splits_a = count_table_splits(ranges_a, table_char_ranges)
        splits_b = count_table_splits(ranges_b, table_char_ranges)
        total_splits_without += splits_a
        total_splits_with += splits_b

        orphans_a = sum(1 for c in chunks_a if _is_orphan_numeric(c))
        orphans_b = sum(1 for c in chunks_b if _is_orphan_numeric(c))
        total_orphans_without += orphans_a
        total_orphans_with += orphans_b

        print(f"\n{'='*75}")
        print(f"  {fpath.name}  ({len(text):,} chars)")
        print(f"{'='*75}")
        print(f"  Tables detected: {len(tables)}")
        print(f"  Chunks:  without protection={len(chunks_a):4d}  "
              f"with protection={len(chunks_b):4d}")
        print(f"  Table splits:  without={splits_a:3d}/{len(tables)}  "
              f"with={splits_b:3d}/{len(tables)}")
        print(f"  Orphan numeric chunks:  without={orphans_a:3d}  "
              f"with={orphans_b:3d}")

        # Show examples of orphan chunks
        if orphans_a > 0:
            print(f"\n  --- Orphan numeric chunks (without protection) ---")
            shown = 0
            for i, c in enumerate(chunks_a):
                if _is_orphan_numeric(c) and shown < 3:
                    first = c.strip().split("\n")[0][:80]
                    print(f"    chunk[{i}] ({len(c)} chars): {first}")
                    shown += 1

        # Show tables that got split without protection but not with
        fixed = []
        for ti, (tstart, tend) in enumerate(table_char_ranges):
            a_containing = sum(1 for cs, ce in ranges_a if cs < tend and ce > tstart)
            b_containing = sum(1 for cs, ce in ranges_b if cs < tend and ce > tstart)
            if a_containing > 1 and b_containing <= 1:
                fixed.append(tables[ti])
        if fixed:
            print(f"\n  --- Tables fixed by protection ({len(fixed)}) ---")
            for t in fixed[:5]:
                print(f"    L{t.start_line}-{t.end_line}: {t.header_text[:60]}  "
                      f"({t.num_data_lines} data rows, {t.char_size} chars)")

    print(f"\n{'='*75}")
    print(f"  TOTALS (across {len(files)} filings, {total_tables} tables)")
    print(f"{'='*75}")
    print(f"  Table splits:  without protection={total_splits_without}  "
          f"with protection={total_splits_with}")
    print(f"  Orphan numeric chunks:  without={total_orphans_without}  "
          f"with={total_orphans_with}")
    print(f"  Table splits fixed: {total_splits_without - total_splits_with}")
    print()


if __name__ == "__main__":
    main()
