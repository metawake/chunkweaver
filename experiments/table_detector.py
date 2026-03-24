"""TableDetector spike — detect financial table regions as keep-together zones.

Identifies runs of numeric data lines, extends backward to include headers,
and forward to include footnotes. Returns regions that should not be split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class TableRegion:
    start_line: int
    end_line: int       # exclusive
    header_text: str    # first non-blank line (table title)
    num_data_lines: int
    char_size: int      # total characters in region


# Patterns for numeric content in financial tables
_MULTI_NUM_RE = re.compile(
    r"[\d,]+\.?\d*"     # numbers with optional commas/decimals
)
_DOLLAR_RE = re.compile(r"\$\s*[\d,]+")
_YEAR_LINE_RE = re.compile(r"^\s*(20\d{2}\s+){2,}")  # "2025 2024 2023"
_FOOTNOTE_RE = re.compile(r"^\s*\([a-z]\)")           # (a), (b), etc.
_UNITS_RE = re.compile(
    r"\(in (millions|billions|thousands)\)", re.IGNORECASE
)


def _is_numeric_line(line: str) -> bool:
    """Check if a line looks like a financial table data row."""
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return False

    nums = _MULTI_NUM_RE.findall(stripped)
    if len(nums) < 2:
        return False

    digit_chars = sum(1 for c in stripped if c.isdigit())
    total_chars = len(stripped)
    if total_chars == 0:
        return False

    return digit_chars / total_chars > 0.15


def _is_table_header(line: str) -> bool:
    """Check if a line looks like a table header (column labels, units, title)."""
    stripped = line.strip()
    if not stripped:
        return False

    if _YEAR_LINE_RE.match(stripped):
        return True
    if _UNITS_RE.search(stripped):
        return True

    return False


def detect_tables(
    text: str,
    min_data_lines: int = 3,
    header_lookback: int = 4,
    max_gap: int = 2,
) -> List[TableRegion]:
    """Detect table regions in plain text.

    Returns regions sorted by start_line. Each region is a contiguous block
    that should be kept together during chunking.
    """
    lines = text.split("\n")
    n = len(lines)
    stripped = [ln.strip() for ln in lines]

    # Phase 1: Mark each line as numeric or not
    is_num = [_is_numeric_line(ln) for ln in lines]

    # Phase 2: Find runs of numeric lines (allowing small gaps)
    runs: List[Tuple[int, int]] = []  # (start, end_exclusive)
    i = 0
    while i < n:
        if is_num[i]:
            run_start = i
            run_end = i + 1
            gap = 0
            j = i + 1
            while j < n:
                if is_num[j]:
                    run_end = j + 1
                    gap = 0
                elif stripped[j] == "":
                    gap += 1
                    if gap > max_gap:
                        break
                else:
                    # Non-numeric, non-blank line inside a run
                    # Could be a sub-header within the table (e.g., "Noncompensation expense:")
                    if len(stripped[j]) < 60 and gap <= 1:
                        run_end = j + 1
                        gap = 0
                    else:
                        break
                j += 1
            data_count = sum(1 for k in range(run_start, run_end) if is_num[k])
            if data_count >= min_data_lines:
                runs.append((run_start, run_end))
            i = run_end
        else:
            i += 1

    # Phase 3: Extend each run backward to include headers
    regions: List[TableRegion] = []
    for run_start, run_end in runs:
        # Look back for header lines
        header_start = run_start
        for k in range(run_start - 1, max(run_start - header_lookback - 1, -1), -1):
            s = stripped[k]
            if not s:
                continue
            if _is_table_header(s):
                header_start = k
            elif len(s) < 60 and not _is_numeric_line(lines[k]):
                header_start = k
            else:
                break

        # Extend forward to include footnotes
        foot_end = run_end
        for k in range(run_end, min(run_end + 6, n)):
            s = stripped[k]
            if not s:
                continue
            if _FOOTNOTE_RE.match(s):
                foot_end = k + 1
            else:
                break

        header_text = ""
        for k in range(header_start, run_end):
            if stripped[k]:
                header_text = stripped[k]
                break

        char_size = sum(len(lines[k]) + 1 for k in range(header_start, foot_end))

        regions.append(TableRegion(
            start_line=header_start,
            end_line=foot_end,
            header_text=header_text[:80],
            num_data_lines=sum(1 for k in range(run_start, run_end) if is_num[k]),
            char_size=char_size,
        ))

    # Merge overlapping regions
    if regions:
        merged = [regions[0]]
        for r in regions[1:]:
            prev = merged[-1]
            if r.start_line <= prev.end_line + 2:
                merged[-1] = TableRegion(
                    start_line=prev.start_line,
                    end_line=max(prev.end_line, r.end_line),
                    header_text=prev.header_text,
                    num_data_lines=prev.num_data_lines + r.num_data_lines,
                    char_size=prev.char_size + r.char_size,
                )
            else:
                merged.append(r)
        regions = merged

    return regions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    filing_dir = Path(__file__).parent / "10k_filings"
    files = sorted(filing_dir.glob("*.txt"))

    for fpath in files:
        text = fpath.read_text()
        regions = detect_tables(text)

        total_chars = sum(r.char_size for r in regions)
        pct = 100 * total_chars / len(text) if text else 0

        print(f"\n{'='*70}")
        print(f"  {fpath.name}  ({len(text):,} chars)")
        print(f"  Detected {len(regions)} table regions  "
              f"({total_chars:,} chars, {pct:.1f}% of document)")
        print(f"{'='*70}")

        for r in regions:
            print(f"  L{r.start_line:5d}-{r.end_line:<5d}  "
                  f"{r.num_data_lines:3d} data rows  "
                  f"{r.char_size:6,} chars  "
                  f"{r.header_text[:60]}")

        print()
