"""Heuristic table detection for plain text documents.

Identifies runs of numeric data lines (financial tables), extends backward
to include headers, and forward to include footnotes. Emits
``KeepTogetherRegion`` annotations so the chunker avoids splitting tables.

Usage::

    from chunkweaver import Chunker
    from chunkweaver.detector_table import TableDetector

    chunker = Chunker(
        target_size=1024,
        detectors=[TableDetector()],
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chunkweaver.detectors import Annotation, BoundaryDetector, KeepTogetherRegion

# ---------------------------------------------------------------------------
# Patterns for numeric content
# ---------------------------------------------------------------------------

_MULTI_NUM_RE = re.compile(r"[\d,]+\.?\d*")
_YEAR_LINE_RE = re.compile(r"^\s*(20\d{2}\s+){2,}")
_FOOTNOTE_RE = re.compile(r"^\s*\([a-z]\)")
_UNITS_RE = re.compile(r"\(in (millions|billions|thousands)\)", re.IGNORECASE)


@dataclass(frozen=True)
class TableRegion:
    """Internal result — a detected table region with metadata."""

    start_line: int
    end_line: int
    start_char: int
    end_char: int
    header_text: str
    num_data_lines: int


def _is_numeric_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return False
    nums = _MULTI_NUM_RE.findall(stripped)
    if len(nums) < 2:
        return False
    digit_chars = sum(1 for c in stripped if c.isdigit())
    return digit_chars / len(stripped) > 0.15


def _is_table_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_YEAR_LINE_RE.match(stripped) or _UNITS_RE.search(stripped))


class TableDetector(BoundaryDetector):
    """Detect financial table regions as keep-together zones.

    Args:
        min_data_lines: Minimum numeric lines to qualify as a table.
        header_lookback: How many lines to look back for table headers.
        max_gap: Maximum blank lines allowed within a table run.
        max_overshoot: Overshoot ratio passed to ``KeepTogetherRegion``.
    """

    def __init__(
        self,
        min_data_lines: int = 3,
        header_lookback: int = 4,
        max_gap: int = 2,
        max_overshoot: float = 1.5,
    ) -> None:
        self.min_data_lines = min_data_lines
        self.header_lookback = header_lookback
        self.max_gap = max_gap
        self.max_overshoot = max_overshoot

    def detect(self, text: str) -> list[Annotation]:
        """Return ``KeepTogetherRegion`` annotations for detected table regions."""
        regions = self._find_regions(text)
        return [
            KeepTogetherRegion(
                start=r.start_char,
                end=r.end_char,
                label=f"table: {r.header_text[:50]}",
                max_overshoot=self.max_overshoot,
            )
            for r in regions
        ]

    def detect_with_metadata(self, text: str) -> list[TableRegion]:
        """Return raw regions with metadata (useful for debugging)."""
        return self._find_regions(text)

    def _find_regions(self, text: str) -> list[TableRegion]:
        lines = text.split("\n")
        n = len(lines)
        stripped = [ln.strip() for ln in lines]
        is_num = [_is_numeric_line(ln) for ln in lines]

        # Phase 1: find runs of numeric lines
        runs: list[tuple[int, int]] = []
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
                        if gap > self.max_gap:
                            break
                    else:
                        if len(stripped[j]) < 60 and gap <= 1:
                            run_end = j + 1
                            gap = 0
                        else:
                            break
                    j += 1
                data_count = sum(1 for k in range(run_start, run_end) if is_num[k])
                if data_count >= self.min_data_lines:
                    runs.append((run_start, run_end))
                i = run_end
            else:
                i += 1

        # Phase 2: extend each run to include headers and footnotes
        line_offsets = self._compute_line_offsets(lines)
        regions: list[TableRegion] = []

        for run_start, run_end in runs:
            header_start = run_start
            for k in range(run_start - 1, max(run_start - self.header_lookback - 1, -1), -1):
                s = stripped[k]
                if not s:
                    continue
                if _is_table_header(s):
                    header_start = k
                elif len(s) < 60 and not _is_numeric_line(lines[k]):
                    header_start = k
                else:
                    break

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

            start_char = line_offsets[header_start]
            end_char = line_offsets[foot_end] if foot_end < len(line_offsets) else len(text)

            regions.append(
                TableRegion(
                    start_line=header_start,
                    end_line=foot_end,
                    start_char=start_char,
                    end_char=end_char,
                    header_text=header_text[:80],
                    num_data_lines=sum(1 for k in range(run_start, run_end) if is_num[k]),
                )
            )

        # Phase 3: merge overlapping regions
        if regions:
            merged = [regions[0]]
            for r in regions[1:]:
                prev = merged[-1]
                if r.start_line <= prev.end_line + 2:
                    merged[-1] = TableRegion(
                        start_line=prev.start_line,
                        end_line=max(prev.end_line, r.end_line),
                        start_char=prev.start_char,
                        end_char=max(prev.end_char, r.end_char),
                        header_text=prev.header_text,
                        num_data_lines=prev.num_data_lines + r.num_data_lines,
                    )
                else:
                    merged.append(r)
            regions = merged

        return regions

    @staticmethod
    def _compute_line_offsets(lines: list[str]) -> list[int]:
        offsets = [0]
        running = 0
        for line in lines:
            running += len(line) + 1
            offsets.append(running)
        return offsets
