"""Boundary detection — scan lines for structural markers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

BoundarySpec = Union[str, tuple[str, int]]
"""A boundary pattern: either a plain regex string (level 0) or a
``(regex, level)`` tuple for hierarchical splitting."""


@dataclass(frozen=True)
class BoundaryMatch:
    """A detected structural boundary in the source text."""

    position: int  # character offset of the line start
    line_number: int  # 0-based line index
    pattern: str  # the regex pattern that matched
    matched_text: str  # the actual text that matched
    level: int = 0  # hierarchy level (0 = strongest boundary)


def detect_boundaries(
    text: str,
    patterns: Sequence[BoundarySpec],
) -> list[BoundaryMatch]:
    """Find all lines in *text* that match any boundary *pattern*.

    Each pattern is either a regex string (treated as level 0) or a
    ``(regex, level)`` tuple for hierarchical boundary detection.
    Patterns are tested in order; first match wins for each line.
    Returns matches sorted by position.
    """
    if not patterns or not text:
        return []

    compiled: list[tuple[str, re.Pattern[str], int]] = []
    for p in patterns:
        if isinstance(p, tuple):
            pat_str, level = p
        else:
            pat_str, level = p, 0
        compiled.append((pat_str, re.compile(pat_str, re.MULTILINE), level))

    matches: list[BoundaryMatch] = []
    seen_positions: set = set()

    offset = 0
    for line_no, line in enumerate(text.split("\n")):
        for pat_str, pat_re, level in compiled:
            m = pat_re.search(line)
            if m and offset not in seen_positions:
                matches.append(
                    BoundaryMatch(
                        position=offset,
                        line_number=line_no,
                        pattern=pat_str,
                        matched_text=m.group(),
                        level=level,
                    )
                )
                seen_positions.add(offset)
                break  # first match wins
        offset += len(line) + 1  # +1 for the newline

    matches.sort(key=lambda b: b.position)
    return matches


def split_at_boundaries(
    text: str,
    boundaries: list[BoundaryMatch],
) -> list[tuple[str, str]]:
    """Split *text* into segments at boundary positions.

    Returns a list of ``(segment_text, boundary_type)`` tuples.
    The boundary_type is ``"section"`` for boundary-delimited segments
    and ``"start"`` for any text before the first boundary.
    """
    if not boundaries:
        return [(text, "start")] if text else []

    segments: list[tuple[str, str]] = []

    if boundaries[0].position > 0:
        segments.append((text[: boundaries[0].position], "start"))

    for i, b in enumerate(boundaries):
        end = boundaries[i + 1].position if i + 1 < len(boundaries) else len(text)
        segment = text[b.position : end]
        if segment:
            segments.append((segment, "section"))

    return segments
