"""Heuristic boundary detection — the abstract layer above regex patterns.

This module provides the ``BoundaryDetector`` ABC and two annotation types
that detectors produce:

- **SplitPoint**: "prefer to start a new chunk here" (e.g. a detected heading)
- **KeepTogetherRegion**: "do not split inside this range" (e.g. a table)

Detectors are composable. Pass a list of detectors to the Chunker; their
annotations are merged together with any regex boundary patterns.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
The existing ``boundaries=[r"^Article \\d+", ...]`` API is unchanged.
Detectors are an *additive* layer — they augment regex patterns, not replace
them. You can use regex-only, detectors-only, or both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union


# ---------------------------------------------------------------------------
# Annotations — what detectors produce
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitPoint:
    """A position where the chunker should prefer to start a new chunk.

    Analogous to a ``BoundaryMatch`` from regex-based detection, but
    produced by heuristic analysis rather than pattern matching.

    Attributes:
        position: Character offset of the line start in the source text.
        line_number: Zero-based line index.
        label: Human-readable description, e.g. ``"heading: Revenue"``.
        level: Hierarchy level (0 = strongest). Higher levels only split
               when the parent segment exceeds ``target_size``.
    """

    position: int
    line_number: int
    label: str = ""
    level: int = 0


@dataclass(frozen=True)
class KeepTogetherRegion:
    """A contiguous region that should not be split across chunks.

    The chunker will try to keep the entire region in a single chunk,
    allowing the chunk to exceed ``target_size`` by up to
    ``max_overshoot`` times. If the region exceeds that threshold, the
    chunker falls back to normal splitting within the region.

    Attributes:
        start: Character offset (inclusive).
        end: Character offset (exclusive).
        label: Human-readable description, e.g. ``"table: Revenue 2023-2025"``.
        max_overshoot: Maximum allowed ratio of ``target_size`` before
            the region is force-split.  Default ``1.5`` means a region
            up to 1536 chars is kept whole when ``target_size=1024``.
    """

    start: int
    end: int
    label: str = ""
    max_overshoot: float = 1.5


Annotation = Union[SplitPoint, KeepTogetherRegion]


# ---------------------------------------------------------------------------
# Abstract detector
# ---------------------------------------------------------------------------

class BoundaryDetector(ABC):
    """Abstract base class for heuristic boundary detection.

    Subclass this to implement custom document structure detection.
    A detector analyzes plain text and returns a list of annotations —
    ``SplitPoint`` and/or ``KeepTogetherRegion`` objects.

    Detectors follow the **Chain of Responsibility** pattern: the
    Chunker runs each detector, collects all annotations, and applies
    them in a single pass. Detectors are independent; they don't need
    to know about each other.

    Example::

        class HeadingDetector(BoundaryDetector):
            def detect(self, text: str) -> list[Annotation]:
                # ... heuristic heading analysis ...
                return [SplitPoint(pos, line_no, "heading: ..."), ...]

        class TableDetector(BoundaryDetector):
            def detect(self, text: str) -> list[Annotation]:
                # ... heuristic table detection ...
                return [KeepTogetherRegion(start, end, "table: ..."), ...]

        chunker = Chunker(
            target_size=1024,
            boundaries=FINANCIAL,           # regex patterns (backward compatible)
            detectors=[HeadingDetector(), TableDetector()],  # heuristic layer
        )
    """

    @abstractmethod
    def detect(self, text: str) -> List[Annotation]:
        """Analyze *text* and return boundary annotations.

        Returns:
            A list of ``SplitPoint`` and/or ``KeepTogetherRegion``
            objects. The list may be empty if no structure is found.
        """
        ...
