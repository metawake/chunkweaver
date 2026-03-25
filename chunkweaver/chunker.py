"""Core chunking engine — structure-aware text splitting for RAG."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Pattern, Sequence, Tuple, Union

from chunkweaver.boundaries import (
    BoundaryMatch,
    BoundarySpec,
    detect_boundaries,
    split_at_boundaries,
)
from chunkweaver.detectors import (
    Annotation,
    BoundaryDetector,
    KeepTogetherRegion,
    SplitPoint,
)
from chunkweaver.models import Chunk
from chunkweaver.sentences import last_n_sentences, split_sentences

_PARAGRAPH_SEP = re.compile(r"\n\s*\n")

# Internal segment: (text, boundary_type, start_offset, hierarchy_level)
_Seg = Tuple[str, str, int, int]


class Chunker:
    """Structure-aware text chunker.

    Splits text at structural boundaries you define, falls back to
    paragraphs/sentences when sections are too large, and overlaps in
    semantic units (sentences) instead of characters.

    Args:
        target_size: Target chunk size in characters.
        overlap: Number of overlap units to prepend from the previous chunk.
        overlap_unit: Unit for overlap — ``"sentence"``, ``"paragraph"``,
                      or ``"chars"``.
        boundaries: Boundary patterns that mark section starts.  Each entry
                    is either a regex string (level 0) or a ``(regex, level)``
                    tuple for hierarchical splitting.  Level 0 boundaries
                    always split; higher levels only split when the parent
                    segment exceeds ``target_size``.
        fallback: How to sub-split oversized segments — ``"paragraph"``,
                  ``"sentence"``, or ``"word"``.
        min_size: Minimum chunk size in characters. Segments shorter than
                  this are merged with the next segment.
        sentence_pattern: Custom regex for sentence boundary detection.
                          The default works for English prose. Override for
                          CJK text, chat logs, or other formats. Can be a
                          string or a compiled ``re.Pattern``.
        keep_together: Regex patterns for lines that must stay attached to
                       the *next* line's content. Useful for table headers,
                       field labels, or any line that is meaningless alone.
                       When a segment starts with a matching line and is
                       below ``target_size``, it won't be split away.
        detectors: Heuristic ``BoundaryDetector`` instances. Their
                   ``SplitPoint`` annotations are merged with regex
                   boundaries; their ``KeepTogetherRegion`` annotations
                   prevent splitting inside protected ranges (allowing
                   overshoot up to each region's ``max_overshoot`` ratio).
        annotations: Pre-computed ``SplitPoint`` and ``KeepTogetherRegion``
                     annotations from an upstream extractor or external
                     tool.  Merged with detector output.
        concurrent: When ``True`` and multiple detectors are provided,
                    run them in parallel via ``ThreadPoolExecutor``.
                    Useful when one or more detectors make remote API
                    calls or run expensive ML inference. Has no effect
                    with zero or one detector.
    """

    def __init__(
        self,
        target_size: int = 1024,
        overlap: int = 2,
        overlap_unit: str = "sentence",
        boundaries: Optional[Sequence[BoundarySpec]] = None,
        fallback: str = "paragraph",
        min_size: int = 200,
        sentence_pattern: Union[str, Pattern[str], None] = None,
        keep_together: Optional[Sequence[str]] = None,
        detectors: Optional[Sequence[BoundaryDetector]] = None,
        annotations: Optional[Sequence[Annotation]] = None,
        concurrent: bool = False,
    ) -> None:
        if target_size < 1:
            raise ValueError("target_size must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap_unit not in ("sentence", "paragraph", "chars"):
            raise ValueError(
                f"overlap_unit must be 'sentence', 'paragraph', or 'chars', "
                f"got {overlap_unit!r}"
            )
        if fallback not in ("paragraph", "sentence", "word"):
            raise ValueError(
                f"fallback must be 'paragraph', 'sentence', or 'word', "
                f"got {fallback!r}"
            )
        if min_size < 0:
            raise ValueError("min_size must be >= 0")

        self.target_size = target_size
        self.overlap = overlap
        self.overlap_unit = overlap_unit
        self.boundaries: List[BoundarySpec] = list(boundaries) if boundaries else []
        self.fallback = fallback
        self.min_size = min_size

        if isinstance(sentence_pattern, str):
            self._sentence_re: Union[Pattern[str], None] = re.compile(sentence_pattern)
        else:
            self._sentence_re = sentence_pattern

        self._keep_together: List[re.Pattern[str]] = []
        if keep_together:
            self._keep_together = [re.compile(p) for p in keep_together]

        self._detectors: List[BoundaryDetector] = list(detectors) if detectors else []
        self._annotations: List[Annotation] = list(annotations) if annotations else []
        self._concurrent = concurrent

    def chunk(self, text: str) -> List[str]:
        """Split *text* into chunks, returning a list of strings."""
        return [c.text for c in self.chunk_with_metadata(text)]

    def chunk_with_metadata(self, text: str) -> List[Chunk]:
        """Split *text* into chunks with full metadata."""
        if not text or not text.strip():
            return []

        extra_boundaries, keep_regions = self._run_detectors(text)

        if self._annotations:
            self._partition_annotations(
                list(self._annotations), extra_boundaries, keep_regions,
            )
            extra_boundaries.sort(key=lambda b: b.position)
            keep_regions.sort(key=lambda r: r.start)

        raw_segments = self._create_segments(text, extra_boundaries, keep_regions)
        merged = self._merge_small_segments(raw_segments, text)
        merged = self._apply_keep_together(merged, text)
        subsplit = self._subsplit_large_segments(merged, keep_regions)
        chunks = self._add_overlap(subsplit)
        return chunks

    # ------------------------------------------------------------------
    # Detector integration
    # ------------------------------------------------------------------

    def _run_detectors(
        self, text: str
    ) -> Tuple[List[BoundaryMatch], List[KeepTogetherRegion]]:
        """Run all detectors and partition results into split points
        and keep-together regions.

        When ``concurrent=True`` and there are 2+ detectors, calls are
        fanned out via ``ThreadPoolExecutor`` so a slow remote detector
        doesn't block fast local ones.
        """
        extra_boundaries: List[BoundaryMatch] = []
        keep_regions: List[KeepTogetherRegion] = []

        if not self._detectors:
            return extra_boundaries, keep_regions

        if self._concurrent and len(self._detectors) >= 2:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(self._detectors)) as pool:
                futures = {
                    pool.submit(d.detect, text): d
                    for d in self._detectors
                }
                for future in as_completed(futures):
                    result_annotations = future.result()
                    self._partition_annotations(
                        result_annotations, extra_boundaries, keep_regions,
                    )
        else:
            for detector in self._detectors:
                self._partition_annotations(
                    detector.detect(text), extra_boundaries, keep_regions,
                )

        extra_boundaries.sort(key=lambda b: b.position)
        keep_regions.sort(key=lambda r: r.start)
        return extra_boundaries, keep_regions

    @staticmethod
    def _partition_annotations(
        annotations: List[Annotation],
        extra_boundaries: List[BoundaryMatch],
        keep_regions: List[KeepTogetherRegion],
    ) -> None:
        """Sort annotations into split points and keep-together regions."""
        for annotation in annotations:
            if isinstance(annotation, SplitPoint):
                extra_boundaries.append(BoundaryMatch(
                    position=annotation.position,
                    line_number=annotation.line_number,
                    pattern=f"[detector:{annotation.label}]",
                    matched_text=annotation.label,
                    level=annotation.level,
                ))
            elif isinstance(annotation, KeepTogetherRegion):
                keep_regions.append(annotation)

    # ------------------------------------------------------------------
    # Step 1 & 2: detect boundaries and split
    # ------------------------------------------------------------------

    def _create_segments(
        self,
        text: str,
        extra_boundaries: Optional[List[BoundaryMatch]] = None,
        keep_regions: Optional[List[KeepTogetherRegion]] = None,
    ) -> List[_Seg]:
        """Return ``(segment_text, boundary_type, start_offset, level)`` tuples.

        When all boundaries share the same level, every boundary
        produces a split (flat mode, backward compatible).  When
        boundaries have mixed levels, **hierarchical mode** kicks in:
        level-0 boundaries always split; level-N boundaries only
        split when the parent segment exceeds ``target_size``.
        """
        boundary_matches = detect_boundaries(text, self.boundaries)

        if extra_boundaries:
            seen = {b.position for b in boundary_matches}
            for eb in extra_boundaries:
                if eb.position not in seen:
                    boundary_matches.append(eb)
                    seen.add(eb.position)
            boundary_matches.sort(key=lambda b: b.position)

        if keep_regions:
            boundary_matches = [
                b for b in boundary_matches
                if not any(r.start < b.position < r.end for r in keep_regions)
            ]

        if not boundary_matches:
            return [(text, "start", 0, 0)]

        max_level = max(b.level for b in boundary_matches)

        if max_level == 0:
            return self._split_flat(text, boundary_matches)

        return self._split_hierarchical(text, boundary_matches, max_level)

    def _split_flat(
        self, text: str, boundaries: List[BoundaryMatch]
    ) -> List[_Seg]:
        """Split at every boundary (all same level)."""
        segments: List[_Seg] = []

        if boundaries[0].position > 0:
            segments.append((text[: boundaries[0].position], "start", 0, 0))

        for i, b in enumerate(boundaries):
            end = boundaries[i + 1].position if i + 1 < len(boundaries) else len(text)
            seg_text = text[b.position : end]
            if seg_text:
                segments.append((seg_text, "section", b.position, b.level))

        return segments

    def _split_hierarchical(
        self,
        text: str,
        boundaries: List[BoundaryMatch],
        max_level: int,
    ) -> List[_Seg]:
        """Recursive descent: split at level 0, refine oversized at level 1, etc."""
        level_0 = [b for b in boundaries if b.level == 0]

        if level_0:
            segments = self._split_flat(text, level_0)
        else:
            segments = [(text, "start", 0, 0)]

        for level in range(1, max_level + 1):
            level_bounds = [b for b in boundaries if b.level == level]
            if not level_bounds:
                continue
            segments = self._refine_at_level(text, segments, level_bounds, level)

        return segments

    def _refine_at_level(
        self,
        text: str,
        segments: List[_Seg],
        boundaries: List[BoundaryMatch],
        level: int,
    ) -> List[_Seg]:
        """Split oversized segments at boundaries of the given level."""
        result: List[_Seg] = []

        for seg_text, seg_type, seg_start, seg_level in segments:
            if len(seg_text) <= self.target_size:
                result.append((seg_text, seg_type, seg_start, seg_level))
                continue

            seg_end = seg_start + len(seg_text)
            internal = [b for b in boundaries if seg_start <= b.position < seg_end]

            if not internal:
                result.append((seg_text, seg_type, seg_start, seg_level))
                continue

            if internal[0].position > seg_start:
                before = text[seg_start : internal[0].position]
                if before:
                    result.append((before, seg_type, seg_start, seg_level))

            for i, b in enumerate(internal):
                end = internal[i + 1].position if i + 1 < len(internal) else seg_end
                sub_text = text[b.position : end]
                if sub_text:
                    result.append((sub_text, "section", b.position, level))

        return result

    # ------------------------------------------------------------------
    # Step 3: merge undersized segments
    # ------------------------------------------------------------------

    def _merge_small_segments(
        self,
        segments: List[_Seg],
        text: str,
    ) -> List[_Seg]:
        """Merge segments shorter than *min_size* with their successor.

        In hierarchical mode, a segment at a higher-priority level
        (lower number) is allowed to merge into a deeper-level
        neighbor — e.g. a chapter heading merges into the first article.
        """
        if not segments or self.min_size <= 0:
            return segments

        merged: List[_Seg] = []
        i = 0
        while i < len(segments):
            seg_text, seg_type, seg_start, seg_level = segments[i]

            while (
                len(seg_text.strip()) < self.min_size
                and i + 1 < len(segments)
                and (
                    segments[i + 1][1] != "section"
                    or seg_level < segments[i + 1][3]
                )
            ):
                i += 1
                next_text = segments[i][0]
                seg_text = text[seg_start : segments[i][2] + len(next_text)]

            merged.append((seg_text, seg_type, seg_start, seg_level))
            i += 1
        return merged

    # ------------------------------------------------------------------
    # Step 3b: keep_together — glue label lines to their content
    # ------------------------------------------------------------------

    def _apply_keep_together(
        self,
        segments: List[_Seg],
        text: str,
    ) -> List[_Seg]:
        """Merge segments whose first line matches a keep_together pattern
        with the following segment, as long as the result fits target_size."""
        if not self._keep_together or not segments:
            return segments

        merged: List[_Seg] = []
        i = 0
        while i < len(segments):
            seg_text, seg_type, seg_start, seg_level = segments[i]
            first_line = seg_text.split("\n", 1)[0]

            if (
                any(p.search(first_line) for p in self._keep_together)
                and i + 1 < len(segments)
            ):
                next_text = segments[i + 1][0]
                combined = text[seg_start : segments[i + 1][2] + len(next_text)]
                if len(combined) <= self.target_size:
                    merged.append((combined, seg_type, seg_start, seg_level))
                    i += 2
                    continue

            merged.append((seg_text, seg_type, seg_start, seg_level))
            i += 1
        return merged

    # ------------------------------------------------------------------
    # Step 4: sub-split oversized segments
    # ------------------------------------------------------------------

    def _subsplit_large_segments(
        self,
        segments: List[_Seg],
        keep_regions: Optional[List[KeepTogetherRegion]] = None,
    ) -> List[_Seg]:
        """Break segments exceeding *target_size* using the fallback strategy.

        When *keep_regions* are provided, the method first isolates
        protected regions into their own segments.  Isolated regions
        that fit within ``max_overshoot * target_size`` are emitted
        whole; larger ones fall back to normal splitting.
        """
        regions = keep_regions or []

        if regions:
            segments = self._isolate_keep_regions(segments, regions)

        result: List[_Seg] = []

        for seg_text, seg_type, seg_start, seg_level in segments:
            if len(seg_text) <= self.target_size:
                result.append((seg_text, seg_type, seg_start, seg_level))
                continue

            # Keep-together segments get overshoot allowance
            if seg_type == "keep_together":
                region = self._find_covering_region(
                    seg_start, seg_start + len(seg_text), regions
                )
                if region:
                    limit = int(self.target_size * region.max_overshoot)
                    if len(seg_text) <= limit:
                        result.append((seg_text, "section", seg_start, seg_level))
                        continue

            sub_parts = self._split_by_fallback(seg_text)
            accumulated = ""
            acc_start = seg_start

            for part in sub_parts:
                if accumulated and len(accumulated) + len(part) > self.target_size:
                    result.append((accumulated, self.fallback, acc_start, seg_level))
                    acc_start = acc_start + len(accumulated)
                    accumulated = part
                else:
                    accumulated += part

            if accumulated:
                if result and len(accumulated.strip()) < self.min_size:
                    prev_text, prev_type, prev_start, prev_level = result[-1]
                    result[-1] = (prev_text + accumulated, prev_type, prev_start, prev_level)
                else:
                    result.append((accumulated, self.fallback, acc_start, seg_level))

        return result

    def _isolate_keep_regions(
        self,
        segments: List[_Seg],
        regions: List[KeepTogetherRegion],
    ) -> List[_Seg]:
        """Carve keep-together regions out of segments so they become
        their own segment entries, preventing the subsplit loop from
        splitting inside them."""
        result: List[_Seg] = []

        for seg_text, seg_type, seg_start, seg_level in segments:
            seg_end = seg_start + len(seg_text)

            overlapping = sorted(
                (r for r in regions if r.start < seg_end and r.end > seg_start),
                key=lambda r: r.start,
            )

            if not overlapping:
                result.append((seg_text, seg_type, seg_start, seg_level))
                continue

            pos = seg_start
            for region in overlapping:
                r_start = max(region.start, seg_start)
                r_end = min(region.end, seg_end)

                if r_start > pos:
                    before = seg_text[pos - seg_start : r_start - seg_start]
                    if before:
                        result.append((before, seg_type, pos, seg_level))

                region_text = seg_text[r_start - seg_start : r_end - seg_start]
                if region_text:
                    result.append((region_text, "keep_together", r_start, seg_level))

                pos = r_end

            if pos < seg_end:
                after = seg_text[pos - seg_start :]
                if after:
                    result.append((after, seg_type, pos, seg_level))

        return result

    @staticmethod
    def _find_covering_region(
        start: int,
        end: int,
        regions: List[KeepTogetherRegion],
    ) -> Optional[KeepTogetherRegion]:
        """Return the keep-together region that covers ``[start, end)``."""
        for r in regions:
            if r.start <= start and r.end >= end:
                return r
        for r in regions:
            if r.start < end and r.end > start:
                return r
        return None

    def _split_by_fallback(self, text: str) -> List[str]:
        """Split text using the configured fallback hierarchy."""
        if self.fallback == "paragraph":
            parts = self._split_paragraphs(text)
            refined: List[str] = []
            for p in parts:
                if len(p) > self.target_size:
                    refined.extend(self._split_to_sentences(p))
                else:
                    refined.append(p)
            final: List[str] = []
            for p in refined:
                if len(p) > self.target_size:
                    final.extend(self._split_at_words(p))
                else:
                    final.append(p)
            return final

        if self.fallback == "sentence":
            parts = self._split_to_sentences(text)
            final = []
            for p in parts:
                if len(p) > self.target_size:
                    final.extend(self._split_at_words(p))
                else:
                    final.append(p)
            return final

        return self._split_at_words(text)

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split on double-newline boundaries, preserving delimiters."""
        parts: List[str] = []
        last = 0
        for m in _PARAGRAPH_SEP.finditer(text):
            parts.append(text[last : m.end()])
            last = m.end()
        if last < len(text):
            parts.append(text[last:])
        return parts or [text]

    def _split_to_sentences(self, text: str) -> List[str]:
        parts = split_sentences(text, pattern=self._sentence_re)
        return parts or [text]

    @staticmethod
    def _split_at_words(text: str) -> List[str]:
        """Split at whitespace boundaries, yielding pieces that include the
        whitespace delimiter so they can be reassembled losslessly."""
        parts: List[str] = []
        for m in re.finditer(r"\S+\s*", text):
            parts.append(m.group())
        if not parts:
            return [text]
        # \S+\s* skips any leading whitespace — reattach it to the first word
        # so "".join(parts) == text.
        first_non_ws = len(text) - len(text.lstrip())
        if first_non_ws > 0:
            parts[0] = text[:first_non_ws] + parts[0]
        return parts

    # ------------------------------------------------------------------
    # Step 5: add overlap
    # ------------------------------------------------------------------

    def _add_overlap(
        self, segments: List[_Seg]
    ) -> List[Chunk]:
        """Build final Chunk objects, prepending overlap from the previous chunk."""
        if not segments:
            return []

        chunks: List[Chunk] = []
        for idx, (seg_text, seg_type, seg_start, seg_level) in enumerate(segments):
            overlap_text = ""
            if idx > 0 and self.overlap > 0:
                prev_text = segments[idx - 1][0]
                overlap_text = self._compute_overlap(prev_text)

            full_text = overlap_text + seg_text if overlap_text else seg_text
            chunks.append(Chunk(
                text=full_text,
                start=seg_start,
                end=seg_start + len(seg_text),
                index=idx,
                boundary_type=seg_type if seg_type != "start" else "section",
                overlap_text=overlap_text,
                boundary_level=seg_level,
            ))

        return chunks

    def _compute_overlap(self, previous_text: str) -> str:
        """Extract the overlap portion from the end of *previous_text*."""
        if self.overlap <= 0:
            return ""

        if self.overlap_unit == "sentence":
            return last_n_sentences(
                previous_text, self.overlap, pattern=self._sentence_re
            )

        if self.overlap_unit == "paragraph":
            paras = [p for p in _PARAGRAPH_SEP.split(previous_text) if p.strip()]
            if not paras:
                return ""
            taken = paras[-self.overlap :]
            return "\n\n".join(taken)

        # chars
        return previous_text[-self.overlap :]
