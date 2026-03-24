"""Core chunking engine — structure-aware text splitting for RAG."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Pattern, Sequence, Tuple, Union

from chunkweaver.boundaries import BoundaryMatch, detect_boundaries, split_at_boundaries
from chunkweaver.models import Chunk
from chunkweaver.sentences import last_n_sentences, split_sentences

_PARAGRAPH_SEP = re.compile(r"\n\s*\n")


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
        boundaries: Regex patterns that mark section starts. Each line of
                    input is tested against these; first match wins.
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
    """

    def __init__(
        self,
        target_size: int = 1024,
        overlap: int = 2,
        overlap_unit: str = "sentence",
        boundaries: Optional[Sequence[str]] = None,
        fallback: str = "paragraph",
        min_size: int = 200,
        sentence_pattern: Union[str, Pattern[str], None] = None,
        keep_together: Optional[Sequence[str]] = None,
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
        self.boundaries: List[str] = list(boundaries) if boundaries else []
        self.fallback = fallback
        self.min_size = min_size

        if isinstance(sentence_pattern, str):
            self._sentence_re: Union[Pattern[str], None] = re.compile(sentence_pattern)
        else:
            self._sentence_re = sentence_pattern

        self._keep_together: List[re.Pattern[str]] = []
        if keep_together:
            self._keep_together = [re.compile(p) for p in keep_together]

    def chunk(self, text: str) -> List[str]:
        """Split *text* into chunks, returning a list of strings."""
        return [c.text for c in self.chunk_with_metadata(text)]

    def chunk_with_metadata(self, text: str) -> List[Chunk]:
        """Split *text* into chunks with full metadata."""
        if not text or not text.strip():
            return []

        raw_segments = self._create_segments(text)
        merged = self._merge_small_segments(raw_segments, text)
        merged = self._apply_keep_together(merged, text)
        subsplit = self._subsplit_large_segments(merged)
        chunks = self._add_overlap(subsplit)
        return chunks

    # ------------------------------------------------------------------
    # Step 1 & 2: detect boundaries and split
    # ------------------------------------------------------------------

    def _create_segments(self, text: str) -> List[Tuple[str, str, int]]:
        """Return ``(segment_text, boundary_type, start_offset)`` triples."""
        boundary_matches = detect_boundaries(text, self.boundaries)
        raw_pairs = split_at_boundaries(text, boundary_matches)

        segments: List[Tuple[str, str, int]] = []
        offset = 0
        for seg_text, seg_type in raw_pairs:
            pos = text.index(seg_text, offset)
            segments.append((seg_text, seg_type, pos))
            offset = pos + len(seg_text)
        return segments

    # ------------------------------------------------------------------
    # Step 3: merge undersized segments
    # ------------------------------------------------------------------

    def _merge_small_segments(
        self,
        segments: List[Tuple[str, str, int]],
        text: str,
    ) -> List[Tuple[str, str, int]]:
        """Merge segments shorter than *min_size* with their successor."""
        if not segments or self.min_size <= 0:
            return segments

        merged: List[Tuple[str, str, int]] = []
        i = 0
        while i < len(segments):
            seg_text, seg_type, seg_start = segments[i]

            while (
                len(seg_text.strip()) < self.min_size
                and i + 1 < len(segments)
                and segments[i + 1][1] != "section"
            ):
                i += 1
                next_text, _, _ = segments[i]
                seg_text = text[seg_start : segments[i][2] + len(next_text)]

            merged.append((seg_text, seg_type, seg_start))
            i += 1
        return merged

    # ------------------------------------------------------------------
    # Step 3b: keep_together — glue label lines to their content
    # ------------------------------------------------------------------

    def _apply_keep_together(
        self,
        segments: List[Tuple[str, str, int]],
        text: str,
    ) -> List[Tuple[str, str, int]]:
        """Merge segments whose first line matches a keep_together pattern
        with the following segment, as long as the result fits target_size."""
        if not self._keep_together or not segments:
            return segments

        merged: List[Tuple[str, str, int]] = []
        i = 0
        while i < len(segments):
            seg_text, seg_type, seg_start = segments[i]
            first_line = seg_text.split("\n", 1)[0]

            if (
                any(p.search(first_line) for p in self._keep_together)
                and i + 1 < len(segments)
            ):
                next_text, _, _ = segments[i + 1]
                combined = text[seg_start : segments[i + 1][2] + len(next_text)]
                if len(combined) <= self.target_size:
                    merged.append((combined, seg_type, seg_start))
                    i += 2
                    continue

            merged.append((seg_text, seg_type, seg_start))
            i += 1
        return merged

    # ------------------------------------------------------------------
    # Step 4: sub-split oversized segments
    # ------------------------------------------------------------------

    def _subsplit_large_segments(
        self, segments: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int]]:
        """Break segments exceeding *target_size* using the fallback strategy."""
        result: List[Tuple[str, str, int]] = []
        for seg_text, seg_type, seg_start in segments:
            if len(seg_text) <= self.target_size:
                result.append((seg_text, seg_type, seg_start))
                continue

            sub_parts = self._split_by_fallback(seg_text)
            accumulated = ""
            acc_start = seg_start

            for part in sub_parts:
                if accumulated and len(accumulated) + len(part) > self.target_size:
                    result.append((accumulated, self.fallback, acc_start))
                    acc_start = acc_start + len(accumulated)
                    accumulated = part
                else:
                    accumulated += part

            if accumulated:
                if result and len(accumulated.strip()) < self.min_size:
                    prev_text, prev_type, prev_start = result[-1]
                    result[-1] = (prev_text + accumulated, prev_type, prev_start)
                else:
                    result.append((accumulated, self.fallback, acc_start))

        return result

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
        return parts or [text]

    # ------------------------------------------------------------------
    # Step 5: add overlap
    # ------------------------------------------------------------------

    def _add_overlap(
        self, segments: List[Tuple[str, str, int]]
    ) -> List[Chunk]:
        """Build final Chunk objects, prepending overlap from the previous chunk."""
        if not segments:
            return []

        chunks: List[Chunk] = []
        for idx, (seg_text, seg_type, seg_start) in enumerate(segments):
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
