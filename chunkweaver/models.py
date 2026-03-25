"""Data models for chunker output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single chunk produced by the Chunker.

    Attributes:
        text: Full chunk content including any overlap prefix.
        start: Character offset of the chunk start in the original text
               (excluding overlap).
        end: Character offset of the chunk end in the original text.
        index: Zero-based chunk index in the output sequence.
        boundary_type: What triggered the split — "section", "paragraph",
                       "sentence", "word", or "keep_together".
        overlap_text: The overlap portion prepended from the previous chunk.
                      Empty string when there is no overlap.
        boundary_level: Hierarchy level of the boundary that started this
                        chunk (0 = strongest boundary).
    """

    text: str
    start: int
    end: int
    index: int = 0
    boundary_type: str = "section"
    overlap_text: str = ""
    boundary_level: int = 0

    @property
    def content_text(self) -> str:
        """Chunk text without the overlap prefix."""
        if self.overlap_text:
            return self.text[len(self.overlap_text) :]
        return self.text

    def __len__(self) -> int:
        return len(self.text)
