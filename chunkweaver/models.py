from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
                       "sentence", "word", or "size".
        overlap_text: The overlap portion prepended from the previous chunk.
                      Empty string when there is no overlap.
    """

    text: str
    start: int
    end: int
    index: int = 0
    boundary_type: str = "section"
    overlap_text: str = ""

    @property
    def content_text(self) -> str:
        """Chunk text without the overlap prefix."""
        if self.overlap_text:
            return self.text[len(self.overlap_text):]
        return self.text

    def __len__(self) -> int:
        return len(self.text)
