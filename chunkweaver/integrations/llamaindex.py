"""Optional LlamaIndex integration — requires ``llama-index-core``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chunkweaver.chunker import Chunker
from chunkweaver.detectors import BoundaryDetector

try:
    from llama_index.core.node_parser import TextSplitter
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install it with: pip install chunkweaver[llamaindex]"
    )


class ChunkWeaverNodeParser(TextSplitter):
    """Structure-aware node parser for LlamaIndex.

    Drop-in replacement for ``SentenceSplitter`` that splits at
    structural boundaries and supports heuristic detectors.

    Example::

        from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser
        from chunkweaver.presets import LEGAL_EU

        parser = ChunkWeaverNodeParser(
            target_size=1024,
            boundaries=LEGAL_EU,
        )
        nodes = parser.get_nodes_from_documents(documents)
    """

    _chunker: Chunker

    def __init__(
        self,
        target_size: int = 1024,
        overlap: int = 2,
        overlap_unit: str = "sentence",
        boundaries: Sequence[str] | None = None,
        fallback: str = "paragraph",
        min_size: int = 200,
        detectors: Sequence[BoundaryDetector] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(
            self,
            "_chunker",
            Chunker(
                target_size=target_size,
                overlap=overlap,
                overlap_unit=overlap_unit,
                boundaries=list(boundaries) if boundaries else [],
                fallback=fallback,
                min_size=min_size,
                detectors=list(detectors) if detectors else [],
            ),
        )

    def split_text(self, text: str) -> list[str]:
        """Split *text* using the chunkweaver engine, conforming to the LlamaIndex contract."""
        return self._chunker.chunk(text)
