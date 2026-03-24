"""Optional LangChain integration — requires ``langchain-text-splitters``."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from chunkweaver.chunker import Chunker

try:
    from langchain_text_splitters import TextSplitter
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-text-splitters. "
        "Install it with: pip install chunkweaver[langchain]"
    )


class ChunkWeaverSplitter(TextSplitter):
    """Drop-in replacement for ``RecursiveCharacterTextSplitter``.

    Uses chunkweaver's boundary-aware algorithm under the hood while
    conforming to the LangChain ``TextSplitter`` interface.

    Example::

        from chunkweaver.integrations.langchain import ChunkWeaverSplitter
        splitter = ChunkWeaverSplitter(target_size=1024, boundaries=[r"^#{1,3}\\s"])
        docs = splitter.create_documents([text])
    """

    def __init__(
        self,
        target_size: int = 1024,
        overlap: int = 2,
        overlap_unit: str = "sentence",
        boundaries: Optional[Sequence[str]] = None,
        fallback: str = "paragraph",
        min_size: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._chunker = Chunker(
            target_size=target_size,
            overlap=overlap,
            overlap_unit=overlap_unit,
            boundaries=list(boundaries) if boundaries else [],
            fallback=fallback,
            min_size=min_size,
        )

    def split_text(self, text: str) -> List[str]:
        return self._chunker.chunk(text)
