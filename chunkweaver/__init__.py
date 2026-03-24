"""chunkweaver — structure-aware text chunking for RAG.

Zero-dependency Python chunker that respects document structure.
"""

from chunkweaver.chunker import Chunker
from chunkweaver.detectors import (
    Annotation,
    BoundaryDetector,
    KeepTogetherRegion,
    SplitPoint,
)
from chunkweaver.models import Chunk
from chunkweaver.sentences import SENTENCE_END, SENTENCE_END_CJK, SENTENCE_END_PERMISSIVE

__all__ = [
    "Chunker",
    "Chunk",
    "BoundaryDetector",
    "SplitPoint",
    "KeepTogetherRegion",
    "Annotation",
    "SENTENCE_END",
    "SENTENCE_END_CJK",
    "SENTENCE_END_PERMISSIVE",
]
__version__ = "0.2.1"
