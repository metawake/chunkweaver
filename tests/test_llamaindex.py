"""Tests for the LlamaIndex integration (skipped if llama-index-core not installed)."""

import pytest

try:
    from llama_index.core.node_parser import TextSplitter
    from llama_index.core.schema import Document

    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False


@pytest.mark.skipif(not HAS_LLAMAINDEX, reason="llama-index-core not installed")
class TestChunkWeaverNodeParser:
    def test_split_text_returns_strings(self):
        from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser

        parser = ChunkWeaverNodeParser(target_size=500, overlap=0)
        text = "Hello world. " * 50
        result = parser.split_text(text)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_with_boundaries(self):
        from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser

        parser = ChunkWeaverNodeParser(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        text = "# Section 1\nContent.\n# Section 2\nMore."
        result = parser.split_text(text)
        assert len(result) == 2

    def test_is_text_splitter_subclass(self):
        from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser

        assert issubclass(ChunkWeaverNodeParser, TextSplitter)

    def test_get_nodes_from_documents(self):
        from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser

        parser = ChunkWeaverNodeParser(target_size=5000, overlap=0, min_size=0)
        docs = [Document(text="# Title\nParagraph one.\n# Part 2\nParagraph two.")]
        nodes = parser.get_nodes_from_documents(docs)
        assert len(nodes) >= 1
        assert all(hasattr(n, "text") for n in nodes)

    def test_detectors_param_accepted(self):
        from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser
        from chunkweaver.detector_heading import HeadingDetector

        parser = ChunkWeaverNodeParser(
            target_size=1024,
            detectors=[HeadingDetector()],
        )
        text = "INTRODUCTION\n\nSome text.\n\nCONCLUSION\n\nFinal text."
        result = parser.split_text(text)
        assert isinstance(result, list)
