"""Tests for the LangChain integration (skipped if langchain not installed)."""

import pytest

try:
    from langchain_text_splitters import TextSplitter

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-text-splitters not installed")
class TestChunkWeaverSplitter:
    def test_split_text_returns_strings(self):
        from chunkweaver.integrations.langchain import ChunkWeaverSplitter

        splitter = ChunkWeaverSplitter(target_size=500, overlap=0)
        text = "Hello world. " * 50
        result = splitter.split_text(text)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_with_boundaries(self):
        from chunkweaver.integrations.langchain import ChunkWeaverSplitter

        splitter = ChunkWeaverSplitter(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        text = "# Section 1\nContent.\n# Section 2\nMore."
        result = splitter.split_text(text)
        assert len(result) == 2

    def test_is_text_splitter_subclass(self):
        from chunkweaver.integrations.langchain import ChunkWeaverSplitter

        assert issubclass(ChunkWeaverSplitter, TextSplitter)
