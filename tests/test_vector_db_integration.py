"""Integration-style tests showing chunkweaver used for vector DB workflows.

These tests verify the chunking output is suitable for ingestion into
vector databases — correct overlap, metadata, and deduplication.
"""

from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU, MARKDOWN


class TestVectorDBWorkflow:
    """Simulate a typical ingest pipeline: chunk -> embed -> store."""

    LEGAL_DOC = (
        "CHAPTER I\n"
        "General provisions\n\n"
        "Article 1\n"
        "Subject-matter and objectives\n"
        "This Regulation lays down rules relating to the protection "
        "of natural persons with regard to the processing of personal data "
        "and rules relating to the free movement of personal data.\n\n"
        "Article 2\n"
        "Material scope\n"
        "This Regulation applies to the processing of personal data wholly "
        "or partly by automated means and to the processing other than by "
        "automated means of personal data which form part of a filing system.\n\n"
        "Article 3\n"
        "Territorial scope\n"
        "This Regulation applies to the processing of personal data in the "
        "context of the activities of an establishment of a controller or a "
        "processor in the Union, regardless of whether the processing takes "
        "place in the Union or not.\n"
    )

    def test_chunks_have_required_metadata_for_vectordb(self):
        chunker = Chunker(
            target_size=5000,
            overlap=2,
            overlap_unit="sentence",
            boundaries=LEGAL_EU,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.LEGAL_DOC)

        for c in chunks:
            assert c.text, "Text must not be empty"
            assert isinstance(c.start, int), "Start offset required"
            assert isinstance(c.end, int), "End offset required"
            assert isinstance(c.index, int), "Index required"
            assert c.boundary_type in ("section", "paragraph", "sentence", "word", "size", "start")

    def test_deduplication_via_content_text(self):
        """Vector DB can deduplicate by using content_text (without overlap)."""
        chunker = Chunker(
            target_size=5000,
            overlap=2,
            overlap_unit="sentence",
            boundaries=LEGAL_EU,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.LEGAL_DOC)

        content_texts = [c.content_text for c in chunks]
        assert len(content_texts) == len(set(content_texts)), "Content texts should be unique"

    def test_overlap_improves_context_window(self):
        """Chunks with overlap contain context from the previous chunk."""
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=LEGAL_EU,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.LEGAL_DOC)

        for c in chunks[1:]:
            assert len(c.text) >= len(c.content_text)

    def test_no_content_loss(self):
        """All original text is covered by chunk content (no gaps)."""
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=LEGAL_EU,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.LEGAL_DOC)
        covered = "".join(c.content_text for c in chunks)
        assert covered == self.LEGAL_DOC

    def test_chunk_sizes_suitable_for_embedding(self):
        """Chunks should not be excessively large for embedding models."""
        chunker = Chunker(
            target_size=512,
            overlap=1,
            overlap_unit="sentence",
            boundaries=LEGAL_EU,
            min_size=50,
        )
        chunks = chunker.chunk(self.LEGAL_DOC)
        for c in chunks:
            assert len(c) < 512 * 2, "Chunk too large for typical embedding model"


class TestMarkdownForVectorDB:
    MD_DOC = (
        "# Installation\n"
        "Run `pip install chunkweaver`.\n\n"
        "# Quick Start\n"
        "Import the Chunker class and create an instance. "
        "Configure target_size and boundaries. "
        "Call chunk() on your text.\n\n"
        "# API Reference\n"
        "## Chunker\n"
        "The main class for chunking text.\n\n"
        "## Chunk\n"
        "A dataclass representing a single chunk.\n"
    )

    def test_sections_are_atomic_for_retrieval(self):
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=MARKDOWN,
            min_size=0,
        )
        chunks = chunker.chunk(self.MD_DOC)
        for c in chunks:
            headers = [line for line in c.split("\n") if line.startswith("#")]
            assert len(headers) <= 1, "Each chunk should contain at most one header"

    def test_batch_ingest_format(self):
        """Simulate preparing chunks for batch vector DB insertion."""
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=MARKDOWN,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.MD_DOC)

        records = []
        for c in chunks:
            records.append({
                "id": f"doc-1-chunk-{c.index}",
                "text": c.text,
                "metadata": {
                    "start": c.start,
                    "end": c.end,
                    "boundary_type": c.boundary_type,
                    "has_overlap": bool(c.overlap_text),
                },
            })

        assert len(records) > 0
        assert all("text" in r for r in records)
        assert all(r["metadata"]["start"] >= 0 for r in records)
