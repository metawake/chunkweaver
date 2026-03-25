"""Tests for the core Chunker class."""

import pytest

from chunkweaver import Chunk, Chunker


class TestChunkerBasic:
    def test_simple_chunk(self):
        chunker = Chunker(target_size=500, overlap=0)
        text = "Hello world. " * 10
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_returns_metadata(self):
        chunker = Chunker(target_size=500, overlap=0)
        text = "Hello world. " * 10
        chunks = chunker.chunk_with_metadata(text)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].index == 0
        assert chunks[0].start >= 0

    def test_chunk_text_covers_original(self):
        """Non-overlap content should cover the full original text."""
        text = "Alpha beta gamma. " * 50
        chunker = Chunker(target_size=100, overlap=0)
        chunks = chunker.chunk_with_metadata(text)
        reconstructed = "".join(c.content_text for c in chunks)
        assert reconstructed == text

    def test_respects_target_size(self):
        text = "Word " * 500
        chunker = Chunker(target_size=200, overlap=0, min_size=0)
        chunks = chunker.chunk(text)
        for c in chunks[:-1]:
            assert len(c) <= 200 * 1.5, f"Chunk too large: {len(c)}"


class TestBoundaryChunking:
    GDPR_SAMPLE = (
        "REGULATION (EU) 2016/679\n\n"
        "Article 1\n"
        "Subject-matter and objectives.\n"
        "This Regulation lays down rules relating to the protection "
        "of natural persons with regard to the processing of personal data.\n\n"
        "Article 2\n"
        "Material scope.\n"
        "This Regulation applies to the processing of personal data wholly "
        "or partly by automated means.\n\n"
        "Article 3\n"
        "Territorial scope.\n"
        "This Regulation applies to controllers established in the Union.\n"
    )

    def test_boundary_splits(self):
        chunker = Chunker(
            target_size=2000,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.GDPR_SAMPLE)
        assert len(chunks) == 4  # preamble + 3 articles

    def test_articles_stay_intact(self):
        chunker = Chunker(
            target_size=2000,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(self.GDPR_SAMPLE)
        article_chunks = [c for c in chunks if "Article 1" in c or "Article 2" in c or "Article 3" in c]
        for c in article_chunks:
            assert c.startswith("Article")

    def test_markdown_boundaries(self):
        text = "# Introduction\nSome intro.\n# Methods\nSome methods.\n# Results\nSome results.\n"
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_rfc_numbered_sections(self):
        text = (
            "1. Introduction\nThis document describes...\n"
            "2. Terminology\nKey words...\n"
            "3. Protocol Overview\nThe protocol works by...\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^\d+\.\s+\S"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 3


class TestFallbackSplitting:
    def test_paragraph_fallback(self):
        text = "Paragraph one. " * 30 + "\n\n" + "Paragraph two. " * 30
        chunker = Chunker(target_size=200, overlap=0, fallback="paragraph", min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_sentence_fallback(self):
        text = "Sentence one. " * 100
        chunker = Chunker(target_size=200, overlap=0, fallback="sentence", min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_word_fallback(self):
        text = "word " * 200
        chunker = Chunker(target_size=100, overlap=0, fallback="word", min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_word_split_preserves_leading_whitespace(self):
        text = "Section 1\n" + "\n\n   +--------+---row1---+\n   +--------+---row2---+\n" * 50
        chunker = Chunker(
            target_size=200, overlap=0, fallback="paragraph",
            boundaries=[r"^Section\s+\d+"], min_size=0,
        )
        chunks = chunker.chunk_with_metadata(text)
        reconstructed = "".join(c.content_text for c in chunks)
        assert reconstructed == text

    def test_subsplit_lossless_roundtrip(self):
        text = "   indented start " + "word " * 100 + "\n\n   more indented"
        chunker = Chunker(target_size=100, overlap=0, fallback="word", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        reconstructed = "".join(c.content_text for c in chunks)
        assert reconstructed == text

    def test_oversized_section_gets_subsplit(self):
        long_article = "Article 1\n" + ("Content sentence. " * 200)
        text = long_article + "\nArticle 2\nShort content."
        chunker = Chunker(
            target_size=300,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 2


class TestMergeSmallSegments:
    def test_small_heading_merged_with_body(self):
        text = "Article 1\nThis is a much longer body of text that exceeds the minimum size threshold."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=50,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "Article 1" in chunks[0]

    def test_min_size_zero_keeps_small_chunks(self):
        text = "# A\nText.\n# B\nMore text."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


class TestOverlap:
    def test_sentence_overlap(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = Chunker(target_size=40, overlap=1, overlap_unit="sentence", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        if len(chunks) > 1:
            assert chunks[1].overlap_text != ""

    def test_no_overlap_on_first_chunk(self):
        text = "A. B. C. D. E. " * 20
        chunker = Chunker(target_size=50, overlap=2, overlap_unit="sentence", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        assert chunks[0].overlap_text == ""

    def test_paragraph_overlap(self):
        text = "Para one content.\n\nPara two content.\n\nPara three content."
        chunker = Chunker(target_size=30, overlap=1, overlap_unit="paragraph", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        if len(chunks) > 1:
            assert chunks[1].overlap_text != ""

    def test_char_overlap(self):
        text = "ABCDEF " * 50
        chunker = Chunker(target_size=50, overlap=10, overlap_unit="chars", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        if len(chunks) > 1:
            assert len(chunks[1].overlap_text) == 10

    def test_zero_overlap(self):
        text = "Hello. World. Test." * 10
        chunker = Chunker(target_size=50, overlap=0, min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        for c in chunks:
            assert c.overlap_text == ""

    def test_overlap_text_comes_from_previous(self):
        text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence. Epsilon sentence."
        chunker = Chunker(target_size=40, overlap=1, overlap_unit="sentence", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        if len(chunks) > 1:
            prev_text = chunks[0].text
            overlap = chunks[1].overlap_text
            assert prev_text.rstrip().endswith(overlap.rstrip())


class TestValidation:
    def test_invalid_target_size(self):
        with pytest.raises(ValueError, match="target_size"):
            Chunker(target_size=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            Chunker(overlap=-1)

    def test_invalid_overlap_unit(self):
        with pytest.raises(ValueError, match="overlap_unit"):
            Chunker(overlap_unit="token")

    def test_invalid_fallback(self):
        with pytest.raises(ValueError, match="fallback"):
            Chunker(fallback="character")

    def test_negative_min_size(self):
        with pytest.raises(ValueError, match="min_size"):
            Chunker(min_size=-1)
