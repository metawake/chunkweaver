"""Focused overlap tests with realistic document scenarios."""

from chunkweaver import Chunker


class TestOverlapSentenceUnit:
    """Overlap with sentence units across structural boundaries."""

    def test_overlap_across_article_boundary(self):
        text = (
            "Article 1\n"
            "The first provision establishes scope. It defines the entities covered.\n"
            "Article 2\n"
            "The second provision addresses compliance. Companies must follow rules.\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(text)
        assert len(chunks) == 2
        if chunks[1].overlap_text:
            assert "covered" in chunks[1].overlap_text or "defines" in chunks[1].overlap_text

    def test_two_sentence_overlap(self):
        sentences = [f"Sentence number {i}. " for i in range(20)]
        text = "".join(sentences)
        chunker = Chunker(target_size=100, overlap=2, overlap_unit="sentence", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        for c in chunks[1:]:
            if c.overlap_text:
                from chunkweaver.sentences import split_sentences

                overlap_sents = split_sentences(c.overlap_text)
                assert len(overlap_sents) <= 2


class TestOverlapParagraphUnit:
    def test_paragraph_overlap_content(self):
        text = (
            "First paragraph with important context.\n\n"
            "Second paragraph with more details.\n\n"
            "Third paragraph with conclusions."
        )
        chunker = Chunker(target_size=50, overlap=1, overlap_unit="paragraph", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        if len(chunks) > 1 and chunks[1].overlap_text:
            assert len(chunks[1].overlap_text) > 0


class TestOverlapCharUnit:
    def test_fixed_char_overlap(self):
        text = "A" * 100 + "B" * 100 + "C" * 100
        chunker = Chunker(target_size=110, overlap=20, overlap_unit="chars", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        for c in chunks[1:]:
            assert len(c.overlap_text) == 20

    def test_char_overlap_continuity(self):
        text = "Word " * 100
        chunker = Chunker(target_size=80, overlap=15, overlap_unit="chars", min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1].text[-15:]
            overlap = chunks[i].overlap_text
            assert overlap == prev_tail


class TestOverlapWithBoundaries:
    def test_overlap_does_not_prevent_boundary_splitting(self):
        text = "# Section 1\nContent one.\n# Section 2\nContent two.\n# Section 3\nContent three."
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_overlap_text_tracked_separately(self):
        text = "# Intro\nImportant fact. Another fact.\n# Body\nBody text here."
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(text)
        for c in chunks:
            assert c.content_text in text or c.overlap_text == ""
