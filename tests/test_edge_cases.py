"""Edge cases: empty text, single line, huge sections, unicode, etc."""

from chunkweaver import Chunk, Chunker


class TestEmptyAndMinimal:
    def test_empty_string(self):
        chunker = Chunker()
        assert chunker.chunk("") == []

    def test_whitespace_only(self):
        chunker = Chunker()
        assert chunker.chunk("   \n\n  \t  ") == []

    def test_single_character(self):
        chunker = Chunker(target_size=10, min_size=0)
        chunks = chunker.chunk("A")
        assert chunks == ["A"]

    def test_single_word(self):
        chunker = Chunker(target_size=100, min_size=0)
        assert chunker.chunk("hello") == ["hello"]

    def test_single_line(self):
        chunker = Chunker(target_size=1000, min_size=0)
        assert chunker.chunk("A single line of text.") == ["A single line of text."]


class TestLargeInputs:
    def test_very_long_single_paragraph(self):
        text = "word " * 10000
        chunker = Chunker(target_size=500, overlap=0, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_many_short_sections(self):
        sections = [f"# Section {i}\nContent {i}.\n" for i in range(100)]
        text = "".join(sections)
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 100

    def test_single_huge_section(self):
        text = "Article 1\n" + "Important legal text. " * 500
        chunker = Chunker(
            target_size=200,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        assert chunks[0].startswith("Article 1")


class TestUnicode:
    def test_unicode_text(self):
        text = "Ärtikel 1\nDiese Verordnung enthält Vorschriften.\nÄrtikel 2\nSie gilt für."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Ärtikel\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_cjk_text(self):
        text = "第一条 本法的目的是保护自然人。\n第二条 本法适用于数据处理。"
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^第[一二三四五六七八九十]+条"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_emoji_in_text(self):
        text = "# Welcome 👋\nHello world!\n# Goodbye 🎉\nSee you!"
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


class TestSpecialPatterns:
    def test_boundary_on_every_line(self):
        text = "\n".join(f"Article {i}" for i in range(10))
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 10

    def test_no_boundary_matches(self):
        text = "Just plain text without any structural markers.\n\nAnother paragraph."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_table_boundary(self):
        text = (
            "Article 1\nSome text.\n"
            "TABLE 1\nHeader1 | Header2\nVal1 | Val2\nVal3 | Val4\n"
            "Article 2\nMore text."
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Article\s+\d+", r"^TABLE\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 3
        table_chunk = [c for c in chunks if "TABLE 1" in c][0]
        assert "Header1" in table_chunk
        assert "Val1" in table_chunk


class TestChunkMetadata:
    def test_offsets_are_valid(self):
        text = "Hello. World. Testing. One. Two. Three." * 5
        chunker = Chunker(target_size=50, overlap=0, min_size=0)
        chunks = chunker.chunk_with_metadata(text)
        for c in chunks:
            assert c.start >= 0
            assert c.end <= len(text)
            assert c.start < c.end

    def test_indices_are_sequential(self):
        text = "# A\nText.\n# B\nMore.\n# C\nEnd."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(text)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_chunk_len(self):
        chunk = Chunk(text="hello world", start=0, end=11)
        assert len(chunk) == 11

    def test_content_text_without_overlap(self):
        chunk = Chunk(text="hello world", start=0, end=11, overlap_text="")
        assert chunk.content_text == "hello world"

    def test_content_text_with_overlap(self):
        chunk = Chunk(text="overlap. actual content", start=9, end=23, overlap_text="overlap. ")
        assert chunk.content_text == "actual content"


class TestNewlines:
    def test_windows_newlines(self):
        text = "Article 1\r\nContent here.\r\nArticle 2\r\nMore content."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Article\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_trailing_newlines(self):
        text = "Some text.\n\n\n\n"
        chunker = Chunker(target_size=5000, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
