"""Tests for boundary detection logic."""

from chunkweaver.boundaries import detect_boundaries, split_at_boundaries


class TestDetectBoundaries:
    def test_no_patterns(self):
        assert detect_boundaries("hello\nworld", []) == []

    def test_empty_text(self):
        assert detect_boundaries("", [r"^Article"]) == []

    def test_single_pattern(self):
        text = "Preamble\nArticle 1\nContent of article 1.\nArticle 2\nContent."
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        assert len(matches) == 2
        assert matches[0].matched_text == "Article 1"
        assert matches[1].matched_text == "Article 2"

    def test_first_match_wins(self):
        text = "# Title\n## Subtitle\n---"
        matches = detect_boundaries(text, [r"^#{1,6}\s", r"^---\s*$"])
        assert len(matches) == 3
        assert matches[0].matched_text == "# "
        assert matches[1].matched_text == "## "
        assert matches[2].matched_text == "---"

    def test_line_numbers_are_correct(self):
        text = "line0\nline1\nArticle 5\nline3"
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        assert len(matches) == 1
        assert matches[0].line_number == 2

    def test_position_is_char_offset(self):
        text = "abc\ndef\nArticle 1\nxyz"
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        assert matches[0].position == len("abc\ndef\n")

    def test_multiple_pattern_types(self):
        text = "CHAPTER I\nSomething\nArticle 1\nText\nSECTION 2\nMore"
        patterns = [r"^CHAPTER\s+", r"^Article\s+\d+", r"^SECTION\s+\d+"]
        matches = detect_boundaries(text, patterns)
        assert len(matches) == 3
        types = [m.matched_text for m in matches]
        assert "CHAPTER " in types[0]
        assert "Article 1" in types[1]
        assert "SECTION 2" in types[2]


class TestSplitAtBoundaries:
    def test_no_boundaries(self):
        result = split_at_boundaries("hello world", [])
        assert len(result) == 1
        assert result[0] == ("hello world", "start")

    def test_empty_text(self):
        assert split_at_boundaries("", []) == []

    def test_text_before_first_boundary(self):
        text = "Preamble text\nArticle 1\nContent"
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        segments = split_at_boundaries(text, matches)
        assert len(segments) == 2
        assert segments[0][1] == "start"
        assert "Preamble" in segments[0][0]
        assert segments[1][1] == "section"
        assert "Article 1" in segments[1][0]

    def test_boundary_at_start(self):
        text = "Article 1\nContent\nArticle 2\nMore"
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        segments = split_at_boundaries(text, matches)
        assert len(segments) == 2
        assert all(s[1] == "section" for s in segments)

    def test_preserves_full_text(self):
        text = "Pre\nArticle 1\nBody\nArticle 2\nEnd"
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        segments = split_at_boundaries(text, matches)
        reassembled = "".join(s[0] for s in segments)
        assert reassembled == text
