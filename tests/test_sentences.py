"""Tests for sentence splitting."""

from chunkweaver.sentences import last_n_sentences, split_sentences


class TestSplitSentences:
    def test_empty_string(self):
        assert split_sentences("") == []

    def test_single_sentence(self):
        result = split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_two_sentences(self):
        text = "First sentence. Second sentence."
        parts = split_sentences(text)
        assert len(parts) == 2
        assert "".join(parts) == text

    def test_question_and_exclamation(self):
        text = "Is this correct? Yes! Absolutely."
        parts = split_sentences(text)
        assert len(parts) == 3
        assert "".join(parts) == text

    def test_lowercase_after_period_no_split(self):
        text = "Version 3.2 was released. It works."
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_abbreviation_false_split(self):
        """Known limitation: abbreviations like Dr. may cause false splits."""
        text = "Dr. Smith went home. He rested."
        parts = split_sentences(text)
        assert "".join(parts) == text

    def test_quoted_continuation(self):
        text = 'She said "Hello." And left.'
        parts = split_sentences(text)
        assert "".join(parts) == text

    def test_roundtrip(self):
        text = "Alpha. Beta. Gamma. Delta."
        assert "".join(split_sentences(text)) == text

    def test_no_trailing_space(self):
        text = "Only one."
        assert split_sentences(text) == ["Only one."]

    def test_multiline(self):
        text = "Line one.\nLine two. Line three."
        parts = split_sentences(text)
        assert "".join(parts) == text

    def test_custom_pattern_as_string(self):
        """The pattern= argument accepts a plain string, not just compiled regex."""
        text = "Alpha; Beta; Gamma; Delta"
        parts = split_sentences(text, pattern=r"(;)(\s+)")
        assert len(parts) == 4
        assert "".join(parts) == text

    def test_permissive_with_cyrillic(self):
        """SENTENCE_END_PERMISSIVE splits Cyrillic text where default would not."""
        from chunkweaver.sentences import SENTENCE_END_PERMISSIVE

        text = "Члан 1 дефинише обим. Члан 2 одређује услове."
        default = split_sentences(text)
        assert len(default) == 1  # default can't split (no A-Z after period)
        permissive = split_sentences(text, pattern=SENTENCE_END_PERMISSIVE)
        assert len(permissive) == 2
        assert "".join(permissive) == text


class TestLastNSentences:
    def test_zero_overlap(self):
        assert last_n_sentences("Hello. World.", 0) == ""

    def test_one_sentence(self):
        text = "First. Second. Third."
        result = last_n_sentences(text, 1)
        assert result == "Third."

    def test_two_sentences(self):
        text = "First. Second. Third."
        result = last_n_sentences(text, 2)
        assert "Second." in result
        assert "Third." in result

    def test_more_than_available(self):
        text = "Only one."
        result = last_n_sentences(text, 5)
        assert result == text

    def test_negative(self):
        assert last_n_sentences("Hello.", -1) == ""
