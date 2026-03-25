"""Tests for boundary presets on realistic text samples."""

import pytest

from chunkweaver import Chunker
from chunkweaver.presets import (
    LEGAL_EU,
    MARKDOWN,
    PLAIN,
    RFC,
    get_preset,
)


class TestPresetPatterns:
    def test_legal_eu_on_gdpr_sample(self):
        text = (
            "CHAPTER I\n"
            "General provisions\n\n"
            "Article 1\n"
            "Subject-matter and objectives\n"
            "(1) This Regulation lays down rules.\n"
            "(2) This Regulation protects fundamental rights.\n\n"
            "Article 2\n"
            "Material scope\n"
            "(1) This Regulation applies to processing.\n\n"
            "SECTION 1\n"
            "Transparency and modalities\n\n"
            "Article 12\n"
            "Transparent information, communication.\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=LEGAL_EU,
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 4

    def test_rfc_numbered_sections(self):
        text = (
            "1. Introduction\n"
            "This memo describes the protocol.\n\n"
            "2. Terminology\n"
            "The key words MUST, MUST NOT.\n\n"
            "3.1 Overview\n"
            "The protocol consists of.\n\n"
            "3.2. Detailed Design\n"
            "Messages are encoded as.\n\n"
            "Appendix A\n"
            "Additional examples.\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=RFC,
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 4

    def test_markdown_headers(self):
        text = (
            "# Title\n"
            "Intro paragraph.\n\n"
            "## Section 1\n"
            "Content of section 1.\n\n"
            "### Subsection 1.1\n"
            "Details here.\n\n"
            "---\n\n"
            "## Section 2\n"
            "More content.\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=MARKDOWN,
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 4

    def test_plain_preset_no_boundaries(self):
        text = "Some text.\n\nMore text.\n\nEven more."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=PLAIN,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_combined_presets(self):
        boundaries = LEGAL_EU + [r"^TABLE\s+"]
        text = "Article 1\nSome law.\nTABLE 1\nColumn A | Column B\n"
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=boundaries,
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


class TestGetPreset:
    def test_known_presets(self):
        assert get_preset("legal-eu") == LEGAL_EU
        assert get_preset("rfc") == RFC
        assert get_preset("markdown") == MARKDOWN
        assert get_preset("plain") == PLAIN

    def test_underscore_variant(self):
        assert get_preset("legal_eu") == LEGAL_EU

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_returns_copy(self):
        result = get_preset("legal-eu")
        result.append("extra")
        assert get_preset("legal-eu") == LEGAL_EU

    def test_leveled_returns_tuples(self):
        result = get_preset("legal-eu", leveled=True)
        assert len(result) > 0
        assert all(isinstance(spec, tuple) and len(spec) == 2 for spec in result)

    def test_leveled_fallback_to_flat(self):
        """Presets without a _LEVELED variant should return the flat version."""
        flat = get_preset("chat")
        leveled = get_preset("chat", leveled=True)
        assert flat == leveled

    def test_leveled_returns_copy(self):
        result = get_preset("rfc", leveled=True)
        result.append(("^extra", 9))
        assert get_preset("rfc", leveled=True) != result
