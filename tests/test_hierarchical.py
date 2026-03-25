"""Tests for hierarchical boundary splitting and annotation ingestion."""

from chunkweaver import Chunker, KeepTogetherRegion, SplitPoint
from chunkweaver.boundaries import BoundarySpec, detect_boundaries
from chunkweaver.presets import (
    FINANCIAL_LEVELED,
    LEGAL_EU,
    LEGAL_EU_LEVELED,
    MARKDOWN_LEVELED,
    RFC_LEVELED,
)

# -----------------------------------------------------------------------
# Fixtures — sample documents for each domain
# -----------------------------------------------------------------------

GDPR_CHAPTER = (
    "PREAMBLE\n"
    "The European Parliament and the Council of the European Union have adopted "
    "this regulation on the protection of personal data.\n\n"
    "CHAPTER I\n"
    "GENERAL PROVISIONS\n\n"
    "Article 1\n"
    "Subject-matter and objectives.\n"
    "This Regulation lays down rules relating to the protection "
    "of natural persons with regard to the processing of personal data "
    "and rules relating to the free movement of personal data.\n"
    "(1) This Regulation protects fundamental rights and freedoms of "
    "natural persons and in particular their right to the protection "
    "of personal data.\n"
    "(2) The free movement of personal data within the Union shall be "
    "neither restricted nor prohibited for reasons connected with the "
    "protection of natural persons.\n\n"
    "Article 2\n"
    "Material scope.\n"
    "This Regulation applies to the processing of personal data wholly "
    "or partly by automated means and to the processing other than by "
    "automated means of personal data which form part of a filing system.\n"
    "(1) This Regulation does not apply to the processing of personal "
    "data in the course of a purely personal or household activity.\n"
    "(2) This Regulation does not apply to the processing of personal "
    "data by competent authorities for the purposes of prevention.\n\n"
    "CHAPTER II\n"
    "PRINCIPLES\n\n"
    "Article 5\n"
    "Principles relating to processing of personal data.\n"
    "(1) Personal data shall be processed lawfully, fairly and in a "
    "transparent manner in relation to the data subject.\n"
    "(2) Personal data shall be collected for specified, explicit and "
    "legitimate purposes and not further processed in a manner that is "
    "incompatible with those purposes.\n"
)


RFC_NESTED = (
    "1. Introduction\n"
    "This document describes the JSON Web Token (JWT) format.\n"
    "1.1 Notational Conventions\n"
    "The key words MUST and SHALL are to be interpreted as described in RFC 2119.\n"
    "1.2 Terminology\n"
    "Base64url encoding is defined in RFC 4648.\n\n"
    "2. Terminology\n"
    "The following terms are used throughout this document.\n"
    "2.1 JSON Web Token\n"
    "A string representing a set of claims as a JSON object.\n"
    "2.2 Claim\n"
    "A piece of information asserted about a subject.\n\n"
    "3. JSON Web Token Overview\n"
    "JWTs represent a set of claims as a JSON object.\n"
    "3.1 Example JWT\n"
    "The following is an example of a JWT Header.\n"
)


SEC_FILING = (
    "PART I\n\n"
    "Item 1. Business\n"
    "The company operates in the technology sector providing cloud services "
    "and enterprise software solutions to customers worldwide.\n"
    "NOTE 1 Revenue Recognition\n"
    "Revenue is recognized when performance obligations are satisfied.\n\n"
    "Item 1A. Risk Factors\n"
    "Investing in our securities involves significant risks.\n"
    "TABLE 1\n"
    "Revenue 2023 2024\n"
    "Product    1000  1200\n"
    "Service    800   950\n\n"
    "PART II\n\n"
    "Item 5. Market\n"
    "Our stock is listed on the NASDAQ stock exchange.\n"
    "Schedule A\n"
    "Supplemental financial data for the fiscal year.\n"
)


MARKDOWN_DOC = (
    "# Project Overview\n"
    "This project provides a toolkit for NLP processing.\n\n"
    "## Installation\n"
    "Run pip install to get started with the package.\n\n"
    "### Requirements\n"
    "Python 3.9 or higher is required.\n\n"
    "### Optional Dependencies\n"
    "Install extras for CLI and LangChain support.\n\n"
    "## Usage\n"
    "Import the main class and configure it.\n\n"
    "### Basic Example\n"
    "Create a chunker with default settings.\n\n"
    "### Advanced Example\n"
    "Use custom boundaries for your domain.\n\n"
    "# API Reference\n"
    "Full documentation of public classes.\n"
)


# -----------------------------------------------------------------------
# detect_boundaries with levels
# -----------------------------------------------------------------------


class TestLeveledDetection:
    def test_tuple_patterns_detected(self):
        text = "CHAPTER I\nArticle 1\nContent\nArticle 2\nMore"
        patterns: list[BoundarySpec] = [
            (r"^CHAPTER\s+[IVX\d]+", 0),
            (r"^Article\s+\d+", 1),
        ]
        matches = detect_boundaries(text, patterns)
        assert len(matches) == 3
        assert matches[0].level == 0
        assert matches[0].matched_text == "CHAPTER I"
        assert matches[1].level == 1
        assert matches[2].level == 1

    def test_mixed_str_and_tuple(self):
        text = "CHAPTER I\nArticle 1\nContent"
        patterns: list[BoundarySpec] = [
            r"^CHAPTER\s+[IVX\d]+",
            (r"^Article\s+\d+", 1),
        ]
        matches = detect_boundaries(text, patterns)
        assert matches[0].level == 0  # str defaults to 0
        assert matches[1].level == 1

    def test_plain_strings_are_level_zero(self):
        text = "Article 1\nContent\nArticle 2\nMore"
        matches = detect_boundaries(text, [r"^Article\s+\d+"])
        for m in matches:
            assert m.level == 0


# -----------------------------------------------------------------------
# Hierarchical chunking — legal documents
# -----------------------------------------------------------------------


class TestHierarchicalLegal:
    def test_small_chapter_stays_intact(self):
        """When a chapter fits target_size, don't split at Article level."""
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=LEGAL_EU_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(GDPR_CHAPTER)
        chapter_chunks = [c for c in chunks if "CHAPTER I" in c.text]
        assert len(chapter_chunks) >= 1
        first_ch = chapter_chunks[0]
        assert "Article 1" in first_ch.text
        assert "Article 2" in first_ch.text

    def test_oversized_chapter_splits_at_articles(self):
        """When a chapter exceeds target_size, split at Article boundaries."""
        chunker = Chunker(
            target_size=400,
            overlap=0,
            boundaries=LEGAL_EU_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(GDPR_CHAPTER)
        article_starts = [c for c in chunks if c.strip().startswith("Article")]
        assert len(article_starts) >= 2

    def test_chapter_heading_merges_with_first_article(self):
        """Chapter heading should merge into first article, not be orphaned."""
        chunker = Chunker(
            target_size=400,
            overlap=0,
            boundaries=LEGAL_EU_LEVELED,
            min_size=50,
        )
        chunks = chunker.chunk(GDPR_CHAPTER)
        orphan_chapters = [c for c in chunks if c.strip() in ("CHAPTER I", "CHAPTER II")]
        assert len(orphan_chapters) == 0

    def test_recitals_stay_with_article_when_small(self):
        """Recitals shouldn't split from their article if it fits."""
        chunker = Chunker(
            target_size=2000,
            overlap=0,
            boundaries=LEGAL_EU_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(GDPR_CHAPTER)
        for c in chunks:
            if "Article 1" in c and c.strip().startswith("Article 1"):
                assert "(1)" in c
                assert "(2)" in c
                break

    def test_boundary_level_on_chunk_metadata(self):
        chunker = Chunker(
            target_size=400,
            overlap=0,
            boundaries=LEGAL_EU_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(GDPR_CHAPTER)
        levels = {c.boundary_level for c in chunks}
        assert 0 in levels or 2 in levels  # at least some level info

    def test_flat_legal_eu_unchanged(self):
        """Original LEGAL_EU (all level 0) should produce same result as before."""
        chunker_flat = Chunker(
            target_size=2000,
            overlap=0,
            boundaries=LEGAL_EU,
            min_size=0,
        )
        chunker_strings = Chunker(
            target_size=2000,
            overlap=0,
            boundaries=[
                r"^Article\s+\d+",
                r"^\(\d+\)\s+",
                r"^CHAPTER\s+[IVX\d]+",
                r"^SECTION\s+\d+",
            ],
            min_size=0,
        )
        chunks_flat = chunker_flat.chunk(GDPR_CHAPTER)
        chunks_str = chunker_strings.chunk(GDPR_CHAPTER)
        assert chunks_flat == chunks_str


# -----------------------------------------------------------------------
# Hierarchical chunking — RFC documents
# -----------------------------------------------------------------------


class TestHierarchicalRFC:
    def test_small_sections_stay_intact(self):
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=RFC_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(RFC_NESTED)
        section_1_chunks = [c for c in chunks if c.strip().startswith("1. ")]
        assert len(section_1_chunks) == 1
        assert "1.1" in section_1_chunks[0]
        assert "1.2" in section_1_chunks[0]

    def test_oversized_section_splits_at_subsections(self):
        chunker = Chunker(
            target_size=200,
            overlap=0,
            boundaries=RFC_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(RFC_NESTED)
        subsection_starts = [c for c in chunks if "1.1" in c or "1.2" in c]
        assert len(subsection_starts) >= 2

    def test_top_level_sections_always_split(self):
        """Level-0 boundaries always split regardless of size."""
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=RFC_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(RFC_NESTED)
        assert len(chunks) >= 3  # at least sections 1, 2, 3


# -----------------------------------------------------------------------
# Hierarchical chunking — financial documents
# -----------------------------------------------------------------------


class TestHierarchicalFinancial:
    def test_small_part_stays_intact(self):
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=FINANCIAL_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(SEC_FILING)
        part_1_chunks = [c for c in chunks if "PART I" in c and "PART II" not in c]
        assert len(part_1_chunks) >= 1
        assert "Item 1." in part_1_chunks[0]

    def test_oversized_part_splits_at_items(self):
        chunker = Chunker(
            target_size=200,
            overlap=0,
            boundaries=FINANCIAL_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(SEC_FILING)
        item_starts = [c for c in chunks if "Item 1." in c or "Item 1A." in c or "Item 5." in c]
        assert len(item_starts) >= 2


# -----------------------------------------------------------------------
# Hierarchical chunking — markdown documents
# -----------------------------------------------------------------------


class TestHierarchicalMarkdown:
    def test_h1_always_splits(self):
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=MARKDOWN_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(MARKDOWN_DOC)
        h1_chunks = [c for c in chunks if c.strip().startswith("# ")]
        assert len(h1_chunks) == 2  # "# Project Overview" and "# API Reference"

    def test_h2_stays_with_h1_when_small(self):
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=MARKDOWN_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(MARKDOWN_DOC)
        overview = [c for c in chunks if "# Project Overview" in c][0]
        assert "## Installation" in overview
        assert "## Usage" in overview

    def test_h2_splits_when_h1_oversized(self):
        chunker = Chunker(
            target_size=200,
            overlap=0,
            boundaries=MARKDOWN_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk(MARKDOWN_DOC)
        h2_chunks = [c for c in chunks if "## " in c]
        assert len(h2_chunks) >= 2


# -----------------------------------------------------------------------
# Annotation ingestion from extractors
# -----------------------------------------------------------------------


class TestAnnotationIngestion:
    def test_split_points_create_boundaries(self):
        text = "First section content here.\nSecond section content here.\nThird section."
        lines = text.split("\n")
        pos_1 = len(lines[0]) + 1
        pos_2 = pos_1 + len(lines[1]) + 1

        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            annotations=[
                SplitPoint(position=pos_1, line_number=1, label="heading"),
                SplitPoint(position=pos_2, line_number=2, label="heading"),
            ],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_keep_together_regions_respected(self):
        text = (
            "Introduction text.\n"
            "Table Header\n"
            "Row 1: 100 200 300\n"
            "Row 2: 400 500 600\n"
            "Row 3: 700 800 900\n"
            "Conclusion text.\n"
        )
        table_start = text.index("Table Header")
        table_end = text.index("Conclusion")

        chunker = Chunker(
            target_size=50,
            overlap=0,
            min_size=0,
            annotations=[
                KeepTogetherRegion(
                    start=table_start,
                    end=table_end,
                    label="table",
                    max_overshoot=3.0,
                ),
            ],
        )
        chunks = chunker.chunk(text)
        table_chunks = [c for c in chunks if "Row 1" in c]
        assert len(table_chunks) == 1
        assert "Row 2" in table_chunks[0]
        assert "Row 3" in table_chunks[0]

    def test_annotations_with_levels(self):
        text = (
            "Chapter One Content.\n"
            "Section A Content.\n"
            "Section B Content.\n"
            "Chapter Two Content.\n"
            "Section C Content.\n"
        )
        lines = text.split("\n")
        offsets = []
        pos = 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln) + 1

        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            annotations=[
                SplitPoint(position=offsets[0], line_number=0, label="ch1", level=0),
                SplitPoint(position=offsets[1], line_number=1, label="secA", level=1),
                SplitPoint(position=offsets[2], line_number=2, label="secB", level=1),
                SplitPoint(position=offsets[3], line_number=3, label="ch2", level=0),
                SplitPoint(position=offsets[4], line_number=4, label="secC", level=1),
            ],
        )
        chunks = chunker.chunk(text)
        # Chapter 1 + sections A&B should stay together (fits target_size)
        assert len(chunks) == 2
        assert "Section A" in chunks[0]
        assert "Section B" in chunks[0]
        assert "Chapter Two" in chunks[1]

    def test_annotations_merge_with_detectors(self):
        text = "Intro.\nHEADING ONE\nBody text for heading one.\nAnnotated split.\nMore text."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[r"^HEADING"],
            annotations=[
                SplitPoint(
                    position=text.index("Annotated"),
                    line_number=3,
                    label="external",
                ),
            ],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_annotations_merge_with_regex_boundaries(self):
        text = "Article 1\nContent.\nAnnotated boundary.\nMore content."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[r"^Article\s+\d+"],
            annotations=[
                SplitPoint(
                    position=text.index("Annotated"),
                    line_number=2,
                    label="external",
                ),
            ],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2  # Article + Annotated

    def test_empty_annotations(self):
        text = "Article 1\nContent.\nArticle 2\nMore."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[r"^Article\s+\d+"],
            annotations=[],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


# -----------------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_string_boundaries_still_work(self):
        text = "Article 1\nContent.\nArticle 2\nMore."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[r"^Article\s+\d+"],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_chunk_has_boundary_level_default(self):
        text = "Article 1\nContent.\nArticle 2\nMore."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[r"^Article\s+\d+"],
        )
        chunks = chunker.chunk_with_metadata(text)
        for c in chunks:
            assert c.boundary_level == 0

    def test_no_boundaries_no_annotations(self):
        text = "Just some plain text without any structure at all."
        chunker = Chunker(target_size=5000, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_preset_strings_backward_compat(self):
        """LEGAL_EU (list of strings) still works with new detect_boundaries."""
        text = "Article 1\nContent\nArticle 2\nMore"
        matches = detect_boundaries(text, LEGAL_EU)
        assert len(matches) == 2
        assert all(m.level == 0 for m in matches)


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestHierarchicalEdgeCases:
    def test_only_deep_levels_no_level_zero(self):
        """When only level-1+ boundaries exist, everything starts as one segment."""
        text = "Some text.\nArticle 1\nContent.\nArticle 2\nMore."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[(r"^Article\s+\d+", 1)],
        )
        chunks = chunker.chunk(text)
        # No level-0 boundaries → entire text is one segment, fits target
        assert len(chunks) == 1

    def test_deep_levels_split_when_oversized(self):
        text = "Some text.\nArticle 1\n" + "Content. " * 100 + "\nArticle 2\n" + "More. " * 100
        chunker = Chunker(
            target_size=200,
            overlap=0,
            min_size=0,
            boundaries=[(r"^Article\s+\d+", 1)],
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_three_levels_progressive_refinement(self):
        """Three-level hierarchy: PART > Section > clause."""
        text = (
            "PART I\n"
            "Section 1\n" + "Content A. " * 30 + "\n"
            "1.1 First clause\n" + "Detail A. " * 30 + "\n"
            "1.2 Second clause\n" + "Detail B. " * 30 + "\n"
            "Section 2\n" + "Content B. " * 30 + "\n"
            "PART II\n"
            "Section 3\n" + "Content C. " * 10 + "\n"
        )
        chunker_big = Chunker(
            target_size=5000,
            overlap=0,
            min_size=0,
            boundaries=[
                (r"^PART\s+[IVX]+", 0),
                (r"^Section\s+\d+", 1),
                (r"^\d+\.\d+\s", 2),
            ],
        )
        chunks_big = chunker_big.chunk(text)
        # At 5000 chars, everything fits → only PART boundaries split
        assert len(chunks_big) == 2

        chunker_small = Chunker(
            target_size=400,
            overlap=0,
            min_size=0,
            boundaries=[
                (r"^PART\s+[IVX]+", 0),
                (r"^Section\s+\d+", 1),
                (r"^\d+\.\d+\s", 2),
            ],
        )
        chunks_small = chunker_small.chunk(text)
        # At 400 chars, PART I needs splitting → Section splits
        assert len(chunks_small) > 2

    def test_overlap_works_with_hierarchy(self):
        chunker = Chunker(
            target_size=400,
            overlap=1,
            overlap_unit="sentence",
            boundaries=LEGAL_EU_LEVELED,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(GDPR_CHAPTER)
        if len(chunks) > 1:
            has_overlap = any(c.overlap_text != "" for c in chunks[1:])
            assert has_overlap
