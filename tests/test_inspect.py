"""Tests for the inspect module — post-chunking diagnostics."""

from chunkweaver import Chunk, Chunker
from chunkweaver.inspect import (
    InspectionReport,
    _analyze_sizes,
    _boundary_breakdown,
    _detect_orphans,
    _find_near_miss_headings,
    _overlap_health,
    _suggest_patterns,
    inspect_chunks,
)

# --- Test documents ---

LEGAL_DOC = """\
REGULATION (EU) 2016/679

CHAPTER I
GENERAL PROVISIONS

Article 1
Subject matter and objectives

1. This Regulation lays down rules relating to the protection of natural
persons with regard to the processing of personal data and rules relating
to the free movement of personal data.

2. This Regulation protects fundamental rights and freedoms of natural
persons and in particular their right to the protection of personal data.

Article 2
Material scope

1. This Regulation applies to the processing of personal data wholly or
partly by automated means and to the processing other than by automated
means of personal data which form part of a filing system.

CHAPTER II
PRINCIPLES

Article 5
Principles relating to processing of personal data

Personal data shall be processed lawfully, fairly and in a transparent
manner in relation to the data subject.

Article 6
Lawfulness of processing

Processing shall be lawful only if and to the extent that at least one
of the following applies to the processing in question.

SCHEDULE A
Service Level Agreement

The provider shall maintain 99.9% uptime for all production services.
Response times shall not exceed 200ms for API endpoints.

SCHEDULE B
Fee Structure

Monthly fees are calculated based on usage tiers as defined herein.
"""

PLAIN_DOC = """\
The quick brown fox jumps over the lazy dog. This is just a paragraph
of plain text with no structural markers whatsoever. It should not
match any preset boundaries.

Another paragraph of plain text follows. Nothing special here either.
Just regular prose without any formatting or structure beyond basic
paragraph breaks. The text continues for a while to make sure we have
enough content for meaningful chunk analysis.

A third paragraph exists here to provide additional content. This helps
ensure that the chunking produces multiple segments for analysis. The
more text we have the better our inspection results will be.
"""


def _make_chunk(
    text: str,
    start: int = 0,
    end: int = 0,
    index: int = 0,
    boundary_type: str = "section",
    overlap_text: str = "",
) -> Chunk:
    return Chunk(
        text=text,
        start=start,
        end=end or start + len(text),
        index=index,
        boundary_type=boundary_type,
        overlap_text=overlap_text,
    )


# -----------------------------------------------------------------------
# Layer 1: Size distribution
# -----------------------------------------------------------------------


class TestSizeDistribution:
    def test_basic_stats(self):
        chunks = [
            _make_chunk("A" * 100, index=0),
            _make_chunk("B" * 200, index=1),
            _make_chunk("C" * 300, index=2),
        ]
        avg, med, mn, mx, cv, over, under = _analyze_sizes(chunks, target_size=500)
        assert avg == 200
        assert med == 200
        assert mn == 100
        assert mx == 300
        assert over == 0
        assert under == 1  # 100 < 200

    def test_oversized_detection(self):
        chunks = [
            _make_chunk("A" * 500, index=0),
            _make_chunk("B" * 2100, index=1),  # > 2x 1000
        ]
        _, _, _, _, _, over, _ = _analyze_sizes(chunks, target_size=1000)
        assert over == 1

    def test_empty_chunks(self):
        avg, med, mn, mx, cv, over, under = _analyze_sizes([], target_size=1024)
        assert avg == 0
        assert cv == 0.0


# -----------------------------------------------------------------------
# Layer 1: Boundary breakdown
# -----------------------------------------------------------------------


class TestBoundaryBreakdown:
    def test_section_splits(self):
        chunks = [
            _make_chunk("A", boundary_type="section"),
            _make_chunk("B", boundary_type="section"),
            _make_chunk("C", boundary_type="section"),
        ]
        counts, ratio = _boundary_breakdown(chunks)
        assert counts["section"] == 3
        assert ratio == 0.0

    def test_fallback_ratio(self):
        chunks = [
            _make_chunk("A", boundary_type="section"),
            _make_chunk("B", boundary_type="paragraph"),
            _make_chunk("C", boundary_type="sentence"),
            _make_chunk("D", boundary_type="word"),
        ]
        counts, ratio = _boundary_breakdown(chunks)
        assert ratio == 0.75  # 3/4 are fallback


# -----------------------------------------------------------------------
# Layer 1: Overlap health
# -----------------------------------------------------------------------


class TestOverlapHealth:
    def test_no_overlap(self):
        chunks = [_make_chunk("Hello world", overlap_text="")]
        assert _overlap_health(chunks) == []

    def test_healthy_overlap(self):
        chunks = [_make_chunk("Overlap. " + "Body text. " * 10, overlap_text="Overlap. ")]
        assert _overlap_health(chunks) == []

    def test_excessive_overlap(self):
        overlap = "X" * 60
        body = "Y" * 40
        chunks = [_make_chunk(overlap + body, index=0, overlap_text=overlap)]
        result = _overlap_health(chunks)
        assert 0 in result


# -----------------------------------------------------------------------
# Layer 1: Orphan detection
# -----------------------------------------------------------------------


class TestOrphanDetection:
    def test_normal_chunks_no_orphans(self):
        chunks = [
            _make_chunk("This is a normal chunk with plenty of content. " * 3),
        ]
        assert _detect_orphans(chunks) == []

    def test_heading_only_chunk(self):
        chunks = [
            _make_chunk("DEFINITIONS", index=0),
            _make_chunk("Long body content. " * 10, index=1),
        ]
        orphans = _detect_orphans(chunks)
        assert 0 in orphans
        assert 1 not in orphans


# -----------------------------------------------------------------------
# Layer 1: Suggestions
# -----------------------------------------------------------------------


class TestSuggestions:
    def test_high_fallback_ratio_suggestion(self):
        report = inspect_chunks(
            [
                _make_chunk("A" * 300, boundary_type="paragraph", index=0),
                _make_chunk("B" * 300, boundary_type="sentence", index=1),
                _make_chunk("C" * 300, boundary_type="paragraph", index=2),
                _make_chunk("D" * 300, boundary_type="word", index=3),
            ],
            "A" * 1200,
            target_size=500,
        )
        assert any("fallback" in s.lower() for s in report.suggestions)

    def test_no_suggestions_for_clean_config(self):
        chunks = [
            _make_chunk("A" * 500, boundary_type="section", index=0),
            _make_chunk("B" * 500, boundary_type="section", index=1),
            _make_chunk("C" * 500, boundary_type="section", index=2),
        ]
        report = inspect_chunks(chunks, "X" * 1500, target_size=600)
        fallback_suggestions = [s for s in report.suggestions if "fallback" in s.lower()]
        assert len(fallback_suggestions) == 0


# -----------------------------------------------------------------------
# Layer 2: Near-miss headings
# -----------------------------------------------------------------------


class TestNearMissHeadings:
    def test_finds_near_misses(self):
        text = "INTRODUCTION\n\nSome body text.\n\nConclusion notes."
        near = _find_near_miss_headings(text, used_line_numbers=set())
        # "INTRODUCTION" should score, "Conclusion notes" might be near-miss
        assert isinstance(near, list)

    def test_excludes_used_lines(self):
        text = "HEADING ONE\n\nBody.\n\nHEADING TWO\n\nMore body."
        # Mark line 0 as already used
        near = _find_near_miss_headings(text, used_line_numbers={0})
        used_lines = {nm.line_number for nm in near}
        assert 0 not in used_lines


# -----------------------------------------------------------------------
# Layer 2: Pattern suggestions
# -----------------------------------------------------------------------


class TestPatternSuggestions:
    def test_detects_schedule_patterns(self):
        suggestions = _suggest_patterns(LEGAL_DOC, existing_boundaries=[])
        patterns = {s.pattern for s in suggestions}
        assert r"^Article\s+\d+" in patterns or any("Article" in s.pattern for s in suggestions)

    def test_no_duplicate_suggestions(self):
        existing = [r"^Article\s+\d+"]
        suggestions = _suggest_patterns(LEGAL_DOC, existing_boundaries=existing)
        patterns = [s.pattern for s in suggestions]
        assert r"^Article\s+\d+" not in patterns

    def test_no_suggestions_for_plain_text(self):
        suggestions = _suggest_patterns(PLAIN_DOC, existing_boundaries=[])
        assert len(suggestions) == 0


# -----------------------------------------------------------------------
# Integration: full inspect_chunks
# -----------------------------------------------------------------------


class TestInspectChunks:
    def test_returns_report(self):
        chunker = Chunker(target_size=300, overlap=0, min_size=0)
        chunks = chunker.chunk_with_metadata(LEGAL_DOC)
        report = inspect_chunks(chunks, LEGAL_DOC, target_size=300)
        assert isinstance(report, InspectionReport)
        assert report.chunk_count > 0

    def test_empty_chunks(self):
        report = inspect_chunks([], "", target_size=1024)
        assert report.chunk_count == 0
        assert "zero chunks" in report.suggestions[0].lower()

    def test_report_string(self):
        chunker = Chunker(target_size=300, overlap=0, min_size=0)
        chunks = chunker.chunk_with_metadata(LEGAL_DOC)
        report = inspect_chunks(chunks, LEGAL_DOC, target_size=300)
        text = report.report()
        assert "chunkweaver inspect" in text
        assert "Boundary breakdown" in text

    def test_with_boundaries(self):
        boundaries = [r"^Article\s+\d+"]
        chunker = Chunker(target_size=300, overlap=0, boundaries=boundaries, min_size=0)
        chunks = chunker.chunk_with_metadata(LEGAL_DOC)
        report = inspect_chunks(
            chunks,
            LEGAL_DOC,
            target_size=300,
            boundaries=boundaries,
        )
        assert report.boundary_counts.get("section", 0) > 0

    def test_with_preset(self):
        from chunkweaver.presets import LEGAL_EU

        chunker = Chunker(target_size=500, overlap=0, boundaries=LEGAL_EU, min_size=0)
        chunks = chunker.chunk_with_metadata(LEGAL_DOC)
        report = inspect_chunks(
            chunks,
            LEGAL_DOC,
            target_size=500,
            boundaries=LEGAL_EU,
        )
        assert report.fallback_ratio < 0.5

    def test_pattern_suggestions_in_report(self):
        chunker = Chunker(target_size=300, overlap=0, min_size=0)
        chunks = chunker.chunk_with_metadata(LEGAL_DOC)
        report = inspect_chunks(chunks, LEGAL_DOC, target_size=300, boundaries=[])
        # Should suggest Article, SCHEDULE, etc.
        if report.pattern_suggestions:
            text = report.report()
            assert "Suggested boundary patterns" in text


# -----------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------


class TestReportFormat:
    def test_report_shows_all_sections(self):
        chunks = [
            _make_chunk("A" * 300, boundary_type="section", index=0),
            _make_chunk("B" * 300, boundary_type="paragraph", index=1),
        ]
        report = inspect_chunks(chunks, "X" * 600, target_size=500)
        text = report.report()
        assert "Chunks:" in text
        assert "Sizes:" in text
        assert "Boundary breakdown" in text

    def test_report_with_coherence(self):
        from chunkweaver.inspect import ChunkCoherenceRating

        chunks = [_make_chunk("Body text. " * 10, index=0)]
        report = inspect_chunks(chunks, "Body text. " * 10, target_size=500)
        report.coherence_ratings = [
            ChunkCoherenceRating(chunk_index=0, rating="coherent", explanation="looks good")
        ]
        report.coherence_summary = {"coherent": 1}
        text = report.report()
        assert "LLM coherence audit" in text
        assert "coherent" in text


# -----------------------------------------------------------------------
# CLI integration
# -----------------------------------------------------------------------


class TestCliInspect:
    def test_inspect_flag_runs(self):
        import io
        import sys

        from chunkweaver.cli import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--inspect", "README.md"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "chunkweaver inspect" in output
        assert "Boundary breakdown" in output
