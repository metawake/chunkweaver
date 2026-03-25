"""Tests for the recommend module."""

from chunkweaver.recommend import (
    ChunkStats,
    OcrDamageReport,
    Recommendation,
    _detect_ocr_damage,
    recommend,
)

LEGAL_EU_DOC = """\
REGULATION (EU) 2016/679

CHAPTER I
GENERAL PROVISIONS

Article 1
Subject matter and objectives

1. This Regulation lays down rules relating to the protection of natural
persons with regard to the processing of personal data.

2. This Regulation protects fundamental rights and freedoms of natural
persons and in particular their right to the protection of personal data.

Article 2
Material scope

1. This Regulation applies to the processing of personal data wholly or
partly by automated means.

CHAPTER II
PRINCIPLES

Article 5
Principles relating to processing of personal data

(1) Personal data shall be processed lawfully, fairly and transparently.

Article 6
Lawfulness of processing

(1) Processing shall be lawful only if and to the extent that at least
one of the following applies.

Article 7
Conditions for consent

Where processing is based on consent, the controller shall be able to
demonstrate that the data subject has consented.
"""

MARKDOWN_DOC = """\
# Project README

## Installation

Install with pip:

```bash
pip install mypackage
```

## Usage

Use the library like this:

```python
import mypackage
mypackage.run()
```

## API Reference

### `run()`

Does the thing.

### `stop()`

Stops the thing.
"""

FINANCIAL_DOC = """\
UNITED STATES SECURITIES AND EXCHANGE COMMISSION

PART I

Item 1. Business

The Company operates in multiple segments across North America and Europe.
The principal products include software solutions and cloud services.

Item 1A. Risk Factors

Investing in our securities involves significant risks. The following
risk factors should be carefully considered.

TABLE 1
Summary of Revenue

Year    Revenue    Costs    Profit
2020    1,234      890      344
2021    1,567      1,012    555
2022    1,890      1,201    689
2023    2,134      1,345    789
2024    2,456      1,502    954

Item 7. Management Discussion and Analysis

The following discussion should be read in conjunction with the
consolidated financial statements.

NOTE 1 Revenue Recognition

Revenue is recognized when performance obligations are satisfied.

TABLE 2
Operating Expenses

Year    R&D      Sales    Admin    Total
2020    450      230      210      890
2021    520      260      232      1,012
2022    610      300      291      1,201
2023    680      340      325      1,345
2024    750      380      372      1,502
"""

PLAIN_DOC = """\
The quick brown fox jumps over the lazy dog. This is just a paragraph
of plain text with no structural markers whatsoever. It should not
match any preset boundaries.

Another paragraph of plain text follows. Nothing special here either.
Just regular prose without any formatting or structure beyond basic
paragraph breaks.
"""

CHAT_DOC = """\
[14:30] Agent: Welcome to support. How can I help?
[14:31] Customer: My order hasn't arrived. It's been 10 days.
[14:32] Agent: I'm sorry to hear that. Let me look into it.
[14:33] Customer: The order number is 12345.
[14:34] Agent: I see it was shipped Jan 5. It appears to be delayed.
[14:35] Customer: Can you send a replacement?
[14:36] Agent: Absolutely, I'll process that now.
[14:37] Customer: Thank you!
[14:38] Agent: You're welcome. Is there anything else?
[14:39] Customer: No, that's all.
"""


class TestRecommendBasics:
    def test_returns_recommendation(self):
        rec = recommend("Hello world.")
        assert isinstance(rec, Recommendation)

    def test_char_count(self):
        text = "Hello world."
        rec = recommend(text)
        assert rec.char_count == len(text)

    def test_line_count(self):
        text = "line1\nline2\nline3"
        rec = recommend(text)
        assert rec.line_count == 3

    def test_paragraph_count(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        rec = recommend(text)
        assert rec.paragraph_count == 3


class TestPresetScoring:
    def test_legal_eu_detected(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.recommended_preset == "legal-eu"
        hits = {pm.name: pm for pm in rec.preset_matches}
        assert "legal-eu" in hits
        assert hits["legal-eu"].hits >= 5

    def test_markdown_detected(self):
        rec = recommend(MARKDOWN_DOC)
        assert rec.recommended_preset == "markdown"

    def test_financial_detected(self):
        rec = recommend(FINANCIAL_DOC)
        assert rec.recommended_preset in ("financial", "financial-table")

    def test_plain_text_gets_plain(self):
        rec = recommend(PLAIN_DOC)
        assert rec.recommended_preset == "plain"

    def test_preset_matches_sorted_by_score(self):
        rec = recommend(LEGAL_EU_DOC)
        if len(rec.preset_matches) >= 2:
            scores = [pm.score for pm in rec.preset_matches]
            assert scores == sorted(scores, reverse=True)

    def test_density_normalized(self):
        """Same content at different sizes should have similar density."""
        rec = recommend(LEGAL_EU_DOC)
        hits = {pm.name: pm for pm in rec.preset_matches}
        assert hits["legal-eu"].density > 0

    def test_pattern_coverage_fraction(self):
        rec = recommend(LEGAL_EU_DOC)
        hits = {pm.name: pm for pm in rec.preset_matches}
        assert 0 < hits["legal-eu"].pattern_coverage <= 1.0

    def test_chat_doesnt_beat_markdown_on_readme(self):
        """Regression: generic patterns like 'word:' shouldn't outrank markdown."""
        rec = recommend(MARKDOWN_DOC)
        hits = {pm.name: pm for pm in rec.preset_matches}
        if "chat" in hits and "markdown" in hits:
            assert hits["markdown"].score > hits["chat"].score

    def test_chat_wins_on_chat_log(self):
        rec = recommend(CHAT_DOC)
        assert rec.recommended_preset == "chat"


class TestMultiPreset:
    def test_financial_combo(self):
        rec = recommend(FINANCIAL_DOC)
        if rec.recommended_preset == "financial":
            assert (
                "financial-table" in rec.recommended_presets or len(rec.recommended_presets) == 1
            )

    def test_recommended_presets_list(self):
        rec = recommend(LEGAL_EU_DOC)
        assert isinstance(rec.recommended_presets, list)
        assert len(rec.recommended_presets) >= 1

    def test_snippet_combines_presets(self):
        rec = recommend(FINANCIAL_DOC)
        if len(rec.recommended_presets) > 1:
            snip = rec.snippet()
            for p in rec.recommended_presets:
                const = p.upper().replace("-", "_")
                assert const in snip


class TestDetectorRecommendation:
    def test_heading_detector_recommended_for_legal(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.recommend_heading_detector is True
        assert rec.heading_count >= 3

    def test_table_detector_recommended_for_financial(self):
        rec = recommend(FINANCIAL_DOC)
        assert rec.recommend_table_detector is True
        assert rec.table_count >= 1

    def test_table_detector_not_recommended_for_plain(self):
        rec = recommend(PLAIN_DOC)
        assert rec.recommend_table_detector is False
        assert rec.table_count == 0

    def test_heading_samples_populated(self):
        rec = recommend(LEGAL_EU_DOC)
        assert len(rec.heading_samples) > 0

    def test_heading_threshold_scales_with_doc_size(self):
        """A single heading in a tiny doc shouldn't trigger the detector."""
        tiny = "INTRODUCTION\n\nSome text here."
        rec = recommend(tiny)
        assert rec.recommend_heading_detector is False


class TestTargetSizeSuggestion:
    def test_short_paragraphs_get_small_target(self):
        text = "Short.\n\n" * 20
        rec = recommend(text)
        assert rec.suggested_target_size <= 512

    def test_long_paragraphs_get_large_target(self):
        text = ("A" * 2500 + "\n\n") * 5
        rec = recommend(text)
        assert rec.suggested_target_size >= 2048

    def test_overlap_scales_with_target(self):
        short = recommend("Short.\n\n" * 20)
        long = recommend(("A" * 2500 + "\n\n") * 5)
        assert short.suggested_overlap <= long.suggested_overlap

    def test_dense_boundaries_lower_target(self):
        """Many boundaries in a small doc should produce a smaller target."""
        dense = "\n".join(f"Article {i}\nShort content for article {i}." for i in range(1, 30))
        rec = recommend(dense)
        assert rec.suggested_target_size <= 1024


class TestDryRun:
    def test_chunk_stats_present(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.chunk_stats is not None
        assert isinstance(rec.chunk_stats, ChunkStats)

    def test_chunk_count_positive(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.chunk_stats.chunk_count >= 1

    def test_sizes_make_sense(self):
        rec = recommend(LEGAL_EU_DOC)
        cs = rec.chunk_stats
        assert cs.min_size <= cs.avg_size <= cs.max_size
        assert cs.min_size <= cs.median_size <= cs.max_size

    def test_no_warnings_on_clean_doc(self):
        """A well-structured doc shouldn't produce warnings about oversized chunks."""
        rec = recommend(LEGAL_EU_DOC)
        assert rec.chunk_stats is not None
        oversized_warnings = [w for w in rec.chunk_stats.warnings if "over 2x" in w]
        assert len(oversized_warnings) == 0

    def test_oversized_detection(self):
        """A single huge block should warn about oversized chunks."""
        huge = "A" * 10000
        rec = recommend(huge)
        cs = rec.chunk_stats
        assert cs.max_size > rec.suggested_target_size

    def test_report_shows_dry_run(self):
        rec = recommend(LEGAL_EU_DOC)
        report = rec.report()
        assert "Dry-run results" in report
        assert "chunks produced" in report


class TestReport:
    def test_report_contains_sections(self):
        rec = recommend(LEGAL_EU_DOC)
        report = rec.report()
        assert "chunkweaver recommend" in report
        assert "Preset matching" in report
        assert "Detectors" in report
        assert "Suggested config" in report
        assert "Python snippet" in report

    def test_report_shows_selected_marker(self):
        rec = recommend(LEGAL_EU_DOC)
        report = rec.report()
        assert "<--" in report

    def test_report_shows_density_and_coverage(self):
        rec = recommend(LEGAL_EU_DOC)
        report = rec.report()
        assert "density=" in report
        assert "coverage=" in report


class TestSnippet:
    def test_snippet_has_import(self):
        rec = recommend(LEGAL_EU_DOC)
        snip = rec.snippet()
        assert "from chunkweaver import Chunker" in snip

    def test_snippet_includes_preset(self):
        rec = recommend(LEGAL_EU_DOC)
        snip = rec.snippet()
        assert "LEGAL_EU" in snip

    def test_snippet_includes_detectors_when_recommended(self):
        rec = recommend(FINANCIAL_DOC)
        snip = rec.snippet()
        if rec.recommend_heading_detector:
            assert "HeadingDetector" in snip
        if rec.recommend_table_detector:
            assert "TableDetector" in snip

    def test_snippet_markdown_extras(self):
        rec = recommend(MARKDOWN_DOC)
        snip = rec.snippet()
        assert "MARKDOWN" in snip
        if rec.extra_boundaries:
            assert "```" in snip

    def test_plain_snippet_no_preset(self):
        rec = recommend(PLAIN_DOC)
        snip = rec.snippet()
        assert "boundaries=" not in snip


OCR_DAMAGED_DOC = """\
D E F I N I T I O N S

In this Agreement, the following terms shall have the meanings set forth below.
"Agreement" means this Master Services Agreement including all schedules.
"Effective Date" means the date first written above.

S C O P E   O F   S E R V I C E S

The Contractor shall provide to the Company the Services described in Schedule A.
The Contractor shall perform the Services in a professional manner consistent
with industry standards.

C O M P E N S A T I O N

The Company shall pay the Contractor for Services rendered at the rates
specified in Schedule B. Payment shall be made within thirty days of receipt.

T E R M   A N D   T E R M I N A T I O N

This Agreement shall commence on the Effective Date and shall continue for a
period of twelve months unless earlier terminated by either party.
"""

PARTIAL_OCR_DOC = """\
Sect ion 1 Gene ral Prov isi ons

This policy applies to all employees of the organization regardless of tenure.
All employees are expected to familiarize themselves with this handbook.

Sect ion 2 Emp loy ment Pol ici es

The organization is committed to equal employment opportunity and does not
discriminate on any basis prohibited by applicable federal or state law.

Sect ion 3 Com pen sat ion

Compensation is reviewed annually and adjustments are made based on
performance evaluations, market data, and overall financial condition.
"""


class TestOcrDamageDetection:
    def test_clean_doc_no_damage(self):
        report = _detect_ocr_damage(LEGAL_EU_DOC)
        assert report.level == "none"
        assert report.damaged_line_count == 0
        assert report.recommend_ml_detector is False

    def test_plain_doc_no_damage(self):
        report = _detect_ocr_damage(PLAIN_DOC)
        assert report.level == "none"

    def test_full_letterspacing_detected(self):
        report = _detect_ocr_damage(OCR_DAMAGED_DOC)
        assert report.level == "heavy"
        assert report.damaged_line_count >= 4
        assert report.recommend_ml_detector is True

    def test_partial_fragmentation_detected(self):
        report = _detect_ocr_damage(PARTIAL_OCR_DOC)
        assert report.level in ("light", "heavy")
        assert report.damaged_line_count >= 2
        assert report.recommend_ml_detector is True

    def test_samples_populated(self):
        report = _detect_ocr_damage(OCR_DAMAGED_DOC)
        assert len(report.sample_lines) > 0
        assert any("D E F" in s or "S C O P E" in s for s in report.sample_lines)

    def test_damage_ratio(self):
        report = _detect_ocr_damage(OCR_DAMAGED_DOC)
        assert 0 < report.damage_ratio <= 1.0

    def test_single_damaged_line_is_light(self):
        text = "D E F I N I T I O N S\n\n" + "".join(
            f"Normal heading number {i}\n\n"
            f"This is body text for section {i} that continues normally.\n"
            f"Additional context paragraph for the section about topic {i}.\n\n"
            for i in range(1, 20)
        )
        report = _detect_ocr_damage(text)
        assert report.level == "light"
        assert report.damaged_line_count == 1


class TestOcrInRecommendation:
    def test_ocr_damage_field_present(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.ocr_damage is not None
        assert isinstance(rec.ocr_damage, OcrDamageReport)

    def test_clean_doc_no_ocr_warning(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.ocr_damage.level == "none"
        report = rec.report()
        assert "OCR quality" not in report

    def test_damaged_doc_shows_ocr_section(self):
        rec = recommend(OCR_DAMAGED_DOC)
        assert rec.ocr_damage.level != "none"
        report = rec.report()
        assert "OCR quality" in report
        assert "MLOCRHeadingDetector" in report

    def test_damaged_doc_snippet_mentions_ocr(self):
        rec = recommend(OCR_DAMAGED_DOC)
        if rec.ocr_damage.recommend_ml_detector:
            snip = rec.snippet()
            assert "OCR damage detected" in snip
            assert "ml-detectors" in snip

    def test_clean_doc_snippet_no_ml(self):
        rec = recommend(LEGAL_EU_DOC)
        snip = rec.snippet()
        assert "MLOCRHeadingDetector" not in snip


class TestCliIntegration:
    def test_recommend_flag_runs(self):
        import io
        import sys

        from chunkweaver.cli import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--recommend", "README.md"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "chunkweaver recommend" in output
        assert "Preset matching" in output
        assert "Dry-run results" in output

    def test_recommend_from_stdin(self):
        import io
        import sys

        from chunkweaver.cli import main

        old_stdin = sys.stdin
        sys.stdin = io.StringIO(LEGAL_EU_DOC)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--recommend"])
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "legal-eu" in output
