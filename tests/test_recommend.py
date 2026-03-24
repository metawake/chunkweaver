"""Tests for the recommend module."""

import pytest

from chunkweaver.recommend import Recommendation, recommend


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

The Company operates in multiple segments.

Item 1A. Risk Factors

TABLE 1
Summary of Revenue

Year    Revenue    Costs    Profit
2020    1,234      890      344
2021    1,567      1,012    555
2022    1,890      1,201    689
2023    2,134      1,345    789
2024    2,456      1,502    954

Item 7. Management Discussion

NOTE 1 Revenue Recognition
"""

PLAIN_DOC = """\
The quick brown fox jumps over the lazy dog. This is just a paragraph
of plain text with no structural markers whatsoever. It should not
match any preset boundaries.

Another paragraph of plain text follows. Nothing special here either.
Just regular prose without any formatting or structure beyond basic
paragraph breaks.
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


class TestPresetMatching:
    def test_legal_eu_detected(self):
        rec = recommend(LEGAL_EU_DOC)
        assert rec.recommended_preset == "legal-eu"
        hits = {pm.name: pm.hits for pm in rec.preset_matches}
        assert hits["legal-eu"] >= 5

    def test_markdown_detected(self):
        rec = recommend(MARKDOWN_DOC)
        assert rec.recommended_preset == "markdown"

    def test_financial_detected(self):
        rec = recommend(FINANCIAL_DOC)
        assert rec.recommended_preset in ("financial", "financial-table")

    def test_plain_text_gets_plain(self):
        rec = recommend(PLAIN_DOC)
        assert rec.recommended_preset == "plain"

    def test_preset_matches_sorted_by_hits(self):
        rec = recommend(LEGAL_EU_DOC)
        if len(rec.preset_matches) >= 2:
            hits = [pm.hits for pm in rec.preset_matches]
            assert hits == sorted(hits, reverse=True)


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


class TestReport:
    def test_report_contains_sections(self):
        rec = recommend(LEGAL_EU_DOC)
        report = rec.report()
        assert "chunkweaver recommend" in report
        assert "Preset matching" in report
        assert "Detectors" in report
        assert "Suggested config" in report
        assert "Python snippet" in report

    def test_report_shows_best_marker(self):
        rec = recommend(LEGAL_EU_DOC)
        report = rec.report()
        assert "<-- best" in report


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


class TestCliIntegration:
    def test_recommend_flag_runs(self):
        from chunkweaver.cli import main
        import io
        import sys

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

    def test_recommend_from_stdin(self):
        from chunkweaver.cli import main
        import io
        import sys

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
