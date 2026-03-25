"""Tests for BoundaryDetector ABC and Chunker integration."""

from __future__ import annotations

import pytest

from chunkweaver import (
    Annotation,
    BoundaryDetector,
    Chunker,
    KeepTogetherRegion,
    SplitPoint,
)

# ---------------------------------------------------------------------------
# Test detectors — concrete implementations for testing
# ---------------------------------------------------------------------------


class FixedSplitDetector(BoundaryDetector):
    """Splits at lines matching exact strings."""

    def __init__(self, markers: list[str]):
        self._markers = markers

    def detect(self, text: str) -> list[Annotation]:
        results: list[Annotation] = []
        offset = 0
        for line_no, line in enumerate(text.split("\n")):
            stripped = line.strip()
            if stripped in self._markers:
                results.append(
                    SplitPoint(
                        position=offset,
                        line_number=line_no,
                        label=f"marker: {stripped}",
                    )
                )
            offset += len(line) + 1
        return results


class FixedKeepTogetherDetector(BoundaryDetector):
    """Marks specific char ranges as keep-together."""

    def __init__(self, regions: list[tuple]):
        self._regions = regions

    def detect(self, text: str) -> list[Annotation]:
        return [
            KeepTogetherRegion(
                start=start,
                end=end,
                label=label,
                max_overshoot=overshoot,
            )
            for start, end, label, overshoot in self._regions
        ]


# ---------------------------------------------------------------------------
# ABC tests
# ---------------------------------------------------------------------------


class TestBoundaryDetectorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BoundaryDetector()

    def test_concrete_subclass(self):
        detector = FixedSplitDetector(["## Heading"])
        assert isinstance(detector, BoundaryDetector)

    def test_detect_returns_list(self):
        detector = FixedSplitDetector(["## Heading"])
        result = detector.detect("line one\n## Heading\nline three")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SplitPoint)


# ---------------------------------------------------------------------------
# SplitPoint integration — detectors add split boundaries
# ---------------------------------------------------------------------------


class TestSplitPointIntegration:
    def test_detector_splits_like_regex(self):
        text = (
            "Intro paragraph.\n\n## Revenue\n\nRevenue was strong.\n\n## Costs\n\nCosts were low."
        )

        chunker_regex = Chunker(
            target_size=50,
            overlap=0,
            boundaries=[r"^## "],
            min_size=0,
        )
        chunker_detector = Chunker(
            target_size=50,
            overlap=0,
            detectors=[FixedSplitDetector(["## Revenue", "## Costs"])],
            min_size=0,
        )

        chunks_regex = chunker_regex.chunk(text)
        chunks_detector = chunker_detector.chunk(text)

        assert len(chunks_regex) == len(chunks_detector)
        assert chunks_regex == chunks_detector

    def test_detector_and_regex_combine(self):
        text = "PART I\n\nSome intro.\n\nSECTION A\n\nContent A.\n\nSECTION B\n\nContent B."

        chunker = Chunker(
            target_size=40,
            overlap=0,
            boundaries=[r"^PART\s+"],
            detectors=[FixedSplitDetector(["SECTION A", "SECTION B"])],
            min_size=0,
        )
        chunks = chunker.chunk(text)

        starts = [c.split("\n")[0].strip() for c in chunks]
        assert "PART I" in starts
        assert "SECTION A" in starts
        assert "SECTION B" in starts

    def test_detector_split_deduplicates_with_regex(self):
        """When a detector split point coincides with a regex boundary,
        it should not create a duplicate split."""
        text = "PART I\n\nContent.\n\nPART II\n\nMore content."

        chunker = Chunker(
            target_size=30,
            overlap=0,
            boundaries=[r"^PART\s+"],
            detectors=[FixedSplitDetector(["PART II"])],
            min_size=0,
        )

        chunks = chunker.chunk(text)
        chunk_texts = [c.strip() for c in chunks]
        assert chunk_texts.count("PART II\n\nMore content.") == 1

    def test_no_detectors_same_as_before(self):
        text = "Hello world.\n\nSecond paragraph.\n\nThird paragraph."

        c1 = Chunker(target_size=30, overlap=0, min_size=0)
        c2 = Chunker(target_size=30, overlap=0, min_size=0, detectors=[])

        assert c1.chunk(text) == c2.chunk(text)


# ---------------------------------------------------------------------------
# KeepTogetherRegion integration — detectors prevent splitting
# ---------------------------------------------------------------------------


class TestKeepTogetherRegionIntegration:
    def _make_table_doc(self) -> str:
        """A document with a 'table' region in the middle."""
        return (
            "Introduction paragraph with context.\n\n"
            "Revenue  100  200  300\n"
            "Costs     50   80  120\n"
            "Profit    50  120  180\n\n"
            "Conclusion paragraph with analysis."
        )

    def test_keep_together_prevents_split(self):
        text = self._make_table_doc()

        # The table region (chars covering the 3 data lines)
        table_start = text.index("Revenue")
        table_end = text.index("180") + len("180")

        # Without protection: table might be split
        chunker_plain = Chunker(target_size=80, overlap=0, min_size=0)
        chunks_plain = chunker_plain.chunk(text)

        # With protection: table should stay together
        detector = FixedKeepTogetherDetector(
            [
                (table_start, table_end, "table: revenue", 2.0),
            ]
        )
        chunker_protected = Chunker(
            target_size=80,
            overlap=0,
            min_size=0,
            detectors=[detector],
        )
        chunks_protected = chunker_protected.chunk(text)

        # Find the chunk containing "Revenue" in each
        _table_chunk_plain = [c for c in chunks_plain if "Revenue" in c]
        table_chunk_protected = [c for c in chunks_protected if "Revenue" in c]

        assert len(table_chunk_protected) >= 1
        # The protected chunk should contain all three rows
        combined = table_chunk_protected[0]
        assert "Revenue" in combined
        assert "Costs" in combined
        assert "Profit" in combined

    def test_keep_together_respects_max_overshoot(self):
        """If a keep-together region exceeds max_overshoot * target_size,
        it falls back to normal splitting."""
        lines = [f"Row {i:3d}  {i * 100:5d}  {i * 200:5d}" for i in range(20)]
        text = "\n".join(lines)

        detector = FixedKeepTogetherDetector(
            [
                (0, len(text), "huge table", 1.2),
            ]
        )

        chunker = Chunker(
            target_size=100,
            overlap=0,
            min_size=0,
            detectors=[detector],
        )
        chunks = chunker.chunk(text)

        # Should still produce multiple chunks because the region is too large
        assert len(chunks) > 1

    def test_keep_together_with_split_detector(self):
        """SplitPoint and KeepTogetherRegion from different detectors
        work together correctly."""
        text = (
            "# Intro\n\nSome text.\n\n"
            "# Table Section\n\n"
            "A  10  20\n"
            "B  30  40\n"
            "C  50  60\n\n"
            "# Conclusion\n\nFinal words."
        )

        table_start = text.index("A  10")
        table_end = text.index("60") + len("60")

        class CombinedDetector(BoundaryDetector):
            def detect(self, text: str) -> list[Annotation]:
                results: list[Annotation] = []
                offset = 0
                for line_no, line in enumerate(text.split("\n")):
                    if line.startswith("# "):
                        results.append(SplitPoint(offset, line_no, line))
                    offset += len(line) + 1
                results.append(KeepTogetherRegion(table_start, table_end, "test table", 2.0))
                return results

        chunker = Chunker(
            target_size=60,
            overlap=0,
            min_size=0,
            detectors=[CombinedDetector()],
        )
        chunks = chunker.chunk(text)

        section_starts = [c.strip().split("\n")[0] for c in chunks]
        assert "# Intro" in section_starts
        assert "# Table Section" in section_starts
        assert "# Conclusion" in section_starts

        table_chunks = [c for c in chunks if "A  10" in c]
        assert len(table_chunks) >= 1
        assert "C  50  60" in table_chunks[0]

    def test_multiple_keep_together_regions(self):
        """Multiple keep-together regions in the same document."""
        text = (
            "Intro.\n\nT1A  1  2\nT1B  3  4\n\nMiddle paragraph.\n\nT2A  5  6\nT2B  7  8\n\nOutro."
        )

        t1_start = text.index("T1A")
        t1_end = text.index("T1B  3  4") + len("T1B  3  4")
        t2_start = text.index("T2A")
        t2_end = text.index("T2B  7  8") + len("T2B  7  8")

        detector = FixedKeepTogetherDetector(
            [
                (t1_start, t1_end, "table 1", 2.0),
                (t2_start, t2_end, "table 2", 2.0),
            ]
        )

        chunker = Chunker(
            target_size=30,
            overlap=0,
            min_size=0,
            detectors=[detector],
        )
        chunks = chunker.chunk(text)

        t1_chunks = [c for c in chunks if "T1A" in c]
        assert len(t1_chunks) >= 1
        assert "T1B" in t1_chunks[0]

        t2_chunks = [c for c in chunks if "T2A" in c]
        assert len(t2_chunks) >= 1
        assert "T2B" in t2_chunks[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDetectorEdgeCases:
    def test_empty_text(self):
        detector = FixedSplitDetector(["marker"])
        chunker = Chunker(target_size=100, detectors=[detector])
        assert chunker.chunk("") == []

    def test_detector_returns_empty(self):
        class EmptyDetector(BoundaryDetector):
            def detect(self, text: str) -> list[Annotation]:
                return []

        chunker = Chunker(target_size=100, detectors=[EmptyDetector()])
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_overlap_works_with_detectors(self):
        text = "First sentence. Second sentence.\n\nTHIRD\n\nFourth sentence."

        chunker = Chunker(
            target_size=30,
            overlap=1,
            overlap_unit="sentence",
            detectors=[FixedSplitDetector(["THIRD"])],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert any("THIRD" in c for c in chunks)

    def test_metadata_preserved_with_detectors(self):
        text = "Intro.\n\nMARKER\n\nContent."
        chunker = Chunker(
            target_size=30,
            overlap=0,
            detectors=[FixedSplitDetector(["MARKER"])],
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(text)
        assert all(hasattr(c, "start") for c in chunks)
        assert all(hasattr(c, "end") for c in chunks)
        assert all(hasattr(c, "boundary_type") for c in chunks)


class TestConcurrentDetectors:
    """Tests for concurrent=True detector fan-out."""

    def test_concurrent_same_results_as_serial(self):
        text = "Intro.\n\nHEADER_A\n\nBody of section A.\n\nHEADER_B\n\nBody of section B."
        detectors = [
            FixedSplitDetector(["HEADER_A"]),
            FixedSplitDetector(["HEADER_B"]),
        ]

        serial = Chunker(
            target_size=60,
            overlap=0,
            detectors=detectors,
            min_size=0,
            concurrent=False,
        )
        concurrent = Chunker(
            target_size=60,
            overlap=0,
            detectors=detectors,
            min_size=0,
            concurrent=True,
        )

        assert serial.chunk(text) == concurrent.chunk(text)

    def test_concurrent_metadata_matches_serial(self):
        text = "Intro.\n\nHEADER_A\n\nBody A.\n\nHEADER_B\n\nBody B."
        detectors = [
            FixedSplitDetector(["HEADER_A"]),
            FixedSplitDetector(["HEADER_B"]),
        ]

        serial_chunks = Chunker(
            target_size=60,
            overlap=0,
            detectors=detectors,
            min_size=0,
            concurrent=False,
        ).chunk_with_metadata(text)

        concurrent_chunks = Chunker(
            target_size=60,
            overlap=0,
            detectors=detectors,
            min_size=0,
            concurrent=True,
        ).chunk_with_metadata(text)

        assert len(serial_chunks) == len(concurrent_chunks)
        for s, c in zip(serial_chunks, concurrent_chunks):
            assert s.text == c.text
            assert s.start == c.start
            assert s.end == c.end

    def test_concurrent_with_single_detector(self):
        text = "Intro.\n\nMARKER\n\nContent."
        chunker = Chunker(
            target_size=30,
            overlap=0,
            detectors=[FixedSplitDetector(["MARKER"])],
            min_size=0,
            concurrent=True,
        )
        chunks = chunker.chunk(text)
        assert any("MARKER" in c for c in chunks)

    def test_concurrent_with_no_detectors(self):
        chunker = Chunker(target_size=100, concurrent=True)
        chunks = chunker.chunk("Hello world.")
        assert chunks == ["Hello world."]

    def test_concurrent_with_slow_detector(self):
        import time

        class SlowDetector(BoundaryDetector):
            def detect(self, text: str) -> list[Annotation]:
                time.sleep(0.1)
                return [SplitPoint(position=0, line_number=0, label="slow")]

        class FastDetector(BoundaryDetector):
            def detect(self, text: str) -> list[Annotation]:
                return []

        text = "Section A.\n\nSection B.\n\nSection C."
        start = time.monotonic()
        chunker = Chunker(
            target_size=100,
            overlap=0,
            detectors=[SlowDetector(), SlowDetector(), FastDetector()],
            concurrent=True,
        )
        chunker.chunk(text)
        elapsed = time.monotonic() - start
        # Two SlowDetectors at 0.1s each; concurrent should take ~0.1s not ~0.2s
        assert elapsed < 0.18


# ---------------------------------------------------------------------------
# Direct unit tests for HeadingDetector
# ---------------------------------------------------------------------------


class TestHeadingDetectorDirect:
    def test_detects_all_caps_heading(self):
        from chunkweaver.detector_heading import HeadingDetector

        text = (
            "\n"
            "DEFINITIONS\n"
            "\n"
            "The following terms shall have the meanings ascribed to them "
            "in this section of the agreement.\n"
        )
        hd = HeadingDetector(min_score=3.0)
        annotations = hd.detect(text)
        assert len(annotations) >= 1
        assert any("DEFINITIONS" in a.label for a in annotations)

    def test_detects_title_case_heading(self):
        from chunkweaver.detector_heading import HeadingDetector

        text = (
            "\n"
            "Risk Factors\n"
            "\n"
            "The company faces several material risks including market "
            "volatility, regulatory changes, competitive dynamics, and "
            "macroeconomic conditions that could materially affect the "
            "company's financial performance and future growth prospects.\n"
            "\n"
            "Liquidity and Capital Resources\n"
            "\n"
            "The company maintains adequate liquidity through its revolving "
            "credit facility and commercial paper program providing access "
            "to approximately five billion dollars across multiple sources.\n"
        )
        hd = HeadingDetector(min_score=3.0)
        annotations = hd.detect(text)
        assert len(annotations) >= 1
        labels = " ".join(a.label for a in annotations)
        assert "Liquidity" in labels or "Risk" in labels

    def test_rejects_numeric_line(self):
        from chunkweaver.detector_heading import HeadingDetector

        text = "12,345,678\n\nSome body text follows here.\n"
        hd = HeadingDetector(min_score=3.0)
        annotations = hd.detect(text)
        labels = [a.label for a in annotations]
        assert not any("12,345" in label for label in labels)

    def test_detect_with_scores_returns_candidates(self):
        from chunkweaver.detector_heading import HeadingDetector

        text = "\nOVERVIEW\n\nThis is the body text of the document.\n"
        hd = HeadingDetector(min_score=3.0)
        candidates = hd.detect_with_scores(text)
        assert len(candidates) >= 1
        assert candidates[0].score >= 3.0
        assert candidates[0].text == "OVERVIEW"

    def test_empty_text(self):
        from chunkweaver.detector_heading import HeadingDetector

        hd = HeadingDetector()
        assert hd.detect("") == []
        assert hd.detect_with_scores("") == []


# ---------------------------------------------------------------------------
# Direct unit tests for TableDetector
# ---------------------------------------------------------------------------


class TestTableDetectorDirect:
    def test_detects_numeric_table(self):
        from chunkweaver.detector_table import TableDetector

        lines = [
            "Revenue Summary",
            "2022  2023  2024",
            "Revenue    100,000  120,000  145,000",
            "Costs       80,000   90,000  105,000",
            "Profit      20,000   30,000   40,000",
            "Margin       20.0%    25.0%    27.6%",
            "",
            "Notes: All figures in USD thousands.",
        ]
        text = "\n".join(lines)
        td = TableDetector(min_data_lines=3)
        annotations = td.detect(text)
        assert len(annotations) >= 1
        assert all(isinstance(a, KeepTogetherRegion) for a in annotations)

    def test_detect_with_metadata_returns_regions(self):
        from chunkweaver.detector_table import TableDetector

        lines = [
            "Quarterly Results",
            "Q1      1,200   2,300",
            "Q2      1,400   2,500",
            "Q3      1,600   2,700",
            "Q4      1,800   2,900",
        ]
        text = "\n".join(lines)
        td = TableDetector(min_data_lines=3)
        regions = td.detect_with_metadata(text)
        assert len(regions) >= 1
        r = regions[0]
        assert r.num_data_lines >= 3
        assert r.start_char >= 0
        assert r.end_char > r.start_char

    def test_no_table_in_prose(self):
        from chunkweaver.detector_table import TableDetector

        text = (
            "The quick brown fox jumped over the lazy dog. "
            "This is ordinary prose with no numeric tables.\n"
        )
        td = TableDetector()
        assert td.detect(text) == []
        assert td.detect_with_metadata(text) == []

    def test_empty_text(self):
        from chunkweaver.detector_table import TableDetector

        td = TableDetector()
        assert td.detect("") == []
