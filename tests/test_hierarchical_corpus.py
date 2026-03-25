"""Cross-domain validation of hierarchical chunking on real documents.

Runs hierarchical vs flat chunking on the benchmark corpus
(GDPR, EU AI Act, CCPA, 8 IETF RFCs, FDA drug label, SEC 10-K)
and validates that hierarchy produces stable, coherent, correctly-sized
results across legal, technical, medical, and financial domains.

Corpus: benchmarks/corpus/
"""

import os
import statistics

import pytest

from chunkweaver import Chunker
from chunkweaver.boundaries import BoundarySpec, detect_boundaries
from chunkweaver.presets import (
    FDA_LABEL,
    FDA_LABEL_LEVELED,
    LEGAL_EU,
    LEGAL_EU_LEVELED,
    RFC,
    RFC_LEVELED,
    SEC_10K,
    SEC_10K_LEVELED,
)

CORPUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "corpus",
)

# The real GDPR has leading whitespace on CHAPTER lines, so we adjust
LEGAL_EU_REAL = [
    r"^\s*CHAPTER\s+[IVX\d]+",
    r"^Article\s+\d+",
    r"^\(\d+\)\s+",
]

LEGAL_EU_REAL_LEVELED: list[BoundarySpec] = [
    (r"^\s*CHAPTER\s+[IVX\d]+", 0),
    (r"^Article\s+\d+", 1),
    (r"^\(\d+\)\s+", 2),
]


def _load(name: str) -> str:
    path = os.path.join(CORPUS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


# -----------------------------------------------------------------------
# Helpers for chunk quality assertions
# -----------------------------------------------------------------------

def _chunk_stats(chunks):
    sizes = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "min": min(sizes),
        "max": max(sizes),
        "mean": int(statistics.mean(sizes)),
        "median": int(statistics.median(sizes)),
        "empty": sum(1 for c in chunks if not c.strip()),
    }


# -----------------------------------------------------------------------
# GDPR (373K chars, 11 chapters, 99 articles)
# -----------------------------------------------------------------------

class TestGDPRHierarchical:
    @pytest.fixture(autouse=True)
    def load_gdpr(self):
        self.text = _load("eu_gdpr_2016_679.txt")

    def test_flat_vs_hierarchical_chunk_count(self):
        """Hierarchical should produce fewer chunks at large target_size."""
        flat = Chunker(target_size=4096, overlap=0, boundaries=LEGAL_EU_REAL, min_size=0)
        hier = Chunker(target_size=4096, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=0)

        flat_chunks = flat.chunk(self.text)
        hier_chunks = hier.chunk(self.text)

        # Hierarchical keeps chapters intact when they fit → fewer chunks
        assert len(hier_chunks) < len(flat_chunks)

    def test_hierarchical_no_empty_chunks(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=100)
        chunks = hier.chunk(self.text)
        assert all(c.strip() for c in chunks)

    def test_hierarchical_respects_target_size(self):
        """Oversized chunks should be limited to reasonable overshoot."""
        hier = Chunker(target_size=2048, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=100)
        chunks = hier.chunk(self.text)
        for c in chunks:
            assert len(c) <= 2048 * 2, f"Chunk too large: {len(c)} chars"

    def test_hierarchical_fewer_than_flat_at_large_target(self):
        """At 8K target, hierarchy should produce fewer chunks than flat."""
        hier = Chunker(target_size=8192, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=0)
        flat = Chunker(target_size=8192, overlap=0, boundaries=LEGAL_EU_REAL, min_size=0)
        hier_n = len(hier.chunk(self.text))
        flat_n = len(flat.chunk(self.text))
        assert hier_n < flat_n

    def test_hierarchical_articles_split_at_small_target(self):
        """At 1K target, articles should be split."""
        hier = Chunker(target_size=1024, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=0)
        chunks = hier.chunk(self.text)
        # More chunks than articles means articles got subsplit
        assert len(chunks) > 50

    def test_full_text_preserved(self):
        """Non-overlap content should cover the full original text."""
        hier = Chunker(target_size=2048, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=0)
        chunks = hier.chunk_with_metadata(self.text)
        reconstructed = "".join(c.content_text for c in chunks)
        assert reconstructed == self.text

    def test_overlap_works_on_real_doc(self):
        hier = Chunker(
            target_size=2048, overlap=2, overlap_unit="sentence",
            boundaries=LEGAL_EU_REAL_LEVELED, min_size=100,
        )
        chunks = hier.chunk_with_metadata(self.text)
        overlap_chunks = [c for c in chunks[1:] if c.overlap_text]
        assert len(overlap_chunks) > 0

    def test_boundary_levels_populated(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=0)
        chunks = hier.chunk_with_metadata(self.text)
        levels = {c.boundary_level for c in chunks}
        assert len(levels) >= 2, f"Expected multiple levels, got {levels}"


# -----------------------------------------------------------------------
# EU AI Act (655K chars — largest document)
# -----------------------------------------------------------------------

class TestAIActHierarchical:
    @pytest.fixture(autouse=True)
    def load_ai_act(self):
        self.text = _load("eu_ai_act_2024_1689.txt")

    def test_hierarchical_handles_large_document(self):
        """655K chars should chunk without errors."""
        hier = Chunker(target_size=2048, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=100)
        chunks = hier.chunk(self.text)
        assert len(chunks) > 50
        assert all(c.strip() for c in chunks)

    def test_flat_vs_hierarchical_difference(self):
        flat = Chunker(target_size=4096, overlap=0, boundaries=LEGAL_EU_REAL, min_size=0)
        hier = Chunker(target_size=4096, overlap=0, boundaries=LEGAL_EU_REAL_LEVELED, min_size=0)
        assert len(hier.chunk(self.text)) < len(flat.chunk(self.text))


# -----------------------------------------------------------------------
# CCPA (172K chars — US legal, different section structure)
# -----------------------------------------------------------------------

class TestCCPAHierarchical:
    CCPA_LEVELED: list[BoundarySpec] = [
        (r"^1798\.\d+", 0),
        (r"^\([a-z]\)", 1),
        (r"^\(\d+\)", 2),
    ]

    @pytest.fixture(autouse=True)
    def load_ccpa(self):
        self.text = _load("ccpa_1798.txt")

    def test_section_boundaries_detected(self):
        matches = detect_boundaries(self.text, self.CCPA_LEVELED)
        level_0 = [m for m in matches if m.level == 0]
        assert len(level_0) >= 15, f"Expected 15+ section matches, got {len(level_0)}"

    def test_hierarchical_chunks_stable(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=self.CCPA_LEVELED, min_size=0)
        chunks = hier.chunk(self.text)
        assert len(chunks) > 10
        assert all(c.strip() for c in chunks)

    def test_text_preserved(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=self.CCPA_LEVELED, min_size=0)
        chunks = hier.chunk_with_metadata(self.text)
        reconstructed = "".join(c.content_text for c in chunks)
        assert reconstructed == self.text


# -----------------------------------------------------------------------
# RFCs (8 documents, 63K–422K chars)
# -----------------------------------------------------------------------

RFC_FILES = [
    "rfc7519_jwt.txt",
    "rfc6749_oauth2.txt",
    "rfc8446_tls13.txt",
    "rfc5246_tls12.txt",
    "rfc2616_http11.txt",
    "rfc7231_http_semantics.txt",
    "rfc7540_http2.txt",
    "rfc6455_websocket.txt",
]


class TestRFCHierarchical:
    @pytest.mark.parametrize("filename", RFC_FILES)
    def test_hierarchical_fewer_chunks_than_flat(self, filename):
        """Hierarchical should keep subsections inside sections when they fit."""
        text = _load(filename)
        flat = Chunker(target_size=4096, overlap=0, boundaries=RFC, min_size=0)
        hier = Chunker(target_size=4096, overlap=0, boundaries=RFC_LEVELED, min_size=0)

        flat_count = len(flat.chunk(text))
        hier_count = len(hier.chunk(text))
        # Hierarchical should have equal or fewer chunks
        assert hier_count <= flat_count, (
            f"{filename}: hier={hier_count} > flat={flat_count}"
        )

    @pytest.mark.parametrize("filename", RFC_FILES)
    def test_no_empty_chunks(self, filename):
        text = _load(filename)
        hier = Chunker(target_size=2048, overlap=0, boundaries=RFC_LEVELED, min_size=100)
        chunks = hier.chunk(text)
        assert all(c.strip() for c in chunks), f"{filename}: has empty chunks"

    @pytest.mark.parametrize("filename", RFC_FILES)
    def test_text_preserved(self, filename):
        """Text is perfectly preserved after word-split fix."""
        text = _load(filename)
        hier = Chunker(target_size=2048, overlap=0, boundaries=RFC_LEVELED, min_size=0)
        chunks = hier.chunk_with_metadata(text)
        reconstructed = "".join(c.content_text for c in chunks)
        assert reconstructed == text, f"{filename}: text not preserved"

    @pytest.mark.parametrize("filename", RFC_FILES)
    def test_target_size_respected(self, filename):
        text = _load(filename)
        hier = Chunker(target_size=2048, overlap=0, boundaries=RFC_LEVELED, min_size=100)
        chunks = hier.chunk(text)
        for i, c in enumerate(chunks):
            assert len(c) <= 2048 * 2, (
                f"{filename} chunk {i}: {len(c)} chars > 4096 limit"
            )


# -----------------------------------------------------------------------
# FDA Drug Label — medical/pharma (42K chars, 15 sections, 23 subsections)
# -----------------------------------------------------------------------

class TestFDALabelHierarchical:
    @pytest.fixture(autouse=True)
    def load_fda(self):
        self.text = _load("fda_metformin_label.txt")

    def test_boundaries_detected(self):
        matches = detect_boundaries(self.text, FDA_LABEL_LEVELED)
        level_0 = [m for m in matches if m.level == 0]
        level_1 = [m for m in matches if m.level == 1]
        assert len(level_0) >= 10, f"Expected 10+ section matches, got {len(level_0)}"
        assert len(level_1) >= 15, f"Expected 15+ subsection matches, got {len(level_1)}"

    def test_flat_vs_hierarchical(self):
        flat = Chunker(target_size=4096, overlap=0, boundaries=FDA_LABEL, min_size=0)
        hier = Chunker(target_size=4096, overlap=0, boundaries=FDA_LABEL_LEVELED, min_size=0)
        flat_n = len(flat.chunk(self.text))
        hier_n = len(hier.chunk(self.text))
        assert hier_n <= flat_n

    def test_no_empty_chunks(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=FDA_LABEL_LEVELED, min_size=100)
        chunks = hier.chunk(self.text)
        assert all(c.strip() for c in chunks)

    def test_text_preserved(self):
        for target in [1024, 2048, 4096]:
            hier = Chunker(
                target_size=target, overlap=0,
                boundaries=FDA_LABEL_LEVELED, min_size=0,
            )
            chunks = hier.chunk_with_metadata(self.text)
            reconstructed = "".join(c.content_text for c in chunks)
            assert reconstructed == self.text, f"target={target}: text not preserved"

    def test_sections_stay_intact_at_large_target(self):
        """At 8K most FDA sections fit in one chunk."""
        hier = Chunker(target_size=8192, overlap=0, boundaries=FDA_LABEL_LEVELED, min_size=0)
        chunks = hier.chunk(self.text)
        # 42K doc at 8K target → roughly 5-8 chunks when sections are kept intact
        assert len(chunks) < 20

    def test_subsections_split_at_small_target(self):
        hier = Chunker(target_size=1024, overlap=0, boundaries=FDA_LABEL_LEVELED, min_size=0)
        chunks = hier.chunk(self.text)
        assert len(chunks) > 15

    def test_boundary_levels_populated(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=FDA_LABEL_LEVELED, min_size=0)
        chunks = hier.chunk_with_metadata(self.text)
        levels = {c.boundary_level for c in chunks}
        assert len(levels) >= 2

    def test_clinical_content_coherence(self):
        """Key clinical sections should not be split mid-content at moderate target."""
        hier = Chunker(target_size=4096, overlap=0, boundaries=FDA_LABEL_LEVELED, min_size=0)
        chunks = hier.chunk(self.text)
        # CONTRAINDICATIONS section is short (~300 chars) — should be in one chunk
        contra_chunks = [c for c in chunks if "CONTRAINDICATIONS" in c[:50]]
        assert len(contra_chunks) <= 1, "CONTRAINDICATIONS split across chunks"


# -----------------------------------------------------------------------
# SEC 10-K — financial (275K chars, 4 PARTs, 14 Items, ~40 sub-headings)
# -----------------------------------------------------------------------

class TestSEC10KHierarchical:
    @pytest.fixture(autouse=True)
    def load_10k(self):
        self.text = _load("sec_enron_10k_2000.txt")

    def test_boundaries_detected(self):
        matches = detect_boundaries(self.text, SEC_10K_LEVELED)
        parts = [m for m in matches if m.level == 0]
        items = [m for m in matches if m.level == 1]
        subheadings = [m for m in matches if m.level == 2]
        assert len(parts) >= 4, f"Expected 4+ PART matches, got {len(parts)}"
        assert len(items) >= 14, f"Expected 14+ Item matches, got {len(items)}"
        assert len(subheadings) >= 10, f"Expected 10+ sub-heading matches, got {len(subheadings)}"

    def test_flat_vs_hierarchical(self):
        flat = Chunker(target_size=4096, overlap=0, boundaries=SEC_10K, min_size=0)
        hier = Chunker(target_size=4096, overlap=0, boundaries=SEC_10K_LEVELED, min_size=0)
        flat_n = len(flat.chunk(self.text))
        hier_n = len(hier.chunk(self.text))
        assert hier_n <= flat_n

    def test_no_empty_chunks(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=SEC_10K_LEVELED, min_size=100)
        chunks = hier.chunk(self.text)
        assert all(c.strip() for c in chunks)

    def test_text_preserved(self):
        for target in [1024, 2048, 4096]:
            hier = Chunker(
                target_size=target, overlap=0,
                boundaries=SEC_10K_LEVELED, min_size=0,
            )
            chunks = hier.chunk_with_metadata(self.text)
            reconstructed = "".join(c.content_text for c in chunks)
            assert reconstructed == self.text, f"target={target}: text not preserved"

    def test_target_size_respected(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=SEC_10K_LEVELED, min_size=100)
        chunks = hier.chunk(self.text)
        for i, c in enumerate(chunks):
            assert len(c) <= 2048 * 2, f"chunk {i}: {len(c)} chars > 4096 limit"

    def test_boundary_levels_populated(self):
        hier = Chunker(target_size=2048, overlap=0, boundaries=SEC_10K_LEVELED, min_size=0)
        chunks = hier.chunk_with_metadata(self.text)
        levels = {c.boundary_level for c in chunks}
        assert len(levels) >= 2

    def test_hierarchy_groups_items_into_parts(self):
        """Hierarchical should produce fewer chunks than flat at same target."""
        flat = Chunker(target_size=16384, overlap=0, boundaries=SEC_10K, min_size=0)
        hier = Chunker(target_size=16384, overlap=0, boundaries=SEC_10K_LEVELED, min_size=0)
        flat_n = len(flat.chunk(self.text))
        hier_n = len(hier.chunk(self.text))
        assert hier_n <= flat_n

    def test_overlap_works(self):
        hier = Chunker(
            target_size=2048, overlap=2, overlap_unit="sentence",
            boundaries=SEC_10K_LEVELED, min_size=100,
        )
        chunks = hier.chunk_with_metadata(self.text)
        overlap_chunks = [c for c in chunks[1:] if c.overlap_text]
        assert len(overlap_chunks) > 0


# -----------------------------------------------------------------------
# Cross-domain comparison: hierarchy value by document type
# -----------------------------------------------------------------------

class TestCrossDomainComparison:
    def test_hierarchy_reduction_across_all_domains(self):
        """Hierarchy should not increase chunk count on any domain."""
        results = []

        configs = [
            ("GDPR", _load("eu_gdpr_2016_679.txt"), LEGAL_EU_REAL, LEGAL_EU_REAL_LEVELED),
            ("RFC-JWT", _load("rfc7519_jwt.txt"), RFC, RFC_LEVELED),
            ("RFC-OAuth2", _load("rfc6749_oauth2.txt"), RFC, RFC_LEVELED),
            ("RFC-TLS13", _load("rfc8446_tls13.txt"), RFC, RFC_LEVELED),
            ("FDA-Metformin", _load("fda_metformin_label.txt"), FDA_LABEL, FDA_LABEL_LEVELED),
            ("SEC-10K-Enron", _load("sec_enron_10k_2000.txt"), SEC_10K, SEC_10K_LEVELED),
        ]

        for name, text, flat_b, hier_b in configs:
            flat_n = len(Chunker(target_size=4096, overlap=0, boundaries=flat_b, min_size=0).chunk(text))
            hier_n = len(Chunker(target_size=4096, overlap=0, boundaries=hier_b, min_size=0).chunk(text))
            reduction = 1 - hier_n / flat_n if flat_n > 0 else 0
            results.append((name, flat_n, hier_n, reduction))

        for name, flat_n, hier_n, reduction in results:
            assert hier_n <= flat_n, (
                f"{name}: hierarchy produced more chunks ({hier_n} > {flat_n})"
            )

        any_reduced = any(r > 0.05 for _, _, _, r in results)
        assert any_reduced, (
            f"No domain showed >5% reduction: {results}"
        )

    def test_text_preserved_all_domains(self):
        """Text perfectly preserved across all domains and target sizes."""
        configs = [
            ("GDPR", _load("eu_gdpr_2016_679.txt"), LEGAL_EU_REAL_LEVELED),
            ("RFC-JWT", _load("rfc7519_jwt.txt"), RFC_LEVELED),
            ("RFC-OAuth2", _load("rfc6749_oauth2.txt"), RFC_LEVELED),
            ("CCPA", _load("ccpa_1798.txt"), [(r"^1798\.\d+", 0), (r"^\([a-z]\)", 1)]),
            ("FDA-Metformin", _load("fda_metformin_label.txt"), FDA_LABEL_LEVELED),
            ("SEC-10K-Enron", _load("sec_enron_10k_2000.txt"), SEC_10K_LEVELED),
        ]

        for name, text, boundaries in configs:
            for target in [1024, 2048, 4096]:
                chunker = Chunker(
                    target_size=target, overlap=0,
                    boundaries=boundaries, min_size=0,
                )
                chunks = chunker.chunk_with_metadata(text)
                reconstructed = "".join(c.content_text for c in chunks)
                assert reconstructed == text, (
                    f"{name} at target={target}: text not preserved"
                )
