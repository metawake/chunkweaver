"""Tests for customization features: sentence_pattern, keep_together, and
domain-specific presets.

These demonstrate how chunkweaver adapts to different text types.
"""

import re

from chunkweaver import SENTENCE_END_CJK, SENTENCE_END_PERMISSIVE, Chunker
from chunkweaver.presets import (
    CHAT,
    CLINICAL,
    FINANCIAL,
    FINANCIAL_TABLE,
    LEGAL_US,
    get_preset,
)
from chunkweaver.sentences import split_sentences

# -----------------------------------------------------------------------
# Custom sentence patterns
# -----------------------------------------------------------------------


class TestSentencePatternCJK:
    """Chinese/Japanese/Korean text uses different punctuation."""

    CJK_TEXT = "第一条规定了保护范围。第二条界定了适用条件。第三条明确了领土管辖权。"

    def test_default_pattern_does_not_split_cjk(self):
        parts = split_sentences(self.CJK_TEXT)
        assert len(parts) == 1

    def test_cjk_pattern_splits_correctly(self):
        parts = split_sentences(self.CJK_TEXT, pattern=SENTENCE_END_CJK)
        assert len(parts) == 3

    def test_chunker_with_cjk_sentence_pattern(self):
        chunker = Chunker(
            target_size=30,
            overlap=1,
            overlap_unit="sentence",
            sentence_pattern=SENTENCE_END_CJK,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.CJK_TEXT)
        assert len(chunks) > 1
        if len(chunks) > 1:
            assert chunks[1].overlap_text != ""

    def test_cjk_overlap_contains_previous_sentence(self):
        chunker = Chunker(
            target_size=40,
            overlap=1,
            overlap_unit="sentence",
            sentence_pattern=SENTENCE_END_CJK,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.CJK_TEXT)
        for c in chunks[1:]:
            if c.overlap_text:
                assert "。" in c.overlap_text


class TestSentencePatternChat:
    """Chat-style text with informal punctuation, no uppercase after periods."""

    CHAT_TEXT = (
        "hey how are you doing? "
        "im good thanks. "
        "what about the project? "
        "its almost done. "
        "nice! lets ship it."
    )

    def test_default_pattern_struggles_with_chat(self):
        parts = split_sentences(self.CHAT_TEXT)
        assert len(parts) < 5

    def test_permissive_pattern_splits_chat(self):
        parts = split_sentences(self.CHAT_TEXT, pattern=SENTENCE_END_PERMISSIVE)
        assert len(parts) >= 4

    def test_chunker_with_permissive_pattern(self):
        chunker = Chunker(
            target_size=60,
            overlap=1,
            overlap_unit="sentence",
            sentence_pattern=SENTENCE_END_PERMISSIVE,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.CHAT_TEXT)
        assert len(chunks) > 1

    def test_custom_string_pattern(self):
        chunker = Chunker(
            target_size=60,
            overlap=1,
            overlap_unit="sentence",
            sentence_pattern=r"([.!?])\s+",
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(self.CHAT_TEXT)
        assert len(chunks) > 1


class TestSentencePatternMixed:
    """Mixed-language text."""

    def test_compiled_pattern(self):
        pattern = re.compile(r"([.!?。！？])(\s*)")
        text = "English sentence. 日本語の文章。Another one!"
        parts = split_sentences(text, pattern=pattern)
        assert len(parts) >= 3


# -----------------------------------------------------------------------
# keep_together
# -----------------------------------------------------------------------


class TestKeepTogether:
    """Lines that must stay attached to what follows (table headers, labels)."""

    def test_table_header_stays_with_data(self):
        text = (
            "Article 1\nSome legal text about regulations.\n"
            "TABLE 1\nName | Value | Date\nAlpha | 100 | 2024\nBeta | 200 | 2025\n"
            "Article 2\nMore legal text."
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^Article\s+\d+", r"^TABLE\s+\d+"],
            keep_together=[r"^TABLE\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        table_chunk = [c for c in chunks if "TABLE 1" in c][0]
        assert "Name | Value" in table_chunk
        assert "Alpha | 100" in table_chunk

    def test_clinical_label_stays_with_content(self):
        text = (
            "CHIEF COMPLAINT:\nPatient presents with chest pain.\n"
            "HPI:\nOnset 2 hours ago while resting. No radiation.\n"
            "ASSESSMENT:\nAcute coronary syndrome suspected.\n"
            "PLAN:\nAdmit for observation. Serial troponins."
        )
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=CLINICAL,
            min_size=0,
        )
        chunks = chunker.chunk(text)
        for c in chunks:
            if "CHIEF COMPLAINT:" in c:
                assert "chest pain" in c
            if "PLAN:" in c:
                assert "troponins" in c

    def test_keep_together_respects_target_size(self):
        """If the combined result would exceed target_size, don't merge."""
        header = "TABLE 1\n"
        rows = "Row data. " * 200
        text = header + rows + "\nArticle 2\nShort."
        chunker = Chunker(
            target_size=200,
            overlap=0,
            boundaries=[r"^TABLE\s+\d+", r"^Article\s+\d+"],
            keep_together=[r"^TABLE\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_keep_together_empty_list(self):
        text = "Some text here."
        chunker = Chunker(target_size=5000, keep_together=[])
        chunks = chunker.chunk(text)
        assert chunks == ["Some text here."]

    def test_keep_together_no_match(self):
        text = "# Title\nContent.\n# Next\nMore content."
        chunker = Chunker(
            target_size=5000,
            overlap=0,
            boundaries=[r"^#{1,6}\s"],
            keep_together=[r"^TABLE\s+\d+"],
            min_size=0,
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


# -----------------------------------------------------------------------
# Domain presets
# -----------------------------------------------------------------------


class TestLegalUSPreset:
    def test_section_numbering(self):
        text = (
            "Section 1\nThis Act may be cited as the Example Act.\n"
            "Section 2\nDefinitions used in this Act.\n"
            "Section 3\nEnforcement provisions.\n"
        )
        chunker = Chunker(target_size=5000, overlap=0, boundaries=LEGAL_US, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_paragraph_sign(self):
        text = "§ 12 Scope of regulation.\n§ 13 Definitions.\n§ 14 Enforcement."
        chunker = Chunker(target_size=5000, overlap=0, boundaries=LEGAL_US, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_contract_whereas(self):
        text = (
            "WHEREAS, the parties wish to enter into an agreement;\n"
            "WHEREAS, the terms have been negotiated in good faith;\n"
            "NOW, THEREFORE the parties agree as follows:\n"
            "1.1 Definitions.\n"
        )
        chunker = Chunker(target_size=5000, overlap=0, boundaries=LEGAL_US, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3


class TestChatPreset:
    def test_timestamp_boundaries(self):
        text = (
            "[14:30] Alice: Hey, how's the project going?\n"
            "[14:31] Bob: Almost done with the API.\n"
            "[14:32] Alice: Great, let's deploy tomorrow.\n"
        )
        chunker = Chunker(target_size=5000, overlap=0, boundaries=CHAT, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_iso_timestamp(self):
        text = "2024-01-15 14:30 Alice: First message.\n2024-01-15 14:31 Bob: Second message.\n"
        chunker = Chunker(target_size=5000, overlap=0, boundaries=CHAT, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_speaker_colon(self):
        text = "Agent: Hello, how can I help?\nCustomer: I have a billing issue.\nAgent: Let me check."
        chunker = Chunker(target_size=5000, overlap=0, boundaries=CHAT, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 3


class TestClinicalPreset:
    DISCHARGE_NOTE = (
        "CHIEF COMPLAINT: Chest pain and shortness of breath.\n"
        "HPI: 65-year-old male presenting with acute onset chest pain "
        "2 hours prior to admission. Pain is substernal, non-radiating.\n"
        "PAST MEDICAL HISTORY: Hypertension, Type 2 DM.\n"
        "MEDICATIONS: Metformin 500mg BID, Lisinopril 10mg daily.\n"
        "ALLERGIES: NKDA.\n"
        "PHYSICAL EXAM: BP 145/90, HR 88, RR 18.\n"
        "ASSESSMENT: Acute coronary syndrome, rule out MI.\n"
        "PLAN: Admit to telemetry. Serial troponins q6h. "
        "Cardiology consult in AM.\n"
    )

    def test_sections_split_correctly(self):
        chunker = Chunker(target_size=5000, overlap=0, boundaries=CLINICAL, min_size=0)
        chunks = chunker.chunk(self.DISCHARGE_NOTE)
        assert len(chunks) >= 7

    def test_assessment_stays_intact(self):
        chunker = Chunker(target_size=5000, overlap=0, boundaries=CLINICAL, min_size=0)
        chunks = chunker.chunk(self.DISCHARGE_NOTE)
        assessment = [c for c in chunks if "ASSESSMENT:" in c]
        assert len(assessment) == 1
        assert "rule out MI" in assessment[0]


class TestFinancialPreset:
    SEC_FILING = (
        "PART I\n\n"
        "Item 1. Business\n"
        "The Company operates globally in financial services.\n\n"
        "Item 1A. Risk Factors\n"
        "Investing involves risks including market volatility.\n\n"
        "PART II\n\n"
        "Item 5. Market Information\n"
        "Common stock is listed on NYSE.\n\n"
        "NOTE 1 Revenue Recognition\n"
        "Revenue is recognized when performance obligations are satisfied.\n"
    )

    def test_sec_filing_structure(self):
        chunker = Chunker(target_size=5000, overlap=0, boundaries=FINANCIAL, min_size=0)
        chunks = chunker.chunk(self.SEC_FILING)
        assert len(chunks) >= 5

    def test_table_boundaries(self):
        text = (
            "TABLE 1\nRevenue | 2023 | 2024\nProduct A | 100 | 120\n"
            "TABLE 2\nExpenses | 2023 | 2024\nSalaries | 50 | 55\n"
        )
        chunker = Chunker(target_size=5000, overlap=0, boundaries=FINANCIAL_TABLE, min_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


class TestGetPresetNewEntries:
    def test_all_presets_accessible(self):
        for name in (
            "legal-eu",
            "legal-us",
            "rfc",
            "markdown",
            "chat",
            "clinical",
            "financial",
            "financial-table",
            "plain",
        ):
            result = get_preset(name)
            assert isinstance(result, list)

    def test_underscore_variant_new_presets(self):
        assert get_preset("legal_us") == LEGAL_US
        assert get_preset("financial_table") == FINANCIAL_TABLE


# -----------------------------------------------------------------------
# Realistic end-to-end scenarios
# -----------------------------------------------------------------------


class TestRealisticScenarios:
    def test_healthcare_vectordb_ingest(self):
        """Full pipeline: clinical note -> chunks -> vector DB records."""
        note = (
            "CHIEF COMPLAINT: Abdominal pain.\n"
            "HPI: 45-year-old female with 3 days of epigastric pain.\n"
            "ASSESSMENT: Possible cholecystitis.\n"
            "PLAN: Ultrasound of gallbladder. NPO. IV fluids.\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=CLINICAL,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(note)

        records = [
            {
                "id": f"note-42-chunk-{c.index}",
                "text": c.text,
                "metadata": {"section": c.boundary_type, "start": c.start},
            }
            for c in chunks
        ]
        assert len(records) >= 3
        assert all(r["text"] for r in records)

    def test_chat_log_chunking_with_overlap(self):
        """Chat logs chunked by speaker turn with sentence overlap."""
        log = (
            "Agent: Welcome to support. How can I help you today?\n"
            "Customer: My order hasn't arrived. It's been 10 days.\n"
            "Agent: I'm sorry to hear that. Let me look into it.\n"
            "Customer: The order number is 12345.\n"
            "Agent: I see the order was shipped on Jan 5. It appears delayed.\n"
        )
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=CHAT,
            sentence_pattern=SENTENCE_END_PERMISSIVE,
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(log)
        assert len(chunks) == 5

    def test_mixed_language_document(self):
        text = (
            "# Overview\nThis is English. 这是中文。And back to English.\n# Details\nMore content."
        )
        chunker = Chunker(
            target_size=5000,
            overlap=1,
            overlap_unit="sentence",
            boundaries=[r"^#{1,6}\s"],
            sentence_pattern=re.compile(r"([.!?。！？])(\s*)"),
            min_size=0,
        )
        chunks = chunker.chunk_with_metadata(text)
        assert len(chunks) == 2
        assert chunks[1].overlap_text != ""
