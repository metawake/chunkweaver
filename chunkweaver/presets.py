"""Built-in boundary pattern presets for common document types.

Each preset is a list of regex strings matched against individual lines.
Combine presets freely: ``boundaries = LEGAL_EU + FINANCIAL_TABLE``.

**Leveled presets** (``_LEVELED`` suffix) assign hierarchy levels to each
pattern.  Level 0 boundaries always split; level 1+ boundaries only split
when the parent segment exceeds ``target_size``.  This keeps entire
sections intact when they fit and progressively refines when they don't.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

from chunkweaver.boundaries import BoundarySpec

# ---------------------------------------------------------------------------
# Legal
# ---------------------------------------------------------------------------

LEGAL_EU: List[str] = [
    r"^Article\s+\d+",
    r"^\(\d+\)\s+",
    r"^CHAPTER\s+[IVX\d]+",
    r"^SECTION\s+\d+",
]

LEGAL_US: List[str] = [
    r"^§\s*\d+",                       # § 12, § 302(a)
    r"^Section\s+\d+",                 # Section 1, Section 302
    r"^PART\s+[IVX\d]+",              # PART I, PART 2
    r"^WHEREAS[,\s]",                  # contract recitals
    r"^NOW,?\s+THEREFORE",            # contract transition
    r"^\d+\.\d+\s+\S",               # numbered clauses: 1.1 Definitions
]

# ---------------------------------------------------------------------------
# Technical
# ---------------------------------------------------------------------------

RFC: List[str] = [
    r"^\d+\.\s+\S",
    r"^\d+\.\d+\.?\s+\S",
    r"^Appendix\s+[A-Z]",
]

MARKDOWN: List[str] = [
    r"^#{1,6}\s",
    r"^---\s*$",
]

# ---------------------------------------------------------------------------
# Chat / conversational
# ---------------------------------------------------------------------------

CHAT: List[str] = [
    r"^\[\d{1,2}:\d{2}(:\d{2})?\]",                 # [14:30] or [14:30:05]
    r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}",          # 2024-01-15 14:30
    r"^\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}",    # 1/15/24 14:30
    r"^[A-Za-z0-9_.\-]+\s*:",                        # username: message
]

# ---------------------------------------------------------------------------
# Healthcare / clinical
# ---------------------------------------------------------------------------

CLINICAL: List[str] = [
    r"^(CHIEF COMPLAINT|CC)\s*:",
    r"^(HISTORY OF PRESENT ILLNESS|HPI)\s*:",
    r"^(PAST MEDICAL HISTORY|PMH)\s*:",
    r"^(MEDICATIONS|MEDS)\s*:",
    r"^(ALLERGIES)\s*:",
    r"^(REVIEW OF SYSTEMS|ROS)\s*:",
    r"^(PHYSICAL EXAM|PE|EXAMINATION)\s*:",
    r"^(ASSESSMENT|IMPRESSION)\s*:",
    r"^(PLAN)\s*:",
    r"^(LAB(ORATORY)?\s*(RESULTS|DATA)?)\s*:",
    r"^(VITAL SIGNS|VITALS)\s*:",
    r"^(DISCHARGE\s*(SUMMARY|INSTRUCTIONS|DIAGNOSIS))\s*:",
    r"^(PROCEDURE|OPERATIVE)\s*(NOTE|REPORT)?\s*:",
]

# ---------------------------------------------------------------------------
# Financial
# ---------------------------------------------------------------------------

FINANCIAL: List[str] = [
    r"^Item\s+\d+[A-Z]?\.?\s",         # SEC filing: Item 1, Item 1A.
    r"^PART\s+[IVX]+\b",               # SEC filing: PART I, PART II
    r"^NOTE\s+\d+",                     # NOTE 1 – Revenue Recognition
    r"^TABLE\s+\d+",                    # TABLE 1
    r"^Schedule\s+[A-Z\d]+",           # Schedule A, Schedule 14A
]

FINANCIAL_TABLE: List[str] = [
    r"^TABLE\s+\d+",
    r"^\|[-\s|]+\|",                    # Markdown table separator row
    r"^[-+]{3,}",                       # ASCII table separator
]

SEC_10K: List[str] = [
    r"^\s*PART\s+[IVX]+\b",            # PART I, PART II (often indented)
    r"^[Ii][Tt][Ee][Mm]\s+\d+[A-Z]?\.?\s",  # Item 1., ITEM 7A. (mixed case)
    r"^\s{0,5}[A-Z][A-Z ]{5,}$",       # ALL-CAPS sub-headings
]

# ---------------------------------------------------------------------------
# Healthcare / pharma
# ---------------------------------------------------------------------------

FDA_LABEL: List[str] = [
    r"^\d+\s+[A-Z]",                   # 1 INDICATIONS AND USAGE
    r"^##\s+\d+\.\d+\s+",             # ## 2.1 Adult Dosage
]

# ---------------------------------------------------------------------------
# No boundaries — pure paragraph/sentence fallback
# ---------------------------------------------------------------------------

PLAIN: List[str] = []

# ---------------------------------------------------------------------------
# Leveled presets — hierarchical boundary priorities
# ---------------------------------------------------------------------------

LEGAL_EU_LEVELED: List[BoundarySpec] = [
    (r"^CHAPTER\s+[IVX\d]+", 0),
    (r"^SECTION\s+\d+", 1),
    (r"^Article\s+\d+", 2),
    (r"^\(\d+\)\s+", 3),
]

LEGAL_US_LEVELED: List[BoundarySpec] = [
    (r"^PART\s+[IVX\d]+", 0),
    (r"^§\s*\d+", 1),
    (r"^Section\s+\d+", 1),
    (r"^WHEREAS[,\s]", 1),
    (r"^NOW,?\s+THEREFORE", 1),
    (r"^\d+\.\d+\s+\S", 2),
]

RFC_LEVELED: List[BoundarySpec] = [
    (r"^\d+\.\s+\S", 0),
    (r"^\d+\.\d+\.?\s+\S", 1),
    (r"^Appendix\s+[A-Z]", 0),
]

MARKDOWN_LEVELED: List[BoundarySpec] = [
    (r"^#\s", 0),
    (r"^##\s", 1),
    (r"^###\s", 2),
    (r"^#{4,6}\s", 3),
    (r"^---\s*$", 0),
]

FINANCIAL_LEVELED: List[BoundarySpec] = [
    (r"^PART\s+[IVX]+\b", 0),
    (r"^Item\s+\d+[A-Z]?\.?\s", 1),
    (r"^NOTE\s+\d+", 1),
    (r"^Schedule\s+[A-Z\d]+", 1),
    (r"^TABLE\s+\d+", 2),
]

SEC_10K_LEVELED: List[BoundarySpec] = [
    (r"^\s*PART\s+[IVX]+\b", 0),
    (r"^[Ii][Tt][Ee][Mm]\s+\d+[A-Z]?\.?\s", 1),
    (r"^\s{0,5}[A-Z][A-Z ]{5,}$", 2),
]

FDA_LABEL_LEVELED: List[BoundarySpec] = [
    (r"^\d+\s+[A-Z]", 0),
    (r"^##\s+\d+\.\d+\s+", 1),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESETS: Dict[str, List[BoundarySpec]] = {
    "legal-eu": LEGAL_EU,
    "legal-us": LEGAL_US,
    "rfc": RFC,
    "markdown": MARKDOWN,
    "chat": CHAT,
    "clinical": CLINICAL,
    "financial": FINANCIAL,
    "financial-table": FINANCIAL_TABLE,
    "sec-10k": SEC_10K,
    "fda-label": FDA_LABEL,
    "plain": PLAIN,
}

PRESETS_LEVELED: Dict[str, List[BoundarySpec]] = {
    "legal-eu": LEGAL_EU_LEVELED,
    "legal-us": LEGAL_US_LEVELED,
    "rfc": RFC_LEVELED,
    "markdown": MARKDOWN_LEVELED,
    "financial": FINANCIAL_LEVELED,
    "sec-10k": SEC_10K_LEVELED,
    "fda-label": FDA_LABEL_LEVELED,
}


def get_preset(name: str, leveled: bool = False) -> List[BoundarySpec]:
    """Return boundary patterns for a named preset.

    Args:
        name: Preset name (e.g. ``"legal-eu"``, ``"rfc"``).
        leveled: If ``True``, return the hierarchical version when
                 available.  Falls back to the flat preset otherwise.

    Raises ``ValueError`` for unknown preset names.
    """
    key = name.lower().replace("_", "-")
    if leveled and key in PRESETS_LEVELED:
        return list(PRESETS_LEVELED[key])
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return list(PRESETS[key])
