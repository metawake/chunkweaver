"""Built-in boundary pattern presets for common document types.

Each preset is a list of regex strings matched against individual lines.
Combine presets freely: ``boundaries = LEGAL_EU + FINANCIAL_TABLE``.
"""

from __future__ import annotations

from typing import Dict, List

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

# ---------------------------------------------------------------------------
# No boundaries — pure paragraph/sentence fallback
# ---------------------------------------------------------------------------

PLAIN: List[str] = []

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESETS: Dict[str, List[str]] = {
    "legal-eu": LEGAL_EU,
    "legal-us": LEGAL_US,
    "rfc": RFC,
    "markdown": MARKDOWN,
    "chat": CHAT,
    "clinical": CLINICAL,
    "financial": FINANCIAL,
    "financial-table": FINANCIAL_TABLE,
    "plain": PLAIN,
}


def get_preset(name: str) -> List[str]:
    """Return boundary patterns for a named preset.

    Raises ``ValueError`` for unknown preset names.
    """
    key = name.lower().replace("_", "-")
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return list(PRESETS[key])
