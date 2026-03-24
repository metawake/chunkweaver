"""HeadingDetector spike — heuristic heading detection for plain text.

Scores each line on multiple signals and returns probable headings.
No ML, no NLP — just text heuristics with adaptive weighting.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class HeadingCandidate:
    line_number: int
    text: str
    score: float
    signals: tuple


# ---------------------------------------------------------------------------
# Rejection filters — lines that should never be headings
# ---------------------------------------------------------------------------
_PAGE_FOOTER_RE = re.compile(
    r"^.{0,40}\|\s*\d{4}\s+Form\s+10-K\s*\|", re.IGNORECASE
)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_TABLE_OF_CONTENTS_RE = re.compile(r"^Table of Contents\s*$", re.IGNORECASE)
_FORM_CHECKBOX_RE = re.compile(r"[☒☐]")
_XBRL_JUNK_RE = re.compile(r"^[a-z0-9\-:]+Member|^iso4217:|^xbrli:|^us-gaap:")
_SEPARATOR_RE = re.compile(r"^[_\-=]{3,}\s*$")
_PHONE_RE = re.compile(r"\(\d{3}\)\s*\d{3}[\-\s]\d{4}")
_PERSON_NAME_RE = re.compile(
    r"^[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+\d{2,3}\s|$)"
)
_MOSTLY_DIGITS_RE = re.compile(r"[\d$,%.()]{4,}")
_SEC_PREFIX_RE = re.compile(
    r"^(Item\s+\d|PART\s+[IVX]|NOTE\s+\d|Schedule\s+[A-Z\d])", re.IGNORECASE
)

# Single words that appear often in tables / boilerplate but aren't headings
_NOISE_WORDS = frozenset({
    "total", "none", "page", "nasdaq", "yes", "no", "or", "and",
    "exhibit", "filed", "filed herewith", "furnished", "incorporated",
    "washington", "delaware", "california", "new york",
})


def _is_title_case(text: str) -> bool:
    words = text.split()
    if len(words) < 1:
        return False
    skip = {"and", "or", "the", "of", "in", "for", "to", "a", "an",
            "on", "at", "by", "with", "from", "&"}
    cap_count = sum(1 for w in words if w[0].isupper() or w.lower() in skip)
    return cap_count / len(words) >= 0.7


def _is_all_caps(text: str) -> bool:
    alpha = [c for c in text if c.isalpha()]
    if len(alpha) < 3:
        return False
    return sum(1 for c in alpha if c.isupper()) / len(alpha) >= 0.8


def _ends_with_sentence_punct(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def detect_headings(
    text: str,
    min_score: float = 4.0,
    max_heading_len: int = 100,
    min_heading_len: int = 3,
) -> List[HeadingCandidate]:
    """Detect probable headings in plain text using adaptive heuristics."""
    lines = text.split("\n")
    n = len(lines)
    if n == 0:
        return []

    # Pre-compute document-level statistics for adaptive weighting
    stripped = [ln.strip() for ln in lines]
    non_empty_lens = [len(s) for s in stripped if len(s) > 10]
    avg_len = statistics.mean(non_empty_lens) if non_empty_lens else 80.0

    # Blank-line density: what fraction of non-empty lines are preceded by a blank?
    preceded_by_blank = 0
    non_empty_count = 0
    for i, s in enumerate(stripped):
        if s:
            non_empty_count += 1
            if i > 0 and stripped[i - 1] == "":
                preceded_by_blank += 1
    blank_density = preceded_by_blank / max(non_empty_count, 1)

    # If >60% of lines have blank_before, it's a heavily-spaced document
    # and blank_before/after become unreliable signals
    blank_weight = 1.0 if blank_density < 0.4 else (0.3 if blank_density < 0.7 else 0.0)

    candidates: List[HeadingCandidate] = []

    for i, line in enumerate(stripped):
        line_len = len(line)

        if line_len < min_heading_len or line_len > max_heading_len:
            continue

        # Hard rejections
        if _PAGE_FOOTER_RE.match(line):
            continue
        if _PAGE_NUMBER_RE.match(line):
            continue
        if _TABLE_OF_CONTENTS_RE.match(line):
            continue
        if _FORM_CHECKBOX_RE.search(line):
            continue
        if _XBRL_JUNK_RE.match(line):
            continue
        if _SEPARATOR_RE.match(line):
            continue
        if "\t" in line:
            continue
        if _PHONE_RE.search(line):
            continue
        if _PERSON_NAME_RE.match(line):
            continue

        # Noise word check
        if line.lower().rstrip(".:,;") in _NOISE_WORDS:
            continue

        score = 0.0
        signals = []

        # --- Negative signals ---
        digit_ratio = sum(1 for c in line if c.isdigit()) / max(line_len, 1)
        if digit_ratio > 0.3:
            score -= 2.0
            signals.append("num_heavy")
        if _MOSTLY_DIGITS_RE.search(line) and not _SEC_PREFIX_RE.match(line):
            score -= 1.0
            signals.append("has_nums")

        if line.startswith("(") or line.startswith("$"):
            score -= 1.5
            signals.append("paren_or_dollar")

        # Single word with <6 chars and not a known heading pattern
        words = line.split()
        if len(words) == 1 and line_len < 6 and not _SEC_PREFIX_RE.match(line):
            score -= 1.0
            signals.append("too_terse")

        # --- Positive signals ---

        # Short line relative to document average (+1.5)
        if line_len < avg_len * 0.4:
            score += 1.5
            signals.append("short")

        # Title Case (+1.0)
        if _is_title_case(line):
            score += 1.0
            signals.append("title_case")

        # ALL CAPS (+1.5)
        if _is_all_caps(line):
            score += 1.5
            signals.append("all_caps")

        # No sentence-ending punctuation (+1.0)
        if not _ends_with_sentence_punct(line):
            score += 1.0
            signals.append("no_period")

        # Ends with colon (+0.5) — e.g., "Financial holding company:"
        if line.rstrip().endswith(":"):
            score += 0.5
            signals.append("colon_end")

        # Preceded by blank line (adaptive weight)
        if i > 0 and stripped[i - 1] == "":
            score += blank_weight
            if blank_weight > 0:
                signals.append("blank_before")

        # Followed by blank line (adaptive, half weight)
        if i < n - 1 and stripped[i + 1] == "":
            score += blank_weight * 0.5
            if blank_weight > 0:
                signals.append("blank_after")

        # Next non-blank line is much longer (+1.0)
        next_content_len = 0
        for j in range(i + 1, min(i + 4, n)):
            if stripped[j]:
                next_content_len = len(stripped[j])
                break
        if next_content_len > line_len * 2 and next_content_len > 50:
            score += 1.0
            signals.append("long_next")

        # SEC section prefix (+1.5)
        if _SEC_PREFIX_RE.match(line):
            score += 1.5
            signals.append("sec_prefix")

        # Multi-word requirement bonus: 2+ word headings are more credible
        if len(words) >= 2:
            score += 0.5
            signals.append("multi_word")

        if score >= min_score:
            candidates.append(HeadingCandidate(
                line_number=i,
                text=line,
                score=score,
                signals=tuple(signals),
            ))

    return candidates


def headings_to_boundary_patterns(candidates: List[HeadingCandidate]) -> List[str]:
    """Convert detected headings to exact-match boundary patterns for Chunker."""
    return [f"^{re.escape(c.text)}$" for c in candidates]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    filing_dir = Path(__file__).parent / "10k_filings"
    files = sorted(filing_dir.glob("*.txt"))

    if not files:
        print("No 10-K files found. Run download_10k.py first.")
        sys.exit(1)

    for fpath in files:
        text = fpath.read_text()
        candidates = detect_headings(text)

        print(f"\n{'='*70}")
        print(f"  {fpath.name}  ({len(text)} chars, {len(text.splitlines())} lines)")
        print(f"  Detected {len(candidates)} headings (min_score=4.0)")
        print(f"{'='*70}")

        for c in candidates:
            sigs = ", ".join(c.signals)
            print(f"  L{c.line_number:5d}  [{c.score:.1f}]  {c.text[:70]:<70s}  ({sigs})")

        print()
