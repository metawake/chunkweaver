"""Heuristic heading detection for plain text documents.

Scores each line on multiple signals (casing, length, whitespace context,
known prefixes) and emits ``SplitPoint`` annotations for probable headings.
No ML, no NLP — just text heuristics with adaptive weighting.

Usage::

    from chunkweaver import Chunker
    from chunkweaver.detector_heading import HeadingDetector

    chunker = Chunker(
        target_size=1024,
        detectors=[HeadingDetector(min_score=4.0)],
    )
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Sequence

from chunkweaver.detectors import Annotation, BoundaryDetector, SplitPoint


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


@dataclass(frozen=True)
class HeadingCandidate:
    """Internal scoring result before conversion to SplitPoint."""
    line_number: int
    position: int
    text: str
    score: float
    signals: tuple


class HeadingDetector(BoundaryDetector):
    """Detect probable section headings using adaptive heuristics.

    Configurable via constructor parameters:

    Args:
        min_score: Minimum cumulative score for a line to be considered
            a heading. Higher values → fewer, more confident detections.
        max_heading_len: Lines longer than this are never headings.
        min_heading_len: Lines shorter than this are never headings.
    """

    def __init__(
        self,
        min_score: float = 4.0,
        max_heading_len: int = 100,
        min_heading_len: int = 3,
    ) -> None:
        self.min_score = min_score
        self.max_heading_len = max_heading_len
        self.min_heading_len = min_heading_len

    def detect(self, text: str) -> List[Annotation]:
        candidates = self._score_lines(text)
        return [
            SplitPoint(
                position=c.position,
                line_number=c.line_number,
                label=f"heading: {c.text[:60]}",
            )
            for c in candidates
        ]

    def detect_with_scores(self, text: str) -> List[HeadingCandidate]:
        """Return raw candidates with scores (useful for debugging)."""
        return self._score_lines(text)

    def _score_lines(self, text: str) -> List[HeadingCandidate]:
        lines = text.split("\n")
        n = len(lines)
        if n == 0:
            return []

        stripped = [ln.strip() for ln in lines]
        non_empty_lens = [len(s) for s in stripped if len(s) > 10]
        avg_len = statistics.mean(non_empty_lens) if non_empty_lens else 80.0

        preceded_by_blank = 0
        non_empty_count = 0
        for i, s in enumerate(stripped):
            if s:
                non_empty_count += 1
                if i > 0 and stripped[i - 1] == "":
                    preceded_by_blank += 1
        blank_density = preceded_by_blank / max(non_empty_count, 1)
        blank_weight = 1.0 if blank_density < 0.4 else (0.3 if blank_density < 0.7 else 0.0)

        candidates: List[HeadingCandidate] = []
        offset = 0

        for i, line in enumerate(stripped):
            line_start = offset
            offset += len(lines[i]) + 1

            line_len = len(line)
            if line_len < self.min_heading_len or line_len > self.max_heading_len:
                continue

            if (
                _PAGE_FOOTER_RE.match(line)
                or _PAGE_NUMBER_RE.match(line)
                or _TABLE_OF_CONTENTS_RE.match(line)
                or _FORM_CHECKBOX_RE.search(line)
                or _XBRL_JUNK_RE.match(line)
                or _SEPARATOR_RE.match(line)
                or "\t" in line
                or _PHONE_RE.search(line)
                or _PERSON_NAME_RE.match(line)
            ):
                continue

            if line.lower().rstrip(".:,;") in _NOISE_WORDS:
                continue

            score = 0.0
            signals: list[str] = []

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

            words = line.split()
            if len(words) == 1 and line_len < 6 and not _SEC_PREFIX_RE.match(line):
                score -= 1.0
                signals.append("too_terse")

            # --- Positive signals ---
            if line_len < avg_len * 0.4:
                score += 1.5
                signals.append("short")

            if _is_title_case(line):
                score += 1.0
                signals.append("title_case")

            if _is_all_caps(line):
                score += 1.5
                signals.append("all_caps")

            if not _ends_with_sentence_punct(line):
                score += 1.0
                signals.append("no_period")

            if line.rstrip().endswith(":"):
                score += 0.5
                signals.append("colon_end")

            if i > 0 and stripped[i - 1] == "":
                score += blank_weight
                if blank_weight > 0:
                    signals.append("blank_before")

            if i < n - 1 and stripped[i + 1] == "":
                score += blank_weight * 0.5
                if blank_weight > 0:
                    signals.append("blank_after")

            next_content_len = 0
            for j in range(i + 1, min(i + 4, n)):
                if stripped[j]:
                    next_content_len = len(stripped[j])
                    break
            if next_content_len > line_len * 2 and next_content_len > 50:
                score += 1.0
                signals.append("long_next")

            if _SEC_PREFIX_RE.match(line):
                score += 1.5
                signals.append("sec_prefix")

            if len(words) >= 2:
                score += 0.5
                signals.append("multi_word")

            if score >= self.min_score:
                candidates.append(HeadingCandidate(
                    line_number=i,
                    position=line_start,
                    text=line,
                    score=score,
                    signals=tuple(signals),
                ))

        return candidates
