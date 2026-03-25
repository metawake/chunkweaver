"""Post-chunking inspection and feedback loop.

Analyzes chunking output to detect problems, surface near-miss
boundaries, and (optionally) audit semantic coherence via LLM.

Three layers:
1. Heuristic diagnostics — size distribution, boundary breakdown,
   overlap health, orphan detection, actionable suggestions
2. Boundary gap detection — near-miss heading candidates and
   unmatched structural patterns
3. LLM coherence audit (optional) — per-chunk coherence rating

Usage::

    from chunkweaver import Chunker
    from chunkweaver.inspect import inspect_chunks

    chunker = Chunker(target_size=1024, boundaries=[...])
    chunks = chunker.chunk_with_metadata(text)
    report = inspect_chunks(chunks, text, target_size=1024)
    print(report.report())
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from chunkweaver.models import Chunk


@dataclass
class NearMissHeading:
    """A line that scored close to the heading threshold but was not used."""

    line_number: int
    text: str
    score: float


@dataclass
class PatternSuggestion:
    """A regex pattern the user might want to add as a boundary."""

    pattern: str
    reason: str
    sample_matches: list[str]


@dataclass
class ChunkCoherenceRating:
    """LLM coherence rating for a single chunk."""

    chunk_index: int
    rating: str  # "coherent", "partial", "incoherent"
    explanation: str


@dataclass
class InspectionReport:
    """Full post-chunking diagnostic report."""

    # Layer 1: size distribution
    chunk_count: int
    avg_size: int
    median_size: int
    min_size: int
    max_size: int
    size_cv: float  # coefficient of variation
    oversized_count: int
    undersized_count: int

    # Layer 1: boundary breakdown
    boundary_counts: dict[str, int]
    fallback_ratio: float  # fraction of chunks using fallback splits

    # Layer 1: overlap health
    high_overlap_chunks: list[int]  # chunk indices where overlap > 40%

    # Layer 1: orphan chunks (heading-only, no body)
    orphan_chunks: list[int]

    # Layer 1: suggestions
    suggestions: list[str]

    # Layer 2: near-miss headings
    near_miss_headings: list[NearMissHeading]

    # Layer 2: pattern suggestions
    pattern_suggestions: list[PatternSuggestion]

    # Layer 3: LLM coherence (optional)
    coherence_ratings: list[ChunkCoherenceRating] | None = None
    coherence_summary: dict[str, int] | None = None

    target_size: int = 1024

    def report(self) -> str:
        """Format as a human-readable CLI report."""
        lines: list[str] = []
        lines.append("=== chunkweaver inspect ===\n")

        lines.append(f"Chunks: {self.chunk_count}")
        lines.append(
            f"Sizes: avg={self.avg_size}, median={self.median_size}, "
            f"min={self.min_size}, max={self.max_size}"
        )
        lines.append(
            f"Size CV: {self.size_cv:.2f} "
            f"({'low variance' if self.size_cv < 0.3 else 'moderate' if self.size_cv < 0.6 else 'high variance'})"
        )
        if self.oversized_count:
            lines.append(f"Oversized (>2x target): {self.oversized_count}")
        if self.undersized_count:
            lines.append(f"Undersized (<200 chars): {self.undersized_count}")
        lines.append("")

        lines.append("--- Boundary breakdown ---")
        for btype, count in sorted(self.boundary_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {btype:15s} {count:3d}")
        lines.append(f"  Fallback ratio: {self.fallback_ratio:.0%}")
        lines.append("")

        if self.high_overlap_chunks:
            lines.append("--- Overlap health ---")
            lines.append(f"  {len(self.high_overlap_chunks)} chunks have overlap >40% of content:")
            for idx in self.high_overlap_chunks[:5]:
                lines.append(f"    chunk {idx}")
            if len(self.high_overlap_chunks) > 5:
                lines.append(f"    ... and {len(self.high_overlap_chunks) - 5} more")
            lines.append("")

        if self.orphan_chunks:
            lines.append("--- Orphan chunks ---")
            lines.append(f"  {len(self.orphan_chunks)} heading-only chunks (no body content):")
            for idx in self.orphan_chunks[:5]:
                lines.append(f"    chunk {idx}")
            lines.append("")

        if self.near_miss_headings:
            lines.append("--- Near-miss headings ---")
            lines.append(
                f"  {len(self.near_miss_headings)} lines scored close to heading "
                f"threshold but were not used as boundaries:"
            )
            for nm in self.near_miss_headings[:5]:
                lines.append(f"    line {nm.line_number}: {nm.text!r} (score={nm.score:.1f})")
            if len(self.near_miss_headings) > 5:
                lines.append(f"    ... and {len(self.near_miss_headings) - 5} more")
            lines.append("")

        if self.pattern_suggestions:
            lines.append("--- Suggested boundary patterns ---")
            for ps in self.pattern_suggestions:
                lines.append(f"  {ps.pattern}")
                lines.append(f"    {ps.reason}")
                for s in ps.sample_matches[:3]:
                    lines.append(f"    e.g. {s!r}")
            lines.append("")

        if self.coherence_ratings is not None:
            lines.append("--- LLM coherence audit ---")
            if self.coherence_summary:
                total = sum(self.coherence_summary.values())
                for rating in ("coherent", "partial", "incoherent"):
                    count = self.coherence_summary.get(rating, 0)
                    pct = count / total * 100 if total else 0
                    lines.append(f"  {rating:12s} {count:3d} ({pct:.0f}%)")

            incoherent = [r for r in self.coherence_ratings if r.rating == "incoherent"]
            if incoherent:
                lines.append("\n  Worst chunks:")
                for r in incoherent[:5]:
                    lines.append(f"    chunk {r.chunk_index}: {r.explanation}")
            lines.append("")

        if self.suggestions:
            lines.append("--- Suggestions ---")
            for s in self.suggestions:
                lines.append(f"  {s}")
            lines.append("")

        if not self.suggestions and not self.near_miss_headings and not self.pattern_suggestions:
            lines.append("No issues found. Chunking looks good.")

        return "\n".join(lines)


# -----------------------------------------------------------------------
# Layer 1: Heuristic diagnostics
# -----------------------------------------------------------------------

_FALLBACK_TYPES = {"paragraph", "sentence", "word"}

_ORPHAN_MAX_CHARS = 80
_ORPHAN_MAX_WORDS = 10


def _analyze_sizes(
    chunks: list[Chunk], target_size: int
) -> tuple[int, int, int, int, float, int, int]:
    """Return (avg, median, min, max, cv, oversized, undersized)."""
    sizes = [len(c.text) for c in chunks]
    if not sizes:
        return 0, 0, 0, 0, 0.0, 0, 0

    avg = int(statistics.mean(sizes))
    med = int(statistics.median(sizes))
    mn = min(sizes)
    mx = max(sizes)
    cv = statistics.stdev(sizes) / avg if avg > 0 and len(sizes) > 1 else 0.0
    oversized = sum(1 for s in sizes if s > target_size * 2)
    undersized = sum(1 for s in sizes if s < 200)
    return avg, med, mn, mx, round(cv, 3), oversized, undersized


def _boundary_breakdown(chunks: list[Chunk]) -> tuple[dict[str, int], float]:
    """Return (counts_by_type, fallback_ratio)."""
    counts: Counter[str] = Counter()
    for c in chunks:
        counts[c.boundary_type] += 1
    total = len(chunks)
    fallback = sum(counts.get(t, 0) for t in _FALLBACK_TYPES)
    ratio = fallback / total if total > 0 else 0.0
    return dict(counts), round(ratio, 3)


def _overlap_health(chunks: list[Chunk]) -> list[int]:
    """Return indices of chunks where overlap > 40% of total text."""
    bad = []
    for c in chunks:
        if c.overlap_text and len(c.text) > 0:
            ratio = len(c.overlap_text) / len(c.text)
            if ratio > 0.4:
                bad.append(c.index)
    return bad


def _detect_orphans(chunks: list[Chunk]) -> list[int]:
    """Return indices of heading-only chunks with no body content."""
    orphans = []
    for c in chunks:
        content = c.content_text.strip()
        if (
            len(content) < _ORPHAN_MAX_CHARS
            and len(content.split()) < _ORPHAN_MAX_WORDS
            and "\n" not in content.strip()
        ):
            orphans.append(c.index)
    return orphans


def _generate_suggestions(
    chunks: list[Chunk],
    target_size: int,
    fallback_ratio: float,
    oversized: int,
    undersized: int,
    high_overlap: list[int],
    orphans: list[int],
) -> list[str]:
    """Generate actionable text suggestions."""
    suggestions: list[str] = []

    if fallback_ratio > 0.5 and len(chunks) > 3:
        suggestions.append(
            f"Most splits ({fallback_ratio:.0%}) use fallback strategy. "
            "Your boundary patterns may not match this document. "
            "Try chunkweaver --recommend to find better patterns."
        )

    if oversized > len(chunks) * 0.2 and len(chunks) > 2:
        suggestions.append(
            f"{oversized} chunks exceed 2x target_size ({target_size * 2} chars). "
            "Consider adding more boundary patterns or reducing target_size."
        )

    if undersized > len(chunks) * 0.3 and len(chunks) > 2:
        suggestions.append(
            f"{undersized} chunks are under 200 chars. "
            "Consider increasing min_size to merge small fragments."
        )

    if high_overlap:
        suggestions.append(
            f"{len(high_overlap)} chunks have overlap >40% of total text. "
            "Consider reducing overlap or increasing target_size."
        )

    if orphans and len(orphans) > 1:
        suggestions.append(
            f"{len(orphans)} chunks appear to be heading-only (no body). "
            "Increase min_size to merge headings with their content."
        )

    return suggestions


# -----------------------------------------------------------------------
# Layer 2: Boundary gap detection
# -----------------------------------------------------------------------


def _find_near_miss_headings(
    text: str,
    used_line_numbers: set[int],
    min_score: float = 3.5,
    floor_score: float = 2.0,
) -> list[NearMissHeading]:
    """Find lines that almost qualified as headings but fell below threshold."""
    from chunkweaver.detector_heading import HeadingDetector

    hd = HeadingDetector(min_score=floor_score)
    candidates = hd.detect_with_scores(text)

    near_misses = []
    for c in candidates:
        if c.score < min_score and c.line_number not in used_line_numbers:
            near_misses.append(
                NearMissHeading(
                    line_number=c.line_number,
                    text=c.text,
                    score=round(c.score, 2),
                )
            )

    near_misses.sort(key=lambda nm: nm.score, reverse=True)
    return near_misses


_STRUCTURAL_PATTERNS = [
    (r"^SCHEDULE\s+[A-Z]", "SCHEDULE markers"),
    (r"^EXHIBIT\s+[A-Z]", "EXHIBIT markers"),
    (r"^APPENDIX\s+[A-Z0-9]", "APPENDIX markers"),
    (r"^ANNEX\s+[A-Z0-9]", "ANNEX markers"),
    (r"^PART\s+[IVX]+\b", "PART markers (Roman numerals)"),
    (r"^CHAPTER\s+\d+", "CHAPTER markers"),
    (r"^Section\s+\d+", "Section markers"),
    (r"^Article\s+\d+", "Article markers"),
    (r"^Item\s+\d+", "Item markers"),
    (r"^NOTE\s+\d+", "NOTE markers"),
    (r"^#{1,3}\s", "Markdown headings"),
    (r"^\d+\.\d+\s", "Numbered sub-sections"),
]


def _suggest_patterns(
    text: str,
    existing_boundaries: list[str],
) -> list[PatternSuggestion]:
    """Scan for common structural patterns not covered by existing boundaries."""
    existing_set = set(existing_boundaries)
    suggestions: list[PatternSuggestion] = []

    for pattern, description in _STRUCTURAL_PATTERNS:
        if pattern in existing_set:
            continue

        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) >= 2:
            suggestions.append(
                PatternSuggestion(
                    pattern=pattern,
                    reason=f"Found {len(matches)} {description} not covered by current boundaries",
                    sample_matches=matches[:5],
                )
            )

    return suggestions


# -----------------------------------------------------------------------
# Layer 3: LLM coherence audit (optional)
# -----------------------------------------------------------------------

_COHERENCE_PROMPT = """\
Rate this text chunk's semantic coherence. Does it contain a complete,
self-contained unit of information, or does it appear cut mid-topic?

Respond with exactly one word: coherent, partial, or incoherent.
Then a brief explanation (one sentence).

Chunk:
{chunk_text}"""


def audit_coherence(
    chunks: list[Chunk],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_chunks: int = 50,
) -> tuple[list[ChunkCoherenceRating], dict[str, int]]:
    """Rate each chunk's semantic coherence using an LLM.

    Sends up to *max_chunks* chunk texts to the OpenAI API and
    classifies each as ``"coherent"``, ``"partial"``, or
    ``"incoherent"``.

    Args:
        chunks: Chunks to evaluate.
        api_key: OpenAI API key.
        model: Chat model to use for scoring.
        max_chunks: Cap on the number of chunks sent to the API.

    Returns:
        A ``(ratings, summary_counts)`` tuple where *ratings* is a list
        of per-chunk ``ChunkCoherenceRating`` objects and
        *summary_counts* is a ``{label: count}`` dict.

    Requires ``openai`` package: ``pip install openai``.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "LLM coherence audit requires the openai package. Install with: pip install openai"
        )

    client = OpenAI(api_key=api_key)
    ratings: list[ChunkCoherenceRating] = []
    sample = chunks[:max_chunks]

    for chunk in sample:
        prompt = _COHERENCE_PROMPT.format(chunk_text=chunk.text[:2000])
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()
            first_word = answer.split()[0].rstrip(".,:")
            if first_word in ("coherent", "partial", "incoherent"):
                rating = first_word
            else:
                rating = "partial"
            explanation = answer[len(first_word) :].strip().lstrip(".,: ")
        except Exception as e:
            rating = "partial"
            explanation = f"API error: {e}"

        ratings.append(
            ChunkCoherenceRating(
                chunk_index=chunk.index,
                rating=rating,
                explanation=explanation,
            )
        )

    summary: Counter[str] = Counter()
    for r in ratings:
        summary[r.rating] += 1

    return ratings, dict(summary)


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------


def inspect_chunks(
    chunks: list[Chunk],
    text: str,
    target_size: int = 1024,
    boundaries: Sequence[str] | None = None,
    heading_threshold: float = 3.5,
) -> InspectionReport:
    """Analyze chunking output and return a diagnostic report.

    Runs Layer 1 (heuristic diagnostics) and Layer 2 (boundary gap
    detection). Layer 3 (LLM audit) can be run separately via
    ``audit_coherence()`` and attached to the report.

    Args:
        chunks: Output from ``Chunker.chunk_with_metadata()``.
        text: The original source text.
        target_size: The ``target_size`` used for chunking.
        boundaries: The boundary patterns used (for gap detection).
        heading_threshold: The ``min_score`` used by HeadingDetector.
    """
    if not chunks:
        return InspectionReport(
            chunk_count=0,
            avg_size=0,
            median_size=0,
            min_size=0,
            max_size=0,
            size_cv=0.0,
            oversized_count=0,
            undersized_count=0,
            boundary_counts={},
            fallback_ratio=0.0,
            high_overlap_chunks=[],
            orphan_chunks=[],
            suggestions=["Document produced zero chunks."],
            near_miss_headings=[],
            pattern_suggestions=[],
            target_size=target_size,
        )

    # Layer 1
    avg, med, mn, mx, cv, oversized, undersized = _analyze_sizes(chunks, target_size)
    boundary_counts, fallback_ratio = _boundary_breakdown(chunks)
    high_overlap = _overlap_health(chunks)
    orphans = _detect_orphans(chunks)
    suggestions = _generate_suggestions(
        chunks,
        target_size,
        fallback_ratio,
        oversized,
        undersized,
        high_overlap,
        orphans,
    )

    # Layer 2
    used_lines: set[int] = set()
    for c in chunks:
        if c.boundary_type == "section":
            line_no = text[: c.start].count("\n")
            used_lines.add(line_no)

    near_misses = _find_near_miss_headings(
        text,
        used_lines,
        min_score=heading_threshold,
    )
    pattern_sugs = _suggest_patterns(text, list(boundaries or []))

    return InspectionReport(
        chunk_count=len(chunks),
        avg_size=avg,
        median_size=med,
        min_size=mn,
        max_size=mx,
        size_cv=cv,
        oversized_count=oversized,
        undersized_count=undersized,
        boundary_counts=boundary_counts,
        fallback_ratio=fallback_ratio,
        high_overlap_chunks=high_overlap,
        orphan_chunks=orphans,
        suggestions=suggestions,
        near_miss_headings=near_misses,
        pattern_suggestions=pattern_sugs,
        target_size=target_size,
    )
