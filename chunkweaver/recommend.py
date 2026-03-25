"""Analyze a document and recommend chunkweaver configuration.

Scans text for preset pattern matches, heading/table signals,
OCR damage indicators, and basic statistics to suggest boundaries,
detectors, and target_size.  Runs a dry-run chunk to validate the
recommendation.

Usage::

    from chunkweaver.recommend import recommend
    rec = recommend(text)
    print(rec.report())
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from chunkweaver.boundaries import detect_boundaries
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.detector_table import TableDetector
from chunkweaver.presets import PRESETS


# Presets that are commonly combined (same domain, complementary patterns)
_COMBO_PAIRS: List[Tuple[str, str]] = [
    ("financial", "financial-table"),
]


@dataclass
class OcrDamageReport:
    """Assessment of OCR / PDF extraction quality."""
    damaged_line_count: int
    total_short_lines: int
    damage_ratio: float  # damaged / total short lines
    level: str  # "none", "light", "heavy"
    recommend_ml_detector: bool
    sample_lines: List[str]


@dataclass
class PresetMatch:
    """How well a preset matched a document."""
    name: str
    hits: int
    density: float  # hits per 100 lines
    pattern_coverage: float  # fraction of preset patterns that fired
    score: float  # combined ranking score
    sample_matches: List[str]


@dataclass
class ChunkStats:
    """Statistics from a dry-run chunk of the document."""
    chunk_count: int
    avg_size: int
    min_size: int
    max_size: int
    median_size: int
    oversized_count: int  # chunks > 2x target
    undersized_count: int  # chunks < min_size
    warnings: List[str]


@dataclass
class Recommendation:
    """Full recommendation for a document."""
    char_count: int
    line_count: int
    paragraph_count: int
    avg_paragraph_chars: int

    preset_matches: List[PresetMatch]
    recommended_presets: List[str]
    extra_boundaries: List[str]

    heading_count: int
    recommend_heading_detector: bool
    heading_samples: List[str]

    table_count: int
    recommend_table_detector: bool
    table_samples: List[str]

    suggested_target_size: int
    suggested_overlap: int

    ocr_damage: Optional[OcrDamageReport] = None
    chunk_stats: Optional[ChunkStats] = None

    @property
    def recommended_preset(self) -> str:
        """Primary preset (first in the list)."""
        return self.recommended_presets[0] if self.recommended_presets else "plain"

    def report(self) -> str:
        """Format as a human-readable CLI report."""
        lines: list[str] = []
        lines.append("=== chunkweaver recommend ===\n")

        lines.append(f"Document: {self.char_count:,} chars, {self.line_count:,} lines, "
                     f"{self.paragraph_count} paragraphs")
        lines.append(f"Avg paragraph: ~{self.avg_paragraph_chars} chars\n")

        lines.append("--- Preset matching ---")
        if self.preset_matches:
            for pm in self.preset_matches:
                is_selected = pm.name in self.recommended_presets
                marker = " <--" if is_selected else ""
                cov_pct = int(pm.pattern_coverage * 100)
                lines.append(
                    f"  {pm.name:20s}  {pm.hits:3d} hits  "
                    f"density={pm.density:.1f}/100ln  "
                    f"coverage={cov_pct}%  "
                    f"score={pm.score:.2f}{marker}"
                )
                for s in pm.sample_matches[:3]:
                    lines.append(f"    e.g. {s!r}")
        else:
            lines.append("  (no preset patterns matched)")

        if len(self.recommended_presets) > 1:
            combo = " + ".join(p.upper().replace("-", "_") for p in self.recommended_presets)
            lines.append(f"\n  Recommended combo: {combo}")
        lines.append("")

        lines.append("--- Detectors ---")
        if self.recommend_heading_detector:
            lines.append(f"  HeadingDetector: YES ({self.heading_count} headings found)")
            for s in self.heading_samples[:3]:
                lines.append(f"    e.g. {s!r}")
        else:
            lines.append(f"  HeadingDetector: not needed ({self.heading_count} headings)")

        if self.recommend_table_detector:
            lines.append(f"  TableDetector:   YES ({self.table_count} tables found)")
            for s in self.table_samples[:3]:
                lines.append(f"    e.g. {s!r}")
        else:
            lines.append(f"  TableDetector:   not needed ({self.table_count} tables)")
        lines.append("")

        if self.ocr_damage and self.ocr_damage.level != "none":
            od = self.ocr_damage
            lines.append("--- OCR quality ---")
            lines.append(f"  Damage level: {od.level} "
                         f"({od.damaged_line_count} damaged lines, "
                         f"{od.damage_ratio:.0%} of short lines)")
            for s in od.sample_lines[:3]:
                lines.append(f"    e.g. {s!r}")
            if od.recommend_ml_detector:
                lines.append("  → Recommend MLOCRHeadingDetector (see examples/ml-detectors/)")
            lines.append("")

        lines.append("--- Suggested config ---")
        lines.append(f"  target_size = {self.suggested_target_size}")
        lines.append(f"  overlap     = {self.suggested_overlap}")
        lines.append("")

        if self.chunk_stats:
            cs = self.chunk_stats
            lines.append("--- Dry-run results ---")
            lines.append(f"  {cs.chunk_count} chunks produced")
            lines.append(f"  sizes: avg={cs.avg_size}, median={cs.median_size}, "
                         f"min={cs.min_size}, max={cs.max_size}")
            if cs.oversized_count:
                lines.append(f"  {cs.oversized_count} chunks over 2x target "
                             f"(>{self.suggested_target_size * 2} chars)")
            if cs.undersized_count:
                lines.append(f"  {cs.undersized_count} chunks under min_size (200 chars)")
            if cs.warnings:
                for w in cs.warnings:
                    lines.append(f"  WARNING: {w}")
            if not cs.warnings and not cs.oversized_count:
                lines.append("  Looks good.")
            lines.append("")

        lines.append("--- Python snippet ---")
        lines.append(self.snippet())

        return "\n".join(lines)

    def snippet(self) -> str:
        """Generate a ready-to-use Python snippet."""
        imports = ["from chunkweaver import Chunker"]
        detector_args: list[str] = []
        comments: list[str] = []

        preset_consts = [
            p.upper().replace("-", "_")
            for p in self.recommended_presets
            if p != "plain"
        ]
        if preset_consts:
            consts_str = ", ".join(preset_consts)
            imports.append(f"from chunkweaver.presets import {consts_str}")

        use_ml_ocr = (
            self.ocr_damage is not None
            and self.ocr_damage.recommend_ml_detector
        )

        if self.recommend_heading_detector and not use_ml_ocr:
            imports.append("from chunkweaver.detector_heading import HeadingDetector")
            detector_args.append("HeadingDetector()")
        if use_ml_ocr:
            comments.append(
                "# OCR damage detected — use ML heading detector\n"
                "# pip install scikit-learn joblib\n"
                "# See examples/ml-detectors/ocr_heading_detector/"
            )
            imports.append("from chunkweaver.detector_heading import HeadingDetector")
            detector_args.append("HeadingDetector()")
            detector_args.append("MLOCRHeadingDetector()")
        if self.recommend_table_detector:
            imports.append("from chunkweaver.detector_table import TableDetector")
            detector_args.append("TableDetector()")

        parts = ["\n".join(imports)]
        if comments:
            parts.append("\n".join(comments))
        parts.append("")
        ctor_lines = ["chunker = Chunker("]
        ctor_lines.append(f"    target_size={self.suggested_target_size},")
        ctor_lines.append(f"    overlap={self.suggested_overlap},")

        if preset_consts:
            if self.extra_boundaries:
                extras = ", ".join(repr(b) for b in self.extra_boundaries)
                boundary_expr = " + ".join(preset_consts) + f" + [{extras}]"
            else:
                boundary_expr = " + ".join(preset_consts)
            ctor_lines.append(f"    boundaries={boundary_expr},")

        if detector_args:
            det_str = ", ".join(detector_args)
            ctor_lines.append(f"    detectors=[{det_str}],")

        ctor_lines.append(")")
        parts.append("\n".join(ctor_lines))
        return "\n".join(parts)


def _detect_ocr_damage(text: str) -> OcrDamageReport:
    """Detect OCR letterspacing artifacts in the document.

    Scans short lines (likely headings/labels) for the telltale pattern:
    abnormally high ratio of single-character alphabetic tokens.
    ``"D E F I N I T I O N S"`` has 11 tokens, 10 are single-char.
    Normal prose never looks like that.
    """
    lines = text.split("\n")
    damaged_lines: list[str] = []
    short_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue

        tokens = stripped.split()
        n_tokens = len(tokens)
        if n_tokens < 3:
            continue

        short_lines += 1

        single_alpha = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
        single_ratio = single_alpha / n_tokens

        # Check for partial fragmentation: many short non-common tokens
        common_short = {
            "the", "and", "for", "not", "but", "are", "was", "has", "its",
            "all", "any", "can", "may", "per", "via", "our", "due", "set",
            "no", "of", "to", "in", "on", "at", "or", "by", "is", "as",
            "an", "be", "do", "if", "so", "up", "it", "he", "we", "a", "i",
        }
        fragments = sum(
            1 for t in tokens
            if 2 <= len(t) <= 4
            and t.lower() not in common_short
            and not t.isdigit()
        )
        fragment_ratio = fragments / n_tokens

        # Letterspacing: >40% single-char alpha tokens
        # Partial damage: >40% fragment tokens AND avg token len < 4
        avg_token_len = sum(len(t) for t in tokens) / n_tokens

        if single_ratio >= 0.4:
            damaged_lines.append(stripped)
        elif fragment_ratio >= 0.4 and avg_token_len < 4.0:
            damaged_lines.append(stripped)

    n_damaged = len(damaged_lines)
    ratio = n_damaged / max(short_lines, 1)

    if n_damaged == 0:
        level = "none"
    elif n_damaged <= 3 or ratio < 0.1:
        level = "light"
    else:
        level = "heavy"

    return OcrDamageReport(
        damaged_line_count=n_damaged,
        total_short_lines=short_lines,
        damage_ratio=round(ratio, 3),
        level=level,
        recommend_ml_detector=level in ("light", "heavy"),
        sample_lines=damaged_lines[:5],
    )


def _count_paragraphs(text: str) -> Tuple[int, int]:
    """Return (paragraph_count, avg_paragraph_chars)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return 0, 0
    avg = sum(len(p) for p in paras) // len(paras)
    return len(paras), avg


def _detect_extra_patterns(text: str) -> List[str]:
    """Detect structural patterns not covered by standard presets."""
    extras: list[str] = []
    if re.search(r"^```", text, re.MULTILINE):
        extras.append(r"^```")
    return extras


def _score_presets(
    text: str, line_count: int
) -> List[PresetMatch]:
    """Score each preset with density and pattern coverage."""
    results: list[PresetMatch] = []
    lines_factor = max(line_count, 1) / 100.0

    for name, patterns in sorted(PRESETS.items()):
        if not patterns:
            continue

        matches = detect_boundaries(text, patterns)
        if not matches:
            continue

        hit_count = len(matches)
        density = hit_count / lines_factor

        matched_patterns = {m.pattern for m in matches}
        coverage = len(matched_patterns) / len(patterns)

        score = density * (0.5 + 0.5 * coverage)

        samples = [m.matched_text for m in matches[:5]]
        results.append(PresetMatch(
            name=name,
            hits=hit_count,
            density=round(density, 2),
            pattern_coverage=round(coverage, 2),
            score=round(score, 2),
            sample_matches=samples,
        ))

    results.sort(key=lambda p: p.score, reverse=True)
    return results


def _pick_presets(matches: List[PresetMatch]) -> List[str]:
    """Pick the best preset(s), including useful combos."""
    if not matches:
        return ["plain"]

    best = matches[0]
    selected = [best.name]

    match_by_name = {m.name: m for m in matches}

    for a, b in _COMBO_PAIRS:
        if best.name == a and b in match_by_name:
            partner = match_by_name[b]
            if partner.hits >= 2 and partner.score >= best.score * 0.1:
                selected.append(b)
                break
        elif best.name == b and a in match_by_name:
            partner = match_by_name[a]
            if partner.hits >= 2 and partner.score >= best.score * 0.1:
                selected.insert(0, a)
                break

    return selected


def _suggest_target_size(
    avg_para: int,
    boundary_count: int,
    char_count: int,
) -> int:
    """Suggest target_size based on paragraph stats and boundary density."""
    if boundary_count > 0:
        avg_segment = char_count / (boundary_count + 1)
        if avg_segment < 400:
            return 512
        elif avg_segment < 900:
            return 768
        elif avg_segment < 1800:
            return 1024
        else:
            return 2048

    if avg_para > 2000:
        return 2048
    elif avg_para > 800:
        return 1024
    elif avg_para > 300:
        return 768
    return 512


def _dry_run(
    text: str,
    recommended_presets: List[str],
    extra_boundaries: List[str],
    target_size: int,
    overlap: int,
    recommend_hd: bool,
    recommend_td: bool,
) -> ChunkStats:
    """Chunk the document with recommended config and collect stats."""
    from chunkweaver.chunker import Chunker
    from chunkweaver.detectors import BoundaryDetector

    boundaries: list[str] = []
    for p in recommended_presets:
        if p != "plain" and p in PRESETS:
            boundaries.extend(PRESETS[p])
    boundaries.extend(extra_boundaries)

    detectors: list[BoundaryDetector] = []
    if recommend_hd:
        detectors.append(HeadingDetector())
    if recommend_td:
        detectors.append(TableDetector())

    chunker = Chunker(
        target_size=target_size,
        overlap=overlap,
        overlap_unit="sentence",
        boundaries=boundaries,
        fallback="paragraph",
        min_size=200,
        detectors=detectors,
    )

    chunks = chunker.chunk(text)
    if not chunks:
        return ChunkStats(
            chunk_count=0, avg_size=0, min_size=0, max_size=0,
            median_size=0, oversized_count=0, undersized_count=0,
            warnings=["Document produced zero chunks"],
        )

    sizes = [len(c) for c in chunks]
    oversized = sum(1 for s in sizes if s > target_size * 2)
    undersized = sum(1 for s in sizes if s < 200)

    warnings: list[str] = []
    if oversized > len(chunks) * 0.2:
        warnings.append(
            f"{oversized}/{len(chunks)} chunks are over 2x target. "
            "Consider adding more boundary patterns or reducing target_size."
        )
    if undersized > len(chunks) * 0.3:
        warnings.append(
            f"{undersized}/{len(chunks)} chunks are very small. "
            "Consider increasing min_size or reducing boundary patterns."
        )
    max_size = max(sizes)
    if max_size > target_size * 3:
        warnings.append(
            f"Largest chunk is {max_size} chars ({max_size / target_size:.1f}x target). "
            "This section may lack internal structure for splitting."
        )

    return ChunkStats(
        chunk_count=len(chunks),
        avg_size=int(statistics.mean(sizes)),
        min_size=min(sizes),
        max_size=max_size,
        median_size=int(statistics.median(sizes)),
        oversized_count=oversized,
        undersized_count=undersized,
        warnings=warnings,
    )


def recommend(text: str) -> Recommendation:
    """Analyze *text* and return a ``Recommendation``."""
    line_count = text.count("\n") + 1
    para_count, avg_para = _count_paragraphs(text)

    preset_results = _score_presets(text, line_count)
    recommended = _pick_presets(preset_results)

    total_boundary_hits = preset_results[0].hits if preset_results else 0
    extra_bounds = _detect_extra_patterns(text)

    # OCR damage assessment
    ocr_report = _detect_ocr_damage(text)

    # Detectors — scale thresholds by document size
    headings_per_100 = max(line_count, 1) / 100.0

    hd = HeadingDetector(min_score=3.5)
    heading_candidates = hd.detect_with_scores(text)
    heading_samples = [c.text for c in heading_candidates[:5]]

    td = TableDetector()
    table_regions = td.detect_with_metadata(text)
    table_samples = [r.header_text for r in table_regions[:5]]

    recommend_hd = (
        len(heading_candidates) >= 3
        and len(heading_candidates) / headings_per_100 >= 0.8
    )
    recommend_td = len(table_regions) >= 1

    target = _suggest_target_size(avg_para, total_boundary_hits, len(text))
    suggested_overlap = 2 if target >= 768 else 1

    chunk_stats = _dry_run(
        text, recommended, extra_bounds, target, suggested_overlap,
        recommend_hd, recommend_td,
    )

    return Recommendation(
        char_count=len(text),
        line_count=line_count,
        paragraph_count=para_count,
        avg_paragraph_chars=avg_para,
        preset_matches=preset_results,
        recommended_presets=recommended,
        extra_boundaries=extra_bounds,
        heading_count=len(heading_candidates),
        recommend_heading_detector=recommend_hd,
        heading_samples=heading_samples,
        table_count=len(table_regions),
        recommend_table_detector=recommend_td,
        table_samples=table_samples,
        suggested_target_size=target,
        suggested_overlap=suggested_overlap,
        ocr_damage=ocr_report,
        chunk_stats=chunk_stats,
    )
