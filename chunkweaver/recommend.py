"""Analyze a document and recommend chunkweaver configuration.

Scans text for preset pattern matches, heading/table signals,
and basic statistics to suggest boundaries, detectors, and target_size.

Usage::

    from chunkweaver.recommend import recommend
    rec = recommend(text)
    print(rec.report())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from chunkweaver.boundaries import detect_boundaries
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.detector_table import TableDetector
from chunkweaver.presets import PRESETS


@dataclass
class PresetMatch:
    """How well a preset matched a document."""
    name: str
    hits: int
    sample_matches: List[str]


@dataclass
class Recommendation:
    """Full recommendation for a document."""
    char_count: int
    line_count: int
    paragraph_count: int
    avg_paragraph_chars: int

    preset_matches: List[PresetMatch]
    recommended_preset: str
    extra_boundaries: List[str]

    heading_count: int
    recommend_heading_detector: bool
    heading_samples: List[str]

    table_count: int
    recommend_table_detector: bool
    table_samples: List[str]

    suggested_target_size: int
    suggested_overlap: int

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
                marker = " <-- best" if pm.name == self.recommended_preset else ""
                lines.append(f"  {pm.name:20s}  {pm.hits:3d} hits{marker}")
                for s in pm.sample_matches[:3]:
                    lines.append(f"    e.g. {s!r}")
        else:
            lines.append("  (no preset patterns matched)")
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

        lines.append("--- Suggested config ---")
        lines.append(f"  target_size = {self.suggested_target_size}")
        lines.append(f"  overlap     = {self.suggested_overlap}")
        lines.append("")

        lines.append("--- Python snippet ---")
        lines.append(self.snippet())

        return "\n".join(lines)

    def snippet(self) -> str:
        """Generate a ready-to-use Python snippet."""
        imports = ["from chunkweaver import Chunker"]
        detector_args: list[str] = []

        if self.recommended_preset != "plain":
            imports.append(
                f"from chunkweaver.presets import {self.recommended_preset.upper().replace('-', '_')}"
            )

        if self.recommend_heading_detector:
            imports.append("from chunkweaver.detector_heading import HeadingDetector")
            detector_args.append("HeadingDetector()")
        if self.recommend_table_detector:
            imports.append("from chunkweaver.detector_table import TableDetector")
            detector_args.append("TableDetector()")

        parts = ["\n".join(imports), ""]
        ctor_lines = ["chunker = Chunker("]
        ctor_lines.append(f"    target_size={self.suggested_target_size},")
        ctor_lines.append(f"    overlap={self.suggested_overlap},")

        if self.recommended_preset != "plain":
            preset_const = self.recommended_preset.upper().replace("-", "_")
            if self.extra_boundaries:
                extras = ", ".join(repr(b) for b in self.extra_boundaries)
                ctor_lines.append(f"    boundaries={preset_const} + [{extras}],")
            else:
                ctor_lines.append(f"    boundaries={preset_const},")

        if detector_args:
            det_str = ", ".join(detector_args)
            ctor_lines.append(f"    detectors=[{det_str}],")

        ctor_lines.append(")")
        parts.append("\n".join(ctor_lines))
        return "\n".join(parts)


def _count_paragraphs(text: str) -> Tuple[int, int]:
    """Return (paragraph_count, avg_paragraph_chars)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return 0, 0
    avg = sum(len(p) for p in paras) // len(paras)
    return len(paras), avg


def _detect_markdown(text: str) -> List[str]:
    """Check for markdown-specific patterns not covered by presets."""
    extras: list[str] = []
    if re.search(r"^```", text, re.MULTILINE):
        extras.append(r"^```")
    return extras


def recommend(text: str) -> Recommendation:
    """Analyze *text* and return a ``Recommendation``."""
    line_count = text.count("\n") + 1
    para_count, avg_para = _count_paragraphs(text)

    # Score each preset
    preset_results: list[PresetMatch] = []
    for name, patterns in sorted(PRESETS.items()):
        if not patterns:
            continue
        matches = detect_boundaries(text, patterns)
        if matches:
            samples = [m.matched_text for m in matches[:5]]
            preset_results.append(PresetMatch(name=name, hits=len(matches), sample_matches=samples))

    preset_results.sort(key=lambda p: p.hits, reverse=True)
    best_preset = preset_results[0].name if preset_results else "plain"

    extra_bounds = _detect_markdown(text)

    # Run detectors
    hd = HeadingDetector(min_score=3.5)
    heading_candidates = hd.detect_with_scores(text)
    heading_samples = [c.text for c in heading_candidates[:5]]

    td = TableDetector()
    table_regions = td.detect_with_metadata(text)
    table_samples = [r.header_text for r in table_regions[:5]]

    recommend_hd = len(heading_candidates) >= 3
    recommend_td = len(table_regions) >= 1

    # Suggest target_size
    if avg_para > 2000:
        target = 2048
    elif avg_para > 800:
        target = 1024
    elif avg_para > 300:
        target = 768
    else:
        target = 512

    suggested_overlap = 2 if target >= 768 else 1

    return Recommendation(
        char_count=len(text),
        line_count=line_count,
        paragraph_count=para_count,
        avg_paragraph_chars=avg_para,
        preset_matches=preset_results,
        recommended_preset=best_preset,
        extra_boundaries=extra_bounds,
        heading_count=len(heading_candidates),
        recommend_heading_detector=recommend_hd,
        heading_samples=heading_samples,
        table_count=len(table_regions),
        recommend_table_detector=recommend_td,
        table_samples=table_samples,
        suggested_target_size=target,
        suggested_overlap=suggested_overlap,
    )
