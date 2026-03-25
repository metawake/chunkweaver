"""Benchmark: heuristic vs ML clinical section detection.

Compares three approaches:
1. CLINICAL preset (regex only)
2. HeadingDetector (heuristic)
3. MLClinicalSectionDetector (trained model)

Tests on two scenarios:
A. Clean notes — sections separated by blank lines, some with headers
B. Dirty notes — blank lines removed between some sections, simulating
   run-on dictated transcription (the hard case)

Usage:
    python benchmark.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chunkweaver import Chunker
from chunkweaver.boundaries import detect_boundaries
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.presets import CLINICAL

from sample_notes import LABELED_NOTES


def _make_clean_note(note: list[tuple[str, str]]) -> tuple[str, list[int]]:
    """Build a clean note with blank-line separators. Return (text, boundary_lines)."""
    parts = [text for _, text in note]
    full = "\n\n".join(parts)
    lines = full.split("\n")

    boundary_lines: list[int] = []
    offset = 0
    for i, (_, text) in enumerate(note):
        if i > 0:
            boundary_line = offset
            for j in range(offset, len(lines)):
                if lines[j].strip():
                    boundary_lines.append(j)
                    break
        section_lines = text.split("\n")
        offset += len(section_lines) + 1

    return full, boundary_lines


def _make_dirty_note(
    note: list[tuple[str, str]], seed: int = 42
) -> tuple[str, list[int]]:
    """Build a dirty note: remove blank lines between ~60% of sections.

    Returns (text, boundary_lines) where boundary_lines are the true
    section-start line numbers.
    """
    rng = random.Random(seed)
    chunks: list[str] = []
    boundary_chars: list[int] = []

    for i, (label, text) in enumerate(note):
        if i == 0:
            chunks.append(text)
        else:
            use_blank = rng.random() < 0.4
            if use_blank:
                chunks.append("\n\n" + text)
            else:
                chunks.append("\n" + text)

    full = "".join(chunks)
    lines = full.split("\n")

    # Find true boundary lines by tracking section text
    boundary_lines: list[int] = []
    line_offset = 0
    for i, (_, text) in enumerate(note):
        if i > 0:
            section_first_line = text.split("\n")[0].strip()
            for j in range(line_offset, len(lines)):
                if lines[j].strip() == section_first_line:
                    boundary_lines.append(j)
                    line_offset = j + 1
                    break

        section_line_count = len(text.split("\n"))
        if i == 0:
            line_offset = section_line_count

    return full, boundary_lines


def _detect_with_preset(text: str) -> set[int]:
    """Detect section starts using CLINICAL preset regex."""
    matches = detect_boundaries(text, CLINICAL)
    return {m.line_number for m in matches}


def _detect_with_heading(text: str) -> set[int]:
    """Detect section starts using HeadingDetector."""
    hd = HeadingDetector(min_score=3.0)
    annotations = hd.detect(text)
    lines = text.split("\n")
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line) + 1

    detected_lines = set()
    for ann in annotations:
        for i, off in enumerate(offsets):
            if off == ann.position:
                detected_lines.add(i)
                break
    return detected_lines


def _detect_with_ml(text: str) -> set[int]:
    """Detect section starts using MLClinicalSectionDetector."""
    from detector import MLClinicalSectionDetector
    det = MLClinicalSectionDetector(threshold=0.5)
    annotations = det.detect(text)
    lines = text.split("\n")
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line) + 1

    detected_lines = set()
    for ann in annotations:
        for i, off in enumerate(offsets):
            if off == ann.position:
                detected_lines.add(i)
                break
    return detected_lines


def _score(
    true_boundaries: list[int],
    detected: set[int],
    tolerance: int = 2,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 with line-number tolerance.

    A detection is "correct" if it falls within `tolerance` lines
    of a true boundary.
    """
    if not true_boundaries and not detected:
        return 1.0, 1.0, 1.0

    true_set = set(true_boundaries)
    tp = 0
    matched_true = set()

    for d in detected:
        for t in true_set:
            if abs(d - t) <= tolerance and t not in matched_true:
                tp += 1
                matched_true.add(t)
                break

    precision = tp / max(len(detected), 1)
    recall = tp / max(len(true_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def run_benchmark():
    print("=" * 70)
    print("Clinical Section Detection Benchmark")
    print("=" * 70)

    methods = {
        "CLINICAL preset": _detect_with_preset,
        "HeadingDetector": _detect_with_heading,
        "ML detector": _detect_with_ml,
    }

    for scenario_name, note_builder in [
        ("CLEAN notes (blank-line separators)", _make_clean_note),
        ("DIRTY notes (run-on, ~60% blank lines removed)", _make_dirty_note),
    ]:
        print(f"\n{'─' * 70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'─' * 70}")

        totals = {name: {"p": [], "r": [], "f1": []} for name in methods}
        note_details: list[dict] = []

        for note_idx, note in enumerate(LABELED_NOTES):
            text, true_bounds = note_builder(note, seed=note_idx) if "dirty" in scenario_name.lower() else note_builder(note)
            line_count = len(text.split("\n"))

            detail = {
                "note": note_idx + 1,
                "lines": line_count,
                "true_boundaries": len(true_bounds),
            }

            for method_name, detect_fn in methods.items():
                detected = detect_fn(text)
                p, r, f1 = _score(true_bounds, detected, tolerance=2)
                totals[method_name]["p"].append(p)
                totals[method_name]["r"].append(r)
                totals[method_name]["f1"].append(f1)
                detail[f"{method_name}_detected"] = len(detected)
                detail[f"{method_name}_f1"] = f1

            note_details.append(detail)

        # Per-note results
        print(f"\n{'Note':>6}  {'Lines':>5}  {'True':>4}  ", end="")
        for name in methods:
            short = name.split()[0][:8]
            print(f"{'|':>3} {short:>8} det  F1   ", end="")
        print()
        print("─" * 90)

        for d in note_details:
            print(f"  {d['note']:>4}  {d['lines']:>5}  {d['true_boundaries']:>4}  ", end="")
            for name in methods:
                det = d[f"{name}_detected"]
                f1 = d[f"{name}_f1"]
                print(f"  |   {det:>4}    {f1:.2f}  ", end="")
            print()

        # Averages
        print("─" * 90)
        print(f"\n  {'Method':<25s}  {'Precision':>9s}  {'Recall':>6s}  {'F1':>6s}")
        print(f"  {'─' * 55}")
        for name in methods:
            avg_p = sum(totals[name]["p"]) / len(totals[name]["p"])
            avg_r = sum(totals[name]["r"]) / len(totals[name]["r"])
            avg_f1 = sum(totals[name]["f1"]) / len(totals[name]["f1"])
            print(f"  {name:<25s}  {avg_p:>9.3f}  {avg_r:>6.3f}  {avg_f1:>6.3f}")

    # Chunking comparison on a specific note
    print(f"\n{'=' * 70}")
    print("Chunking comparison: Note 2 (implicit transitions, dirty)")
    print(f"{'=' * 70}")

    note = LABELED_NOTES[1]  # Implicit transitions
    dirty_text, _ = _make_dirty_note(note, seed=1)

    configs = [
        ("CLINICAL preset only", {"boundaries": CLINICAL}),
        ("HeadingDetector", {"detectors": [HeadingDetector(min_score=3.0)]}),
        ("ML detector", {"detectors": [_load_ml_detector()]}),
        ("CLINICAL + ML", {"boundaries": CLINICAL, "detectors": [_load_ml_detector()]}),
    ]

    for label, kwargs in configs:
        chunker = Chunker(target_size=1024, overlap=0, min_size=0, **kwargs)
        chunks = chunker.chunk(dirty_text)
        sizes = [len(c) for c in chunks]
        print(f"\n  {label}:")
        print(f"    {len(chunks)} chunks, avg={sum(sizes)//len(sizes)}, "
              f"min={min(sizes)}, max={max(sizes)}")
        for i, c in enumerate(chunks):
            preview = c.strip()[:80].replace("\n", " ")
            print(f"    [{i}] ({len(c):>4} chars) {preview}...")


def _load_ml_detector():
    from detector import MLClinicalSectionDetector
    return MLClinicalSectionDetector(threshold=0.5)


if __name__ == "__main__":
    run_benchmark()
