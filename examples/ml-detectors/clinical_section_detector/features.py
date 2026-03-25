"""Feature extraction for clinical section boundary detection.

Extracts per-line features from text, designed for clinical notes.
Features capture patterns that signal section transitions: header-like
formatting, medical abbreviation density, content type shifts.
"""

from __future__ import annotations

import re
from typing import List

# Clinical section header patterns (partial matches also score)
_HEADER_PATTERNS = [
    re.compile(r"^(HISTORY OF PRESENT ILLNESS|HPI)\s*:?", re.IGNORECASE),
    re.compile(r"^(CHIEF COMPLAINT|CC)\s*:?", re.IGNORECASE),
    re.compile(r"^(PAST MEDICAL HISTORY|PMH)\s*:?", re.IGNORECASE),
    re.compile(r"^(MEDICATIONS?|MEDS|CURRENT MEDICATIONS?)\s*:?", re.IGNORECASE),
    re.compile(r"^(ALLERGIES|ALLERGY|NKDA)\s*:?", re.IGNORECASE),
    re.compile(r"^(REVIEW OF SYSTEMS|ROS)\s*:?", re.IGNORECASE),
    re.compile(r"^(PHYSICAL EXAM(INATION)?|PE|EXAM(INATION)?)\s*:?", re.IGNORECASE),
    re.compile(r"^(ASSESSMENT|IMPRESSION|A/P)\s*:?", re.IGNORECASE),
    re.compile(r"^(PLAN|TREATMENT PLAN)\s*:?", re.IGNORECASE),
    re.compile(r"^(LAB(ORATORY)?\s*(DATA|RESULTS)?|LABS)\s*:?", re.IGNORECASE),
    re.compile(r"^(VITAL SIGNS?|VITALS|VS)\s*:?", re.IGNORECASE),
    re.compile(r"^(FAMILY HISTORY|FH|FHX)\s*:?", re.IGNORECASE),
    re.compile(r"^(SOCIAL HISTORY|SH|SHX)\s*:?", re.IGNORECASE),
    re.compile(r"^(IMAGING|RADIOLOGY|X-?RAY|CT|MRI)\s*:?", re.IGNORECASE),
    re.compile(r"^(DISCHARGE|DISPOSITION)\s*:?", re.IGNORECASE),
    re.compile(r"^(MENTAL STATUS|MSE)\s*:?", re.IGNORECASE),
    re.compile(r"^(PROCEDURE|OPERATIVE)\s*:?", re.IGNORECASE),
]

# Medical abbreviation patterns (signals clinical content type)
_MED_ABBREVS = re.compile(
    r"\b(mg|mcg|BID|TID|QID|QHS|PRN|PO|IV|IM|SQ|QD|daily|"
    r"q\d+h|x\d+|tab|caps?)\b", re.IGNORECASE
)
_VITAL_SIGNS = re.compile(
    r"\b(BP|HR|RR|T|Temp|O2|SpO2|sat|GCS|BMI)\s*:?\s*\d", re.IGNORECASE
)
_LAB_VALUES = re.compile(
    r"\b(WBC|Hgb|Hb|Plt|Na|K|Cl|CO2|BUN|Cr|glucose|troponin|"
    r"BNP|INR|A1c|TSH|CRP|ESR|lactate)\s*:?\s*[\d.]", re.IGNORECASE
)
_NUMBERED_LIST = re.compile(r"^\s*\d+[\.\)]\s")
_DASH_LIST = re.compile(r"^\s*[-•]\s")

# Transition phrases that often start new sections in dictated notes
_TRANSITION_PHRASES = re.compile(
    r"^(On exam(ination)?|On review|Her |His |The patient|"
    r"So in summary|In summary|Overall|"
    r"I (am going to|will|recommend|plan)|"
    r"We will|"
    r"(Psychiatric|Medical|Surgical|Family|Social)\s+(history|hx))",
    re.IGNORECASE,
)


def extract_line_features(lines: list[str], line_idx: int) -> list[float]:
    """Extract features for a single line given its context.

    Returns a fixed-length feature vector.
    """
    line = lines[line_idx]
    n = len(lines)
    features: list[float] = []

    # --- Line properties ---
    line_len = len(line)
    features.append(min(line_len / 200.0, 1.0))  # normalized length

    alpha = [c for c in line if c.isalpha()]
    cap_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
    features.append(cap_ratio)

    features.append(1.0 if line.rstrip().endswith(":") else 0.0)
    features.append(1.0 if line.rstrip().endswith(".") else 0.0)

    # --- Header pattern match ---
    header_match = any(p.match(line) for p in _HEADER_PATTERNS)
    features.append(1.0 if header_match else 0.0)

    # --- Content type signals ---
    med_count = len(_MED_ABBREVS.findall(line))
    features.append(min(med_count / 5.0, 1.0))

    vital_match = bool(_VITAL_SIGNS.search(line))
    features.append(1.0 if vital_match else 0.0)

    lab_match = bool(_LAB_VALUES.search(line))
    features.append(1.0 if lab_match else 0.0)

    features.append(1.0 if _NUMBERED_LIST.match(line) else 0.0)
    features.append(1.0 if _DASH_LIST.match(line) else 0.0)

    features.append(1.0 if _TRANSITION_PHRASES.match(line) else 0.0)

    # --- Context features ---
    prev_line = lines[line_idx - 1] if line_idx > 0 else ""
    next_line = lines[line_idx + 1] if line_idx < n - 1 else ""

    features.append(1.0 if prev_line.strip() == "" else 0.0)  # blank before
    features.append(1.0 if next_line.strip() == "" else 0.0)  # blank after

    # Content type shift detection
    prev_has_meds = bool(_MED_ABBREVS.search(prev_line)) if prev_line else False
    curr_has_meds = bool(_MED_ABBREVS.search(line))
    features.append(1.0 if prev_has_meds != curr_has_meds else 0.0)

    prev_has_vitals = bool(_VITAL_SIGNS.search(prev_line)) if prev_line else False
    curr_has_vitals = bool(_VITAL_SIGNS.search(line))
    features.append(1.0 if prev_has_vitals != curr_has_vitals else 0.0)

    # Length ratio to neighbors
    prev_len = len(prev_line.strip())
    features.append(line_len / max(prev_len, 1) if prev_len > 0 else 1.0)

    # Position in document
    features.append(line_idx / max(n - 1, 1))

    # Is this the first line?
    features.append(1.0 if line_idx == 0 else 0.0)

    return features


FEATURE_NAMES = [
    "line_length_norm",
    "cap_ratio",
    "ends_colon",
    "ends_period",
    "header_pattern",
    "med_abbrev_density",
    "has_vitals",
    "has_labs",
    "numbered_list",
    "dash_list",
    "transition_phrase",
    "blank_before",
    "blank_after",
    "med_content_shift",
    "vital_content_shift",
    "length_ratio_prev",
    "position_in_doc",
    "is_first_line",
]

NUM_FEATURES = len(FEATURE_NAMES)
