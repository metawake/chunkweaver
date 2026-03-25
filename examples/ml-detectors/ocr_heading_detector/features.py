"""Feature extraction for OCR-damaged heading detection.

Key insight: letterspaced OCR artifacts like "D E F I N I T I O N S"
have a distinctive fingerprint — abnormally high ratio of single-char
tokens. Normal prose never has 90% single-char words on one line.

Additional signals: line shortness relative to neighbors, contextual
position (preceded by blank, followed by longer content), and partial
word fragments from broken OCR.
"""

from __future__ import annotations
import re
import string

import numpy as np


def extract_features(lines: list[str], idx: int) -> dict[str, float]:
    """Extract features for a single line in context."""
    line = lines[idx]
    tokens = line.split()
    n_tokens = len(tokens)

    # --- Core letterspacing features ---
    single_char_tokens = sum(1 for t in tokens if len(t) == 1) if tokens else 0
    single_char_ratio = single_char_tokens / n_tokens if n_tokens > 0 else 0.0

    # Spaces vs non-space characters
    n_chars = len(line)
    n_spaces = line.count(" ")
    n_nonspace = n_chars - n_spaces
    space_ratio = n_spaces / n_chars if n_chars > 0 else 0.0

    # Average token length (letterspaced → ~1.0, normal → ~4-5)
    avg_token_len = (sum(len(t) for t in tokens) / n_tokens) if n_tokens > 0 else 0.0

    # "Collapsed" form: what the line looks like with extra spaces removed
    collapsed = re.sub(r"\s+", "", line)
    collapsed_len = len(collapsed)
    expansion_ratio = n_chars / collapsed_len if collapsed_len > 0 else 1.0

    # --- Partial spacing features ---
    # Count tokens that look like word fragments (2-4 chars, not common words)
    COMMON_SHORT = {"the", "and", "for", "not", "but", "are", "was", "has", "its",
                    "all", "any", "can", "may", "per", "via", "our", "due", "set",
                    "no", "of", "to", "in", "on", "at", "or", "by", "is", "as",
                    "an", "be", "do", "if", "so", "up", "it", "he", "we", "a", "i"}
    fragment_tokens = sum(
        1 for t in tokens
        if 2 <= len(t) <= 4 and t.lower() not in COMMON_SHORT and not t.isdigit()
    ) if tokens else 0
    fragment_ratio = fragment_tokens / n_tokens if n_tokens > 0 else 0.0

    # --- Line-level features ---
    line_len = len(line.strip())
    is_short = 1.0 if line_len < 60 else 0.0

    # Uppercase ratio of non-space characters
    alpha_chars = [c for c in line if c.isalpha()]
    upper_ratio = (sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)) if alpha_chars else 0.0

    # Starts with digit/number pattern (e.g., "1.", "2.3", "Section")
    starts_with_number = 1.0 if re.match(r"^\s*\d", line) else 0.0

    # Punctuation density (headings have less punctuation than body text)
    punct_count = sum(1 for c in line if c in string.punctuation and c not in ".-()") if line else 0
    punct_ratio = punct_count / n_chars if n_chars > 0 else 0.0

    # Contains legal/section keywords in collapsed form
    collapsed_lower = collapsed.lower()
    has_section_keyword = 1.0 if any(
        kw in collapsed_lower
        for kw in [
            "section", "article", "chapter", "part", "schedule",
            "definitions", "recitals", "whereas", "amendment",
            "exhibit", "appendix", "annex", "form", "item",
            "compensation", "termination", "confidential",
            "general", "provision", "regulation",
        ]
    ) else 0.0

    # --- Context features ---
    # Blank line before
    blank_before = 1.0 if idx == 0 or lines[idx - 1].strip() == "" else 0.0

    # Blank line after
    blank_after = 1.0 if idx == len(lines) - 1 or (idx + 1 < len(lines) and lines[idx + 1].strip() == "") else 0.0

    # Length ratio vs next non-blank line
    next_content_len = 0
    for j in range(idx + 1, min(idx + 4, len(lines))):
        if lines[j].strip():
            next_content_len = len(lines[j].strip())
            break
    len_ratio_next = line_len / next_content_len if next_content_len > 0 else 1.0

    # Length ratio vs previous non-blank line
    prev_content_len = 0
    for j in range(idx - 1, max(idx - 4, -1), -1):
        if lines[j].strip():
            prev_content_len = len(lines[j].strip())
            break
    len_ratio_prev = line_len / prev_content_len if prev_content_len > 0 else 1.0

    # --- OCR-specific: consecutive single-char token runs ---
    # "D E F I N I T I O N S" has a run of 11 single-char tokens.
    # "- W-9 tax form" has max run of 1.
    max_single_run = 0
    current_run = 0
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            current_run += 1
            max_single_run = max(max_single_run, current_run)
        else:
            current_run = 0

    # Regularity of spacing — OCR letterspacing produces evenly spaced chars
    # e.g. "D E F" has gaps [1,1,1], "Dear Mr" has gaps [3]
    # Measure: std dev of inter-token gaps (lower = more regular = more OCR-like)
    gaps = []
    if len(tokens) >= 2:
        pos = 0
        for t in tokens:
            start = line.find(t, pos)
            if start > pos and gaps is not None:
                gaps.append(start - pos)
            pos = start + len(t)
    gap_std = float(np.std(gaps)) if len(gaps) >= 2 else 0.0
    gap_regularity = 1.0 / (1.0 + gap_std)  # higher = more regular spacing

    # Ratio of the line that would collapse to a "known heading" pattern
    # after removing spaces — detects damaged versions of real headings
    collapsed_words = re.sub(r"\s+", " ", collapsed_lower).split()
    n_collapsed_words = len(collapsed_words)

    return {
        "single_char_ratio": single_char_ratio,
        "space_ratio": space_ratio,
        "avg_token_len": avg_token_len,
        "expansion_ratio": expansion_ratio,
        "fragment_ratio": fragment_ratio,
        "line_len": float(line_len),
        "is_short": is_short,
        "upper_ratio": upper_ratio,
        "starts_with_number": starts_with_number,
        "punct_ratio": punct_ratio,
        "has_section_keyword": has_section_keyword,
        "blank_before": blank_before,
        "blank_after": blank_after,
        "len_ratio_next": len_ratio_next,
        "len_ratio_prev": len_ratio_prev,
        "n_tokens": float(n_tokens),
        "max_single_char_run": float(max_single_run),
        "gap_regularity": gap_regularity,
        "n_collapsed_words": float(n_collapsed_words),
    }


FEATURE_NAMES = list(extract_features(["test line", "another line"], 0).keys())
