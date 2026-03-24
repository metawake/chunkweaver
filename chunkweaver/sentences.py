"""Lightweight sentence splitting using regex heuristics.

No NLP dependencies. Handles the common case well; known to mis-split on
abbreviations like "Dr. Smith" (documented limitation).

The default pattern works for English and most Latin-script languages.
For CJK, chat transcripts, or other formats, pass a custom pattern
to ``split_sentences`` or configure ``sentence_pattern`` on the Chunker.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Union

SENTENCE_END = re.compile(r'([.!?])(\s+)(?=[A-Z"\(])')

SENTENCE_END_CJK = re.compile(r'([。！？])(\s*)')

SENTENCE_END_PERMISSIVE = re.compile(r'([.!?。！？])(\s+)')


def split_sentences(
    text: str,
    pattern: Union[str, Pattern[str], None] = None,
) -> List[str]:
    """Split *text* into sentences.

    Args:
        text: The text to split.
        pattern: Optional custom regex for sentence boundaries.
                 Must contain at least one capture group whose end marks the
                 split point. When ``None``, the default English pattern is
                 used.

    Returns a list of sentence strings. Trailing whitespace between sentences
    is attached to the preceding sentence so round-tripping preserves the
    original text exactly: ``"".join(split_sentences(t)) == t``.
    """
    if not text:
        return []

    if pattern is None:
        compiled = SENTENCE_END
    elif isinstance(pattern, str):
        compiled = re.compile(pattern)
    else:
        compiled = pattern

    sentences: List[str] = []
    last = 0

    for m in compiled.finditer(text):
        end = m.end()
        sentences.append(text[last:end])
        last = end

    if last < len(text):
        sentences.append(text[last:])

    return sentences


def last_n_sentences(
    text: str,
    n: int,
    pattern: Union[str, Pattern[str], None] = None,
) -> str:
    """Return the last *n* sentences of *text* as a single string."""
    if n <= 0:
        return ""
    parts = split_sentences(text, pattern=pattern)
    return "".join(parts[-n:])
