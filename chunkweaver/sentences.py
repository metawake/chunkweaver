"""Lightweight sentence splitting using regex heuristics.

No NLP dependencies. Handles the common case well; known to mis-split on
abbreviations like "Dr. Smith" (documented limitation).

The default ``SENTENCE_END`` pattern works for English and unaccented
Latin-script languages. It requires ``[A-Z]`` after the punctuation, so
it will *not* split correctly for:

- **Cyrillic** scripts (Serbian, Montenegrin, Russian, …)
- **Accented Latin** capitals (Spanish ``Á``, French ``É``, …)
- **CJK** sentence-ending punctuation (``。！？``)

Use ``SENTENCE_END_PERMISSIVE`` for script-agnostic splitting, or
``SENTENCE_END_CJK`` for CJK-only text. Pass a custom pattern to
``split_sentences`` or configure ``sentence_pattern`` on the Chunker.
"""

from __future__ import annotations

import re
from re import Pattern

SENTENCE_END = re.compile(r'([.!?])(\s+)(?=[A-Z"\(])')

SENTENCE_END_CJK = re.compile(r"([。！？])(\s*)")

SENTENCE_END_PERMISSIVE = re.compile(r"([.!?。！？])(\s+)")


def split_sentences(
    text: str,
    pattern: str | Pattern[str] | None = None,
) -> list[str]:
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

    sentences: list[str] = []
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
    pattern: str | Pattern[str] | None = None,
) -> str:
    """Return the last *n* sentences of *text* as a single string.

    Args:
        text: Source text to extract sentences from.
        n: Number of trailing sentences to return. If *n* exceeds the
           total sentence count, the full text is returned.
        pattern: Custom sentence boundary regex, forwarded to
                 ``split_sentences``.
    """
    if n <= 0:
        return ""
    parts = split_sentences(text, pattern=pattern)
    return "".join(parts[-n:])
