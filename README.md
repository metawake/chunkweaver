# chunkweaver

**Structure-aware text chunking for RAG. Zero dependencies.**

[![PyPI version](https://img.shields.io/pypi/v/chunkweaver.svg)](https://pypi.org/project/chunkweaver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The problem

`RecursiveCharacterTextSplitter` splits by size and hopes for the best.
It doesn't know that "Article 17" is a section header, or that a table
shouldn't be split in half. The result: **incoherent chunks that straddle
structural boundaries**, producing blurry embeddings and failed retrievals.

Our [LLM-as-judge benchmark](benchmark/README.md) on 11 structured documents
(GDPR, EU AI Act, CCPA, 8 IETF RFCs) and 58 queries shows:

| Metric | chunkweaver | Naive 512-char |
|--------|-------------|----------------|
| Full-answer chunks (top-3) | **13%** | 2% |
| Per-query wins | **18** | 3 |
| Binomial p-value | **0.0015** | — |

Structure-aware chunks are **6x more likely to contain a complete answer**
(p < 0.002). See [benchmark/](benchmark/) for methodology, code, and
reproduction steps.

## The fix

chunkweaver splits at **structural boundaries you define**, falls back to
paragraphs/sentences when sections are too large, and overlaps in
**semantic units** (sentences) instead of characters.

- **Zero dependencies** — stdlib only, no LangChain/LlamaIndex tax
- **User-defined boundaries** — regex patterns, not hard-coded heuristics
- **Semantic overlap** — sentences, not characters
- **Full metadata** — offsets, boundary types, overlap tracking
- **Drop-in LangChain replacement** — optional integration

## Install

```bash
pip install chunkweaver
```

Extras:

```bash
pip install chunkweaver[cli]        # CLI with click
pip install chunkweaver[langchain]  # LangChain TextSplitter integration
pip install chunkweaver[dev]        # pytest + coverage
```

## Quick start

```python
from chunkweaver import Chunker

# Minimal — just better defaults
chunker = Chunker(target_size=1024)
chunks = chunker.chunk(text)

# Full configuration
chunker = Chunker(
    target_size=1024,
    overlap=2,
    overlap_unit="sentence",
    boundaries=[
        r"^Article\s+\d+",   # GDPR articles
        r"^#{1,3}\s",        # Markdown headers
        r"^\d+\.\d+\s",      # Numbered sections (RFC)
        r"^TABLE\s+",        # Table headers
    ],
    fallback="paragraph",
    min_size=200,
)

# List of strings
chunks = chunker.chunk(text)

# With metadata
chunks = chunker.chunk_with_metadata(text)
for c in chunks:
    print(c.text)            # full chunk content
    print(c.start, c.end)    # character offsets in original
    print(c.boundary_type)   # "section" | "paragraph" | "sentence" | "word"
    print(c.overlap_text)    # the overlap prefix (if any)
    print(c.content_text)    # text without overlap (for dedup)
```

## Vector DB integration

chunkweaver is designed for vector database ingest pipelines:

```python
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU

chunker = Chunker(
    target_size=1024,
    overlap=2,
    overlap_unit="sentence",
    boundaries=LEGAL_EU,
)

chunks = chunker.chunk_with_metadata(document_text)

# Prepare records for your vector DB
records = [
    {
        "id": f"doc-{doc_id}-chunk-{c.index}",
        "text": c.text,
        "metadata": {
            "source": filename,
            "start": c.start,
            "end": c.end,
            "boundary_type": c.boundary_type,
            "has_overlap": bool(c.overlap_text),
        },
    }
    for c in chunks
]

# Embed and upsert into Pinecone / Qdrant / Weaviate / ChromaDB / etc.
```

## Presets

Built-in boundary patterns for common document types:

```python
from chunkweaver.presets import (
    LEGAL_EU, LEGAL_US, RFC, MARKDOWN,
    CHAT, CLINICAL, FINANCIAL, FINANCIAL_TABLE, PLAIN,
)
```

| Preset | Domain | Detects |
|--------|--------|---------|
| `LEGAL_EU` | EU legislation | `Article N`, `CHAPTER`, `SECTION`, `(1)` recitals |
| `LEGAL_US` | US law / contracts | `§ N`, `Section N`, `WHEREAS`, `1.1` clauses |
| `RFC` | IETF RFCs | `1. Intro`, `3.1 Overview`, `Appendix A` |
| `MARKDOWN` | Markdown | `# headings`, `---` rules |
| `CHAT` | Chat logs | `[14:30]`, ISO timestamps, `speaker:` turns |
| `CLINICAL` | Medical notes | `HPI:`, `ASSESSMENT:`, `PLAN:`, etc. |
| `FINANCIAL` | SEC filings | `Item 1.`, `PART I`, `NOTE 1`, `Schedule A` |
| `FINANCIAL_TABLE` | Data tables | `TABLE N`, markdown/ASCII separators |
| `PLAIN` | Any | No boundaries — pure paragraph/sentence fallback |

Combine presets freely:

```python
boundaries = LEGAL_EU + [r"^TABLE\s+", r"^Annex\s+"]
boundaries = FINANCIAL + FINANCIAL_TABLE
```

## LangChain integration

Drop-in replacement for `RecursiveCharacterTextSplitter`:

```python
from chunkweaver.integrations.langchain import ChunkWeaverSplitter

splitter = ChunkWeaverSplitter(
    target_size=1024,
    overlap=2,
    boundaries=[r"^#{1,3}\s"],
)

# Works with LangChain document loaders
docs = splitter.create_documents([text])
```

Requires: `pip install chunkweaver[langchain]`

## CLI

```bash
# Basic usage
chunkweaver document.txt --size 1024 --overlap 2

# With boundary patterns
chunkweaver file.txt --boundaries "^Article\s+\d+" "^CHAPTER\s+"

# Use a preset
chunkweaver legal_doc.txt --preset legal-eu --format json

# JSONL output (one chunk per line, pipe-friendly)
chunkweaver file.txt --preset markdown --format jsonl

# Preview boundary detection (tune your patterns)
chunkweaver file.txt --detect-boundaries --boundaries "^Article\s+\d+"

# Pipe from stdin
cat document.txt | chunkweaver --size 1024 --preset rfc
```

## Customization cookbook

chunkweaver is designed to be adapted to any text type. Here are recipes
for common scenarios.

### Chat logs & customer support transcripts

Chat text is informal — no uppercase after periods, no paragraph structure.
The default sentence regex (`[.!?]\s+(?=[A-Z"(])`) won't split it well.
Use the `CHAT` preset for turn-level boundaries and `SENTENCE_END_PERMISSIVE`
for overlap that works with lowercase text:

```python
from chunkweaver import Chunker, SENTENCE_END_PERMISSIVE
from chunkweaver.presets import CHAT

chunker = Chunker(
    target_size=512,
    overlap=1,
    overlap_unit="sentence",
    boundaries=CHAT,
    sentence_pattern=SENTENCE_END_PERMISSIVE,
    min_size=0,
)

chat_log = """[14:30] Agent: Welcome to support. How can I help?
[14:31] Customer: My order hasn't arrived. It's been 10 days.
[14:32] Agent: I'm sorry to hear that. Let me look into it.
[14:33] Customer: The order number is 12345.
[14:34] Agent: I see it was shipped Jan 5. It appears to be delayed."""

chunks = chunker.chunk(chat_log)
# Each speaker turn becomes its own chunk
```

### Chinese / Japanese / Korean text

CJK languages use different sentence-ending punctuation (。！？).
The default regex won't detect these:

```python
from chunkweaver import Chunker, SENTENCE_END_CJK

chunker = Chunker(
    target_size=512,
    overlap=1,
    overlap_unit="sentence",
    sentence_pattern=SENTENCE_END_CJK,
)

text = "第一条规定了保护范围。第二条界定了适用条件。第三条明确了领土管辖权。"
chunks = chunker.chunk(text)
```

For mixed-language documents, use a combined pattern:

```python
import re

chunker = Chunker(
    target_size=512,
    overlap=1,
    sentence_pattern=re.compile(r'([.!?。！？])(\s*)'),
)
```

### Healthcare / clinical notes

Discharge summaries and clinical notes have predictable section headers.
The `CLINICAL` preset recognizes `CHIEF COMPLAINT:`, `HPI:`, `ASSESSMENT:`,
`PLAN:`, and many more:

```python
from chunkweaver import Chunker
from chunkweaver.presets import CLINICAL

chunker = Chunker(
    target_size=1024,
    overlap=1,
    overlap_unit="sentence",
    boundaries=CLINICAL,
    min_size=50,   # merge very short sections like "ALLERGIES: NKDA"
)

note = """CHIEF COMPLAINT: Chest pain and shortness of breath.
HPI: 65-year-old male presenting with acute onset chest pain.
ASSESSMENT: Acute coronary syndrome, rule out MI.
PLAN: Admit to telemetry. Serial troponins q6h."""

chunks = chunker.chunk(note)
# Each clinical section stays intact
```

### Financial tables (keep headers with data)

The biggest problem with table chunking: headers get separated from values.
Use `keep_together` to glue table headers to their data rows:

```python
from chunkweaver import Chunker
from chunkweaver.presets import FINANCIAL, FINANCIAL_TABLE

chunker = Chunker(
    target_size=1024,
    overlap=0,
    boundaries=FINANCIAL + FINANCIAL_TABLE,
    keep_together=[r"^TABLE\s+\d+"],  # TABLE header stays with its rows
    min_size=0,
)

report = """Item 1. Business
The Company operates in financial services.

TABLE 1
Revenue | 2023 | 2024
Product A | 100M | 120M
Product B | 50M | 60M

Item 2. Properties
Headquarters located in New York."""

chunks = chunker.chunk(report)
# TABLE 1 + its rows stay in one chunk
```

### US contracts & legal filings

US legal documents use `§`, `Section`, `WHEREAS`, and numbered clauses:

```python
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_US

chunker = Chunker(
    target_size=1024,
    overlap=2,
    boundaries=LEGAL_US,
)

contract = """WHEREAS, the parties wish to enter into an agreement;
WHEREAS, the terms have been negotiated in good faith;
NOW, THEREFORE the parties agree as follows:
Section 1 Definitions.
1.1 "Agreement" means this document.
Section 2 Obligations.
§ 3 Governing law."""

chunks = chunker.chunk(contract)
```

### Custom boundaries for any domain

You're not limited to presets. Any regex that matches line starts works:

```python
# Jupyter notebooks (markdown cells)
boundaries = [r"^# In\[\d+\]", r"^#{1,3}\s"]

# Log files
boundaries = [r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}"]

# Email threads
boundaries = [r"^From:", r"^On .+ wrote:", r"^>+ On"]

# LaTeX
boundaries = [r"^\\section\{", r"^\\subsection\{", r"^\\chapter\{"]

# reStructuredText
boundaries = [r"^={3,}\s*$", r"^-{3,}\s*$", r"^\.\.\s+\w+::"]

chunker = Chunker(target_size=1024, boundaries=boundaries)
```

### Tuning tips

**Choosing `target_size`**: Larger chunks = better retrieval but fewer results
per query. Start with 1024 for dense prose, 512 for short-form content (chat,
clinical notes), 2048 for legal/technical documents with long sections.

**Choosing `overlap`**: 2 sentences is a good default. Use 0 when chunks are
already small or when you need exact deduplication. Use `overlap_unit="chars"`
with `overlap=100` for predictable sizing.

**Choosing `min_size`**: Set to 0 when every boundary should produce a chunk
(e.g., chat turns). Set to 200+ when standalone headings should merge with
their body text.

**Debugging boundaries**: Use `--detect-boundaries` on the CLI to preview
what your patterns match before chunking:

```bash
chunkweaver doc.txt --detect-boundaries --boundaries "^Article\s+\d+"
# line 5: [^Article\s+\d+] 'Article 1'
# line 23: [^Article\s+\d+] 'Article 2'
```

## How it works

1. **Detect boundaries** — scan each line against your regex patterns
2. **Split at boundaries** — create one segment per structural section
3. **Sub-split oversized segments** — break large sections at paragraph → sentence → word boundaries
4. **Merge undersized segments** — combine tiny segments (like standalone headings) with their body text
5. **Add overlap** — prepend the last N sentences/paragraphs/chars from the previous chunk
6. **Return** — chunks with full metadata (offsets, boundary type, overlap tracking)

## Architecture

```
chunkweaver/
├── __init__.py          # Public API: Chunker, Chunk, sentence patterns
├── chunker.py           # Core algorithm + keep_together logic
├── models.py            # Chunk dataclass
├── boundaries.py        # Boundary detection engine
├── sentences.py         # Configurable sentence splitting (regex, no NLP)
├── presets.py           # 9 domain presets (legal, clinical, chat, etc.)
├── cli.py               # CLI entry point
└── integrations/
    └── langchain.py     # LangChain TextSplitter wrapper
```

**Design principles:**
- Each module has a single responsibility
- No deeply nested conditionals — small, testable functions
- All decisions are logged/exposed via chunk metadata
- Zero dependencies for core; optional extras for CLI and LangChain

## API reference

### `Chunker(target_size, overlap, overlap_unit, boundaries, fallback, min_size, sentence_pattern, keep_together)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | `int` | `1024` | Target chunk size in characters |
| `overlap` | `int` | `2` | Number of overlap units from previous chunk |
| `overlap_unit` | `str` | `"sentence"` | `"sentence"`, `"paragraph"`, or `"chars"` |
| `boundaries` | `list[str]` | `[]` | Regex patterns marking section starts |
| `fallback` | `str` | `"paragraph"` | Sub-split strategy: `"paragraph"`, `"sentence"`, `"word"` |
| `min_size` | `int` | `200` | Minimum chunk size (merge smaller segments) |
| `sentence_pattern` | `str \| Pattern \| None` | `None` | Custom regex for sentence detection (default: English) |
| `keep_together` | `list[str] \| None` | `None` | Patterns for lines that must stay with next segment |

### `Chunker.chunk(text) → list[str]`

Returns a list of chunk strings.

### `Chunker.chunk_with_metadata(text) → list[Chunk]`

Returns a list of `Chunk` objects with full metadata.

### `Chunk`

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Full chunk content (including overlap) |
| `start` | `int` | Start offset in original text (excluding overlap) |
| `end` | `int` | End offset in original text |
| `index` | `int` | Zero-based chunk index |
| `boundary_type` | `str` | What triggered the split |
| `overlap_text` | `str` | The overlap prefix |
| `content_text` | `str` | Text without overlap (property) |

## Known limitations

- **Sentence detection** defaults to a simple regex (`[.!?]\s+(?=[A-Z"(])`). Abbreviations like "Dr. Smith" may cause false splits. For non-English or informal text, pass `sentence_pattern` — built-in alternatives: `SENTENCE_END_CJK`, `SENTENCE_END_PERMISSIVE`.
- **Boundaries are line-level** regex matches — they won't detect inline structural markers.
- **No tokenizer awareness** — `target_size` is in characters, not tokens. For token budgets, estimate `tokens ≈ chars / 4`.
- **Flat hierarchy** — all boundary patterns are equal. A `(1)` inside Article 5 matches the same as `(1)` at document level. For deeply nested structures, consider scoping your patterns more tightly.

## License

MIT
