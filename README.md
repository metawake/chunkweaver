# chunkweaver

**The structure-aware chunking layer between your extractor and your vector DB.**

[![PyPI version](https://img.shields.io/pypi/v/chunkweaver.svg)](https://pypi.org/project/chunkweaver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Where chunkweaver fits

```
  PDF / DOCX / HTML              Your vector DB
        │                              ▲
        ▼                              │
  ┌───────────┐    ┌──────────────┐    │
  │ Extractor │───▶│ chunkweaver  │────┘
  │           │    │              │
  │ unstructured   │ boundaries   │  Embed + upsert
  │ marker-pdf     │ detectors    │  into Pinecone,
  │ docling        │ presets      │  Qdrant, Weaviate,
  │ pdfminer       │ overlap      │  ChromaDB, etc.
  └───────────┘    └──────────────┘
```

Your extractor turns files into text. Your vector DB stores embeddings.
chunkweaver sits in the middle — splitting that text at structural
boundaries so each chunk is a coherent unit of meaning, not an arbitrary
slice of characters.

## The problem

Documents have non-uniform information density. Within a section, each
sentence constrains the next — information is coherent. At a section
boundary, the topic shifts — a new context begins. Standard chunkers,
including LangChain's `RecursiveCharacterTextSplitter`, treat this
density as uniform. They don't know that "Article 17" starts a new legal
section, or that a financial table's header row belongs with its data.
The result: chunks that straddle topic boundaries, producing blurry
embeddings and incomplete retrievals.

The fix is cheap: in structured documents, surface markers (headings,
article numbers, table rules) are reliable proxies for where topics
change. Detecting `^Article\s+\d+` costs O(n) character comparisons —
orders of magnitude less than computing semantic boundaries from
embeddings — and captures most of the boundary information.

<p align="center">
  <img src="assets/fixed-vs-structure-aware.png" alt="Fixed-size chunking cuts through sentences; structure-aware splits follow logical sections" width="320" />
</p>

Our [LLM-as-judge benchmark](benchmark/README.md) on 14 structured documents
across four domains (legal, technical, medical, financial) and 58 queries shows:

| Baseline | CW wins | Baseline wins | p-value |
|----------|---------|---------------|---------|
| Naive 600-char | **15** | 4 | **0.019** |
| LangChain RCTS | **11** | 4 | 0.119 |

chunkweaver significantly outperforms naive chunking (p < 0.02). Against
LangChain RCTS, the win ratio is similar (11:4) but not significant at
this sample size — RCTS's paragraph heuristic captures some structural
signal, narrowing the gap. The advantage is clearest on documents with
explicit section markers. See [benchmark/](benchmark/) for full results,
methodology, and reproduction steps.

## What chunkweaver does

Three layers of structure-aware splitting:

1. **Regex boundaries** — you tell the chunker where sections start (`^Article \d+`, `^## `, `^Item 1.`)
2. **Hierarchical levels** — CHAPTER splits always; Article splits only when CHAPTER is oversized; recitals split only when Article is oversized
3. **Heuristic detectors** — the chunker discovers structure itself (headings by casing/whitespace, tables by numeric patterns)

All layers work together. Detectors can emit **split points** ("start a new chunk here") or **keep-together regions** ("don't split this table"). Upstream extractors can inject pre-computed annotations directly. When they conflict, keep-together wins.

- **Zero dependencies** — stdlib only, no LangChain/LlamaIndex tax
- **Hierarchical boundaries** — CHAPTER > Section > Article > clause, split only as deep as needed
- **User-defined boundaries** — regex patterns, not hard-coded heuristics
- **Annotation ingestion** — accept pre-computed structure from any extractor
- **Heuristic detectors** — `HeadingDetector`, `TableDetector` for semi-structured documents
- **Semantic overlap** — sentences, not characters
- **Full metadata** — offsets, boundary types, hierarchy levels, overlap tracking
- **Integrations** — LangChain and LlamaIndex drop-ins; Unstructured planned

## FAQ — "What if I need to..."

| Question | Answer |
|----------|--------|
| **...chunk PDFs or DOCX files?** | Yes — run your file through any extractor that outputs text/markdown (marker-pdf, docling, pdfminer, Azure Document Intelligence), then feed the result to chunkweaver. It operates on text, not file formats. |
| **...keep tables from being split?** | Yes — `TableDetector()` marks tables as keep-together regions. The chunker won't cut inside them. See [Financial documents](#financial-documents-tables--headings). |
| **...handle OCR-damaged headings like `D E F I N I T I O N S`?** | Yes — `chunkweaver --recommend` detects letterspacing artifacts and suggests `MLOCRHeadingDetector`. See [examples/ml-detectors/](examples/ml-detectors/). |
| **...split on custom section markers?** | Yes — pass any regex as `boundaries=[r"^Article\s+\d+"]`. See [Custom boundaries](#custom-boundaries-for-any-domain). |
| **...keep chapters intact but split oversized ones at articles?** | Yes — use leveled presets like `LEGAL_EU_LEVELED`. Level 0 always splits; deeper levels only split when needed. See [Hierarchical boundaries](#hierarchical-boundaries). |
| **...pass structure from my PDF extractor?** | Yes — pass `SplitPoint` and `KeepTogetherRegion` annotations directly via the `annotations` parameter. See [Annotation ingestion](#annotation-ingestion-from-extractors). |
| **...auto-detect the right config for my document?** | Yes — `chunkweaver --recommend myfile.txt` analyzes structure and suggests presets, detectors, and target size. |
| **...use it with LangChain?** | Yes — `ChunkWeaverSplitter` is a drop-in `TextSplitter`. `pip install chunkweaver[langchain]`. |
| **...use it with LlamaIndex?** | Yes — `ChunkWeaverNodeParser` is a drop-in `NodeParser`. `pip install chunkweaver[llamaindex]`. |
| **...chunk Chinese / Japanese / Korean text?** | Yes — use `SENTENCE_END_CJK` for sentence splitting. See [CJK text](#chinese--japanese--korean-text). |
| **...chunk chat logs or support transcripts?** | Yes — `CHAT` preset splits on speaker turns and timestamps. See [Chat logs](#chat-logs--customer-support-transcripts). |
| **...chunk clinical notes?** | Yes — `CLINICAL` preset recognizes `HPI:`, `ASSESSMENT:`, `PLAN:`, etc. See [Healthcare / clinical notes](#healthcare--clinical-notes). |
| **...chunk FDA drug labels?** | Yes — `FDA_LABEL` / `FDA_LABEL_LEVELED` presets split on numbered sections and subsections. See [FDA drug labels](#fda-drug-labels). |
| **...chunk SEC 10-K filings?** | Yes — `SEC_10K` / `SEC_10K_LEVELED` presets split on `PART`/`Item` hierarchy plus ALL-CAPS sub-headings. See [SEC filings](#sec-filings-10-k). |
| **...write my own structure detector?** | Yes — subclass `BoundaryDetector` and return `SplitPoint` / `KeepTogetherRegion`. See [Custom detectors](#custom-detectors). |
| **...get token counts instead of character counts?** | Not directly — `target_size` is in characters. Estimate `tokens ≈ chars / 4`. |
| **...check if my chunking config is good?** | Yes — `chunkweaver --inspect myfile.txt` analyzes chunk quality, flags problems (oversized chunks, high fallback ratio, orphan headings), and suggests fixes. |
| **...get LLM-based quality feedback on chunks?** | Yes — `chunkweaver --inspect --llm-audit myfile.txt` rates each chunk's semantic coherence via GPT-4o-mini (requires `OPENAI_API_KEY`). |
| **...call a remote ML server or API from a detector?** | Yes — `BoundaryDetector.detect()` can do anything internally (HTTP, gRPC, etc.). Use `concurrent=True` on the Chunker to fan out multiple detectors in parallel. |
| **...install any ML or NLP dependencies?** | No — core is stdlib-only. ML detectors are optional examples, not requirements. |

## Install

```bash
pip install chunkweaver
```

Extras:

```bash
pip install chunkweaver[cli]        # CLI with click
pip install chunkweaver[langchain]   # LangChain TextSplitter integration
pip install chunkweaver[llamaindex]  # LlamaIndex NodeParser integration
pip install chunkweaver[dev]         # pytest + coverage
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
    CHAT, CLINICAL, FINANCIAL, FINANCIAL_TABLE,
    SEC_10K, FDA_LABEL, PLAIN,
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
| `SEC_10K` | SEC annual reports | `PART I`–`IV`, `Item N.`, ALL-CAPS sub-headings |
| `FDA_LABEL` | Drug labels | `1 INDICATIONS`, `## 2.1 Adult Dosage` |
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

## LlamaIndex integration

Drop-in `NodeParser` for LlamaIndex ingestion pipelines:

```python
from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser
from chunkweaver.presets import LEGAL_EU

parser = ChunkWeaverNodeParser(
    target_size=1024,
    boundaries=LEGAL_EU,
    overlap=2,
)

# Works with LlamaIndex document loaders and ingestion pipelines
nodes = parser.get_nodes_from_documents(documents)
```

Supports all chunkweaver features including detectors:

```python
from chunkweaver.detector_heading import HeadingDetector

parser = ChunkWeaverNodeParser(
    target_size=1024,
    detectors=[HeadingDetector()],
)
```

Requires: `pip install chunkweaver[llamaindex]`

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

# Analyze a document and get configuration recommendations
chunkweaver --recommend my_document.txt
```

The `--recommend` flag scans a document for structural signals and suggests
which preset, detectors, and `target_size` to use — with a ready-to-paste
Python snippet:

```
=== chunkweaver recommend ===

Document: 12,340 chars, 380 lines, 45 paragraphs
Avg paragraph: ~274 chars

--- Preset matching ---
  legal-eu                8 hits <-- best
  financial               3 hits

--- Detectors ---
  HeadingDetector: YES (12 headings found)
  TableDetector:   YES (2 tables found)

--- Python snippet ---
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.detector_table import TableDetector

chunker = Chunker(
    target_size=1024,
    overlap=2,
    boundaries=LEGAL_EU,
    detectors=[HeadingDetector(), TableDetector()],
)
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

### FDA drug labels

FDA prescribing information follows a standardized structure: numbered
top-level sections (1 INDICATIONS, 2 DOSAGE, …) with numbered
subsections (2.1 Adult Dosage, 5.1 Lactic Acidosis, …):

```python
from chunkweaver import Chunker
from chunkweaver.presets import FDA_LABEL_LEVELED

chunker = Chunker(
    target_size=2048,
    overlap=2,
    overlap_unit="sentence",
    boundaries=FDA_LABEL_LEVELED,
    # Level 0: "1 INDICATIONS AND USAGE" — always splits
    # Level 1: "## 2.1 Adult Dosage" — splits only if section is oversized
)

chunks = chunker.chunk(prescribing_info_text)
```

Tested on the full Metformin prescribing information (42K chars,
15 sections, 23 subsections). At 4K target, the CONTRAINDICATIONS
section stays intact as a single coherent chunk.

### SEC filings (10-K)

SEC annual reports follow a PART → Item → sub-heading hierarchy:

```python
from chunkweaver import Chunker
from chunkweaver.presets import SEC_10K_LEVELED

chunker = Chunker(
    target_size=2048,
    overlap=2,
    overlap_unit="sentence",
    boundaries=SEC_10K_LEVELED,
    # Level 0: "PART I" — always splits
    # Level 1: "Item 1. BUSINESS" — splits only if PART is oversized
    # Level 2: ALL-CAPS sub-headings — splits only if Item is oversized
)

chunks = chunker.chunk(filing_text)
```

Handles both `Item` and `ITEM` casing (common in EDGAR filings).
Tested on the Enron 10-K (276K chars, 4 PARTs, 14 Items, ~40
sub-headings). For heavier table coverage, combine with `TableDetector`:

```python
from chunkweaver.detector_table import TableDetector

chunker = Chunker(
    target_size=2048,
    boundaries=SEC_10K_LEVELED,
    detectors=[TableDetector()],
)
```

### Financial documents (tables + headings)

The biggest problems with financial document chunking: tables get split
in half, and section headings get separated from their content.

**Option A — regex `keep_together`** (simple, for known table markers):

```python
from chunkweaver import Chunker
from chunkweaver.presets import FINANCIAL, FINANCIAL_TABLE

chunker = Chunker(
    target_size=1024,
    boundaries=FINANCIAL + FINANCIAL_TABLE,
    keep_together=[r"^TABLE\s+\d+"],
)
```

**Option B — heuristic detectors** (discovers structure automatically):

```python
from chunkweaver import Chunker
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.detector_table import TableDetector
from chunkweaver.presets import FINANCIAL

chunker = Chunker(
    target_size=1024,
    boundaries=FINANCIAL,
    detectors=[HeadingDetector(), TableDetector()],
)
```

Option B finds headings by casing/whitespace patterns and tables by
numeric run detection — no regex tuning needed. On SEC 10-K filings,
`TableDetector` keeps 80% of financial tables intact vs. 21% without it.

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

## Hierarchical boundaries

Flat boundaries treat every pattern equally — `CHAPTER`, `Article`, and `(1)` all
produce the same split. Hierarchical boundaries assign **levels**: level 0 always
splits; deeper levels only split when the parent segment exceeds `target_size`.

This means: if an entire chapter fits in one chunk, it stays as one chunk. If it's
too big, chunkweaver splits at Article boundaries. If an article is still too big,
it splits at recital boundaries. The chunker descends only as deep as needed.

```python
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU_LEVELED

chunker = Chunker(
    target_size=2048,
    overlap=2,
    boundaries=LEGAL_EU_LEVELED,
    # CHAPTER (level 0) → always splits
    # SECTION (level 1) → splits only if chapter is oversized
    # Article (level 2) → splits only if section is oversized
    # (1) recital (level 3) → splits only if article is oversized
)
```

Any boundary can be leveled by passing a `(pattern, level)` tuple instead of a
plain string:

```python
# Custom hierarchy for your domain
boundaries = [
    (r"^PART\s+[IVX]+", 0),      # strongest boundary
    (r"^Section\s+\d+", 1),       # splits only if PART is oversized
    (r"^\d+\.\d+\s", 2),          # splits only if Section is oversized
]
```

Mix with flat strings freely — plain strings default to level 0:

```python
boundaries = LEGAL_EU_LEVELED + [r"^Annex\s+"]  # Annex = level 0
```

### Leveled presets

| Preset | Hierarchy |
|--------|-----------|
| `LEGAL_EU_LEVELED` | CHAPTER > SECTION > Article > (N) recital |
| `LEGAL_US_LEVELED` | PART > Section/§ > clause |
| `RFC_LEVELED` | top-level section > subsection |
| `MARKDOWN_LEVELED` | `#` > `##` > `###` > `####`+ |
| `FINANCIAL_LEVELED` | PART > Item/NOTE/Schedule > TABLE |
| `SEC_10K_LEVELED` | PART > Item > ALL-CAPS sub-heading |
| `FDA_LABEL_LEVELED` | numbered section > numbered subsection |

Flat presets (`LEGAL_EU`, `RFC`, etc.) are unchanged and fully backward compatible.

## Annotation ingestion from extractors

When your upstream tool (Unstructured, Docling, Azure Document Intelligence,
a custom PDF parser) already knows where sections and tables are, you can pass
that structure directly — no regex needed:

```python
from chunkweaver import Chunker, SplitPoint, KeepTogetherRegion

chunker = Chunker(
    target_size=1024,
    annotations=[
        SplitPoint(position=0, line_number=0, label="Title", level=0),
        SplitPoint(position=1200, line_number=45, label="Section 2", level=1),
        KeepTogetherRegion(start=3000, end=3800, label="Revenue Table"),
    ],
)

chunks = chunker.chunk(document_text)
```

Annotations are merged with regex boundaries and detector output — use any
combination of all three:

```python
chunker = Chunker(
    target_size=1024,
    boundaries=LEGAL_EU_LEVELED,          # regex patterns
    detectors=[HeadingDetector()],         # heuristic detectors
    annotations=[                          # pre-computed from extractor
        KeepTogetherRegion(start=5000, end=6200, label="table"),
    ],
)
```

This architecture makes chunkweaver the **universal chunking layer** downstream
of any extractor — the extractor does the hard layout/vision work, chunkweaver
consumes the structured output and handles sizing, overlap, and merge logic.

## How it works

1. **Run detectors + merge annotations** — heuristic detectors and pre-computed annotations produce split points + keep-together regions
2. **Detect boundaries** — scan each line against your regex patterns, merge with detector/annotation split points, suppress splits inside keep-together regions
3. **Hierarchical split** — split at level-0 boundaries first; for oversized segments, descend to level-1 boundaries, then level-2, etc. (flat when all boundaries share level 0)
4. **Isolate protected regions** — carve keep-together regions (tables) into their own segments
5. **Sub-split oversized segments** — break large sections at paragraph → sentence → word boundaries; allow protected regions to overshoot `target_size`
6. **Merge undersized segments** — combine tiny segments (like standalone headings) with their body text; hierarchy-aware (chapter headings merge into first article)
7. **Add overlap** — prepend the last N sentences/paragraphs/chars from the previous chunk
8. **Return** — chunks with full metadata (offsets, boundary type, hierarchy level, overlap tracking)

## Heuristic detectors

For documents without clean regex-matchable section markers — SEC filings,
scanned contracts, extracted PDFs — chunkweaver provides heuristic
detectors that discover structure from text patterns.

### HeadingDetector

Scores each line on multiple signals (casing, length, whitespace context,
known prefixes) and emits split points at probable headings.

```python
from chunkweaver import Chunker
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.presets import FINANCIAL

chunker = Chunker(
    target_size=1024,
    boundaries=FINANCIAL,
    detectors=[HeadingDetector(min_score=4.0)],
)
```

Works well on documents with Title Case or ALL CAPS headings — SEC
filings, legal contracts, government reports, technical manuals.

### TableDetector

Identifies runs of numeric data lines, extends backward to include
column headers, and marks them as keep-together regions. The chunker
will not split inside a protected table (allowing up to 1.5x
`target_size` overshoot).

```python
from chunkweaver import Chunker
from chunkweaver.detector_table import TableDetector
from chunkweaver.presets import FINANCIAL

chunker = Chunker(
    target_size=1024,
    boundaries=FINANCIAL,
    detectors=[TableDetector()],
)
```

On SEC 10-K filings, TableDetector keeps **80% of financial tables
intact** (vs. 21% without it).

### Custom detectors

Implement the `BoundaryDetector` ABC to add your own structure
detection:

```python
from chunkweaver import BoundaryDetector, SplitPoint, KeepTogetherRegion

class MyDetector(BoundaryDetector):
    def detect(self, text):
        results = []
        # Emit SplitPoint where you want chunk breaks
        # Emit KeepTogetherRegion for ranges that must stay whole
        return results

chunker = Chunker(
    target_size=1024,
    detectors=[MyDetector()],
)
```

## Architecture

```
chunkweaver/
├── __init__.py            # Public API: Chunker, Chunk, detectors, sentence patterns
├── chunker.py             # Core algorithm: hierarchical split + detector + annotation merge
├── detectors.py           # BoundaryDetector ABC, SplitPoint, KeepTogetherRegion
├── detector_heading.py    # HeadingDetector — heuristic heading detection
├── detector_table.py      # TableDetector — financial table keep-together
├── models.py              # Chunk dataclass (with boundary_level)
├── boundaries.py          # Regex boundary detection engine (with BoundarySpec levels)
├── sentences.py           # Configurable sentence splitting (regex, no NLP)
├── presets.py             # 11 flat presets + 7 leveled presets
├── recommend.py           # Document analysis and config recommendations
├── inspect.py             # Post-chunking diagnostics and LLM audit
├── cli.py                 # CLI entry point
└── integrations/
    ├── langchain.py       # LangChain TextSplitter wrapper
    └── llamaindex.py      # LlamaIndex NodeParser wrapper
```

**Design principles:**
- Each module has a single responsibility
- No deeply nested conditionals — small, testable functions
- All decisions are logged/exposed via chunk metadata
- Zero dependencies for core; optional extras for CLI and integrations
- Detectors and annotations are composable — stack any combination without conflicts
- Hierarchical boundaries degrade gracefully to flat splitting when all levels are equal

## API reference

### `Chunker(target_size, overlap, overlap_unit, boundaries, fallback, min_size, sentence_pattern, keep_together, detectors, annotations, concurrent)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | `int` | `1024` | Target chunk size in characters |
| `overlap` | `int` | `2` | Number of overlap units from previous chunk |
| `overlap_unit` | `str` | `"sentence"` | `"sentence"`, `"paragraph"`, or `"chars"` |
| `boundaries` | `list[BoundarySpec]` | `[]` | Regex patterns or `(regex, level)` tuples marking section starts |
| `fallback` | `str` | `"paragraph"` | Sub-split strategy: `"paragraph"`, `"sentence"`, `"word"` |
| `min_size` | `int` | `200` | Minimum chunk size (merge smaller segments) |
| `sentence_pattern` | `str \| Pattern \| None` | `None` | Custom regex for sentence detection (default: English) |
| `keep_together` | `list[str] \| None` | `None` | Patterns for lines that must stay with next segment |
| `detectors` | `list[BoundaryDetector] \| None` | `None` | Heuristic detectors for structure discovery |
| `annotations` | `list[Annotation] \| None` | `None` | Pre-computed `SplitPoint` / `KeepTogetherRegion` from extractors |
| `concurrent` | `bool` | `False` | Run detectors in parallel via `ThreadPoolExecutor` |

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
| `boundary_level` | `int` | Hierarchy level of the boundary (0 = strongest) |
| `overlap_text` | `str` | The overlap prefix |
| `content_text` | `str` | Text without overlap (property) |

## Part of the RAG retrieval quality ecosystem

chunkweaver is one of three tools that together cover the full
pre-deployment diagnostic for RAG retrieval:

| Tool | Role | What it measures |
|------|------|-----------------|
| **[chunkweaver](https://github.com/metawake/chunkweaver)** | Ingestion | Structure-aware chunking — controls *what text enters the prompt* |
| **[ragtune](https://github.com/metawake/ragtune)** | Evaluation | Retrieval metrics (Recall@K, MRR, bootstrap CI) — measures *how well your pipeline retrieves* |
| **[ragprobe](https://github.com/metawake/ragprobe)** | Pre-deployment | Domain difficulty analysis — predicts *how hard retrieval will be* before you build |

Each tool measures one thing well. Together they answer the three questions
that determine whether a RAG system will work in production:

1. **Is my chunking good?** — `chunkweaver --inspect` + `chunkweaver --recommend`
2. **Is my retrieval good?** — `ragtune simulate` + `ragtune --ci`
3. **Is my domain hard?** — `ragprobe` difficulty scoring

```bash
# Typical workflow
chunkweaver --recommend my_doc.txt        # what config to use
chunkweaver my_doc.txt --preset legal-eu  # chunk it

ragtune ingest ./chunks --pre-chunked     # embed + store
ragtune simulate --queries golden.json    # measure retrieval

ragprobe analyze --corpus ./docs          # how hard is this domain
```

## Known limitations

- **Sentence detection** defaults to a simple regex (`[.!?]\s+(?=[A-Z"(])`). Abbreviations like "Dr. Smith" may cause false splits. For non-English or informal text, pass `sentence_pattern` — built-in alternatives: `SENTENCE_END_CJK`, `SENTENCE_END_PERMISSIVE`.
- **Boundaries are line-level** regex matches — they won't detect inline structural markers.
- **No tokenizer awareness** — `target_size` is in characters, not tokens. For token budgets, estimate `tokens ≈ chars / 4`.

## License

MIT
