# chunkweaver

**RAG chunker that respects document structure.**

[![PyPI version](https://img.shields.io/pypi/v/chunkweaver.svg)](https://pypi.org/project/chunkweaver/)
[![CI](https://github.com/metawake/chunkweaver/actions/workflows/ci.yml/badge.svg)](https://github.com/metawake/chunkweaver/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/metawake/chunkweaver/branch/main/graph/badge.svg)](https://codecov.io/gh/metawake/chunkweaver)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Install

```bash
pip install chunkweaver
```

Extras:

```bash
pip install chunkweaver[langchain]   # LangChain TextSplitter integration
pip install chunkweaver[llamaindex]  # LlamaIndex NodeParser integration
pip install chunkweaver[cli]         # CLI (included in base install)
```

## Quick start

```python
from chunkweaver import Chunker

# Minimal — just better defaults
chunker = Chunker(target_size=1024)
chunks = chunker.chunk(text)

# With structure-aware boundaries
from chunkweaver.presets import LEGAL_EU

chunker = Chunker(
    target_size=1024,
    overlap=2,
    overlap_unit="sentence",
    boundaries=LEGAL_EU,
)

chunks = chunker.chunk(text)

# With metadata
chunks = chunker.chunk_with_metadata(text)
for c in chunks:
    print(c.text)            # full chunk content
    print(c.start, c.end)    # character offsets in original
    print(c.boundary_type)   # "section" | "paragraph" | "sentence" | "word" | "keep_together"
    print(c.overlap_text)    # the overlap prefix (if any)
    print(c.content_text)    # text without overlap (for dedup)
```

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

## Why it matters

Standard chunkers, including LangChain's `RecursiveCharacterTextSplitter`,
don't know that "Article 17" starts a new legal section, or that a table's
header row belongs with its data. The result: chunks that straddle topic
boundaries, producing blurry embeddings and incomplete retrievals.

The fix is cheap: surface markers (headings, article numbers, table rules)
are reliable proxies for where topics change. Detecting them costs O(n)
character comparisons — orders of magnitude less than computing semantic
boundaries from embeddings.

Our [LLM-as-judge benchmark](benchmark/README.md) on 11 documents across
four domains and 58 queries:

| Baseline | CW wins | Baseline wins | p-value |
|----------|---------|---------------|---------|
| Naive 600-char | **15** | 4 | **0.019** |
| LangChain RCTS | **11** | 4 | 0.119 |

See [benchmark/](benchmark/) for full results, methodology, and reproduction steps.

## Features

- **Zero dependencies** — stdlib only, no LangChain/LlamaIndex tax
- **Regex boundaries** — you tell the chunker where sections start (`^Article \d+`, `^## `, `^Item 1.`)
- **Hierarchical levels** — CHAPTER > Section > Article > clause; split only as deep as needed
- **Heuristic detectors** — `HeadingDetector`, `TableDetector` discover structure from text patterns
- **Annotation ingestion** — accept pre-computed structure from any extractor
- **Semantic overlap** — sentences, not characters
- **Full metadata** — offsets, boundary types, hierarchy levels, overlap tracking
- **Integrations** — LangChain and LlamaIndex drop-ins

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

Most presets have `_LEVELED` variants with hierarchical splits:
`LEGAL_EU_LEVELED`, `LEGAL_US_LEVELED`, `RFC_LEVELED`, `MARKDOWN_LEVELED`,
`FINANCIAL_LEVELED`, `SEC_10K_LEVELED`, `FDA_LABEL_LEVELED`.
See [docs/cookbook.md](docs/cookbook.md#hierarchical-boundaries) for details.

## CLI

```bash
chunkweaver document.txt --size 1024 --overlap 2
chunkweaver legal_doc.txt --preset legal-eu --format json
chunkweaver file.txt --detect-boundaries --boundaries "^Article\s+\d+"
cat document.txt | chunkweaver --size 1024 --preset rfc
chunkweaver --recommend my_document.txt
chunkweaver --inspect my_document.txt
```

The `--recommend` flag analyzes a document and suggests the right config:

```
=== chunkweaver recommend ===

Document: 12,340 chars, 380 lines, 45 paragraphs

--- Preset matching ---
  legal-eu                8 hits <-- best

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

## Integrations

### LangChain

Drop-in replacement for `RecursiveCharacterTextSplitter`:

```python
from chunkweaver.integrations.langchain import ChunkWeaverSplitter

splitter = ChunkWeaverSplitter(
    target_size=1024,
    overlap=2,
    boundaries=[r"^#{1,3}\s"],
)
docs = splitter.create_documents([text])
```

Requires: `pip install chunkweaver[langchain]`

### LlamaIndex

Drop-in `NodeParser` for ingestion pipelines:

```python
from chunkweaver.integrations.llamaindex import ChunkWeaverNodeParser
from chunkweaver.presets import LEGAL_EU

parser = ChunkWeaverNodeParser(
    target_size=1024,
    boundaries=LEGAL_EU,
    overlap=2,
)
nodes = parser.get_nodes_from_documents(documents)
```

Requires: `pip install chunkweaver[llamaindex]`

## Heuristic detectors

For documents without clean section markers — SEC filings, scanned
contracts, extracted PDFs — heuristic detectors discover structure
from text patterns.

```python
from chunkweaver import Chunker
from chunkweaver.detector_heading import HeadingDetector
from chunkweaver.detector_table import TableDetector

chunker = Chunker(
    target_size=1024,
    detectors=[HeadingDetector(), TableDetector()],
)
```

**HeadingDetector** scores lines on casing, length, whitespace context,
and known prefixes. Works well on Title Case and ALL CAPS headings.

**TableDetector** identifies numeric data runs and marks them as
keep-together regions. On SEC 10-K filings, keeps **80% of financial
tables intact** vs. 21% without it.

Custom detectors: subclass `BoundaryDetector` and return `SplitPoint` /
`KeepTogetherRegion`. See [examples/ml-detectors/](examples/ml-detectors/)
for scikit-learn examples.

## Documentation

- **[Cookbook](docs/cookbook.md)** — domain recipes (clinical, FDA, SEC, financial, legal, chat, CJK), hierarchical boundaries, annotation ingestion, vector DB integration, tuning tips
- **[API Reference](docs/api.md)** — full parameter tables, `Chunk` attributes, algorithm details, architecture
- **[FAQ](docs/faq.md)** — "What if I need to..." for common questions
- **[Benchmark](benchmark/README.md)** — LLM-as-judge methodology, reproduction steps, raw results

## Ecosystem

Part of a RAG tools suite for retrieval quality:

| Tool | Role | What it measures |
|------|------|-----------------|
| **[chunkweaver](https://github.com/metawake/chunkweaver)** | Ingestion | Structure-aware chunking — controls *what text enters the prompt* |
| **[ragtune](https://github.com/metawake/ragtune)** | Evaluation | Retrieval metrics (Recall@K, MRR, bootstrap CI) — measures *how well your pipeline retrieves* |
| **[ragprobe](https://github.com/metawake/ragprobe)** | Pre-deployment | Domain difficulty analysis — predicts *how hard retrieval will be* before you build |

```bash
chunkweaver --recommend my_doc.txt              # what config to use
chunkweaver my_doc.txt --preset legal-eu \
    --export-dir ./chunks/                      # chunk → one .txt per chunk
ragtune ingest ./chunks/ --pre-chunked          # embed + store
ragtune simulate --queries golden.json          # measure retrieval
ragprobe analyze --corpus ./docs                # how hard is this domain
```

`--export-dir` writes ragtune-compatible files directly — no glue script needed.
Use `--format json` with `--recommend` or `--inspect` for CI-friendly output.

## Known limitations

- **Sentence detection** defaults to a simple regex (`[.!?]\s+(?=[A-Z"(])`). Abbreviations like "Dr. Smith" may cause false splits. For non-English or informal text, pass `sentence_pattern` — built-in alternatives: `SENTENCE_END_CJK`, `SENTENCE_END_PERMISSIVE`. See the [cookbook](docs/cookbook.md#cyrillic-accented-latin-and-other-non-english-scripts) for Cyrillic, Spanish, and other script-specific guidance.
- **Boundaries are line-level** regex matches — they won't detect inline structural markers.
- **No tokenizer awareness** — `target_size` is in characters, not tokens. For token budgets, estimate `tokens ≈ chars / 4`.

## Author

[Oleksii Alexapolsky](https://www.linkedin.com/in/alexey-a-181a614/) ([𝕏](https://x.com/thewake)) — building retrieval quality tools: [chunkweaver](https://github.com/metawake/chunkweaver), [ragtune](https://github.com/metawake/ragtune), [ragprobe](https://github.com/metawake/ragprobe).

## License

MIT
