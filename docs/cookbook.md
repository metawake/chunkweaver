# Cookbook

Domain-specific recipes and advanced configuration for chunkweaver.

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

## Chat logs & customer support transcripts

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

## Chinese / Japanese / Korean text

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

## Cyrillic, accented Latin, and other non-English scripts

The core chunker is script-agnostic — it operates on Python `str` (Unicode),
uses `\S+\s*` for word splitting, and measures size in characters. Paragraphs
split on `\n\s*\n`, which is universal. **No NLP models or language packs
are involved.**

The one thing that needs attention is **sentence splitting**. The default
regex `([.!?])(\s+)(?=[A-Z"(])` requires an ASCII uppercase letter after
the punctuation, so it won't fire for:

- **Cyrillic** (Serbian, Montenegrin, Russian, Ukrainian, Bulgarian) —
  sentences start with `А-Я`, not `A-Z`.
- **Accented Latin** (Spanish `¡Á`, French `É`, Turkish `İ`, etc.) —
  accented capitals fall outside `[A-Z]`.

Use `SENTENCE_END_PERMISSIVE` — it splits on `.!?` followed by whitespace
regardless of what comes next:

```python
from chunkweaver import Chunker, SENTENCE_END_PERMISSIVE

chunker = Chunker(
    target_size=1024,
    overlap=2,
    overlap_unit="sentence",
    sentence_pattern=SENTENCE_END_PERMISSIVE,
)

# Serbian Cyrillic
text = "Члан 1 дефинише обим заштите. Члан 2 одређује услове примене."
chunks = chunker.chunk(text)

# Spanish
text = "El artículo 1 define el alcance. Ángela revisó las condiciones."
chunks = chunker.chunk(text)
```

For tighter control, write a script-aware pattern:

```python
import re

# Latin + Cyrillic uppercase after sentence-ending punctuation
SENTENCE_END_MULTI = re.compile(
    r'([.!?])(\s+)(?=[A-ZÀ-ÖØ-ÞА-ЯЂЈЉЊЋЏЁҐЄІЇ¿¡"(])'
)

chunker = Chunker(
    target_size=1024,
    sentence_pattern=SENTENCE_END_MULTI,
)
```

**Presets and detectors** are domain-specific (EU law, SEC filings, FDA
labels, etc.) and use English/Latin structural markers. They won't match
Cyrillic or non-English headings, but this is harmless — unmatched patterns
simply don't fire. Write custom `boundaries` for your document structure:

```python
# Montenegrin/Serbian legal document
boundaries = [
    r"^Члан\s+\d+",          # Član (Article)
    r"^Одељак\s+",            # Odeljak (Section)
    r"^ГЛАВА\s+[IVXLC]+",    # GLAVA (Chapter, Roman numerals)
]

# Spanish legal document
boundaries = [
    r"^Artículo\s+\d+",
    r"^Sección\s+\d+",
    r"^CAPÍTULO\s+[IVXLC]+",
]

chunker = Chunker(target_size=1024, boundaries=boundaries)
```

`HeadingDetector` uses English title-case heuristics and noise word lists —
skip it for non-English documents or provide your own `BoundaryDetector`
subclass. `TableDetector` is mostly numeric and works across scripts.

## Healthcare / clinical notes

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

## FDA drug labels

FDA prescribing information follows a standardized structure: numbered
top-level sections (1 INDICATIONS, 2 DOSAGE, ...) with numbered
subsections (2.1 Adult Dosage, 5.1 Lactic Acidosis, ...):

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

## SEC filings (10-K)

SEC annual reports follow a PART -> Item -> sub-heading hierarchy:

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

## Financial documents (tables + headings)

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

## US contracts & legal filings

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

## Custom boundaries for any domain

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

## Custom detectors

Implement the `BoundaryDetector` ABC to add your own structure detection:

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

Detectors can do anything internally — call an HTTP API, run a local ML
model, or apply domain heuristics. Use `concurrent=True` on the Chunker
to fan out multiple detectors in parallel via `ThreadPoolExecutor`.

See [examples/ml-detectors/](../examples/ml-detectors/) for scikit-learn
based examples (OCR heading detection, clinical section detection).

## Tuning tips

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

## Ecosystem: ragtune pipeline

chunkweaver's `--export-dir` writes one `.txt` file per chunk — the exact
format [ragtune](https://github.com/metawake/ragtune) expects for
`--pre-chunked` ingestion. Three commands, no glue script:

```bash
# 1. Chunk with structure-aware boundaries
chunkweaver legal_doc.txt --preset legal-eu --export-dir ./chunks/

# 2. Embed and upsert into your vector DB
ragtune ingest ./chunks/ --pre-chunked --collection legal --embedder ollama

# 3. Measure retrieval quality
ragtune simulate --collection legal --queries golden.json
```

Compare two configs side by side:

```bash
chunkweaver doc.txt --preset legal-eu --size 1024 --export-dir ./chunks-1k/
chunkweaver doc.txt --preset legal-eu --size 512  --export-dir ./chunks-512/

ragtune ingest ./chunks-1k/  --pre-chunked --collection a --embedder ollama
ragtune ingest ./chunks-512/ --pre-chunked --collection b --embedder ollama

ragtune compare --collections a,b --queries golden.json
```

Use `--format json` on any chunkweaver command to get machine-readable
output for CI pipelines:

```bash
chunkweaver doc.txt --recommend --format json | jq .suggested_target_size
chunkweaver doc.txt --preset legal-eu --inspect --format json | jq .fallback_ratio
```

## Ecosystem: ragprobe on chunks

[ragprobe](https://github.com/metawake/ragprobe) measures domain difficulty
— how hard retrieval will be — using vocabulary specificity. Feed it your
**chunks** instead of raw documents to see whether your chunking strategy
improves or hurts term discrimination:

```python
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU
from ragprobe import DomainProbe

text = open("regulation.txt").read()
queries = ["What is the right to erasure?", "Who is the data controller?"]

# Score raw document (paragraph-level passages)
raw_report = DomainProbe(corpus=[text], queries=queries).score()

# Score after chunking
chunker = Chunker(target_size=1024, boundaries=LEGAL_EU)
chunks = chunker.chunk(text)
chunked_report = DomainProbe(corpus=chunks, queries=queries).score()

print(f"Raw specificity:     {raw_report.specificity:.2f} ({raw_report.difficulty})")
print(f"Chunked specificity: {chunked_report.specificity:.2f} ({chunked_report.difficulty})")
```

If chunked specificity is **higher**, your chunking is producing more
discriminative passages — queries map to fewer, more specific chunks.
If it's **lower**, your chunks may be too small (fragmenting key terms
across boundaries) or too large (diluting specificity).

This runs in seconds with no embeddings or vector DB — pure lexical
analysis. Use it as a fast dev-time sanity check before committing to
a full embedding pipeline.
