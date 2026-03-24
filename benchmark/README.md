# Benchmark: chunkweaver vs naive chunking

End-to-end retrieval quality comparison using **LLM-as-judge** evaluation.

## Results

**11 structured documents** (GDPR, EU AI Act, CCPA, 8 IETF RFCs),
**58 queries**, top-3 chunks judged per query, **348 total LLM judgments**.

| Metric | chunkweaver | Naive (512-char) |
|--------|-------------|------------------|
| Full-answer chunks | **23 / 174** (13%) | 3 / 174 (2%) |
| Insufficient chunks | 53 / 174 (30%) | 68 / 174 (39%) |
| Per-query wins | **18** | 3 |
| Paired wins (by rank) | **55** | 26 |

Both tests statistically significant (binomial, two-sided):
- Per-query best-of-3: **p = 0.0015**
- Paired by rank: **p = 0.0017**

### What this means

When a user queries a RAG system, chunkweaver's top-3 retrieved chunks are
**6x more likely to contain a complete answer** compared to fixed-size
chunking (13% vs 2% full-answer rate). The effect is consistent across
legal regulations (GDPR, EU AI Act, CCPA) and technical specifications
(RFCs covering JWT, OAuth, TLS, HTTP, WebSocket, HTTP/2).

### What this does NOT claim

- These results apply to **structured documents** (legislation, specs,
  medical notes). Unstructured text (blog posts, chat logs) won't see
  the same benefit.
- Document-level metrics (Recall@5, MRR) are nearly identical between
  the two approaches. The advantage is **chunk-level quality**, not
  document retrieval.
- Evaluated with a single embedding model (nomic-embed-text via Ollama)
  and a single judge (gpt-4o-mini). Results may vary with other models.

## Reproduce

### Prerequisites

- Python 3.9+ with `chunkweaver` and `openai` installed
- [Qdrant](https://qdrant.tech/) running locally (port 6333/6334)
- [Ollama](https://ollama.ai/) running locally with `nomic-embed-text`
- [ragtune](https://github.com/metawake/ragtune) binary
- An OpenAI API key (for gpt-4o-mini judge calls, ~$0.10 total)

### 1. Export chunkweaver chunks

```bash
# From the structchunk repo root
python benchmark/export_chunks.py
# → benchmark/chunks-chunkweaver/ (6636 chunks from 11 docs)
```

The script reads the corpus from `../ragtune/benchmarks/hierarchical/corpus/`,
applies domain-specific presets (LEGAL_EU for regulations, RFC for specs),
and writes one chunk per file.

### 2. Ingest both collections

```bash
RAGTUNE=../ragtune/ragtune
CORPUS=../ragtune/benchmarks/hierarchical/corpus
CHUNKS=benchmark/chunks-chunkweaver

# Chunkweaver (pre-chunked)
$RAGTUNE ingest "$CHUNKS" \
  --pre-chunked \
  --collection expanded-chunkweaver \
  --embedder ollama --store qdrant

# Naive baseline (512-char fixed-size)
$RAGTUNE ingest "$CORPUS" \
  --chunk-size 512 --chunk-overlap 64 \
  --collection expanded-naive \
  --embedder ollama --store qdrant
```

### 3. Run LLM-as-judge

```bash
export OPENAI_API_KEY=sk-...

python benchmark/llm_judge.py \
  --queries ../ragtune/benchmarks/hierarchical/queries-expanded.json \
  --cw-collection expanded-chunkweaver \
  --naive-collection expanded-naive \
  --top-k 3
```

The script:
1. Embeds all 58 queries via Ollama
2. Searches both Qdrant collections (top-3 per query)
3. Sends each (query, chunk) pair to gpt-4o-mini for a sufficiency rating
   (`full` / `partial` / `insufficient`)
4. Compares ratings and runs a two-sided binomial test
5. Saves detailed results to `runs/`

### 4. Plug in your own chunker

To test a different chunking approach, replace step 1 with your own
chunk export. Write one chunk per `.txt` file into a directory, then
use `--pre-chunked` in step 2. The judge script works with any Qdrant
collection — just change `--cw-collection`.

## Corpus

| Document | Type | Source |
|----------|------|--------|
| EU GDPR (2016/679) | Legal | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj) |
| EU AI Act (2024/1689) | Legal | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) |
| CCPA (Cal. Civ. Code 1798) | Legal | [CPPA](https://cppa.ca.gov/regulations/pdf/ccpa_statute.pdf) |
| RFC 7519 (JWT) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc7519) |
| RFC 6749 (OAuth 2.0) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc6749) |
| RFC 8446 (TLS 1.3) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc8446) |
| RFC 5246 (TLS 1.2) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc5246) |
| RFC 2616 (HTTP/1.1) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc2616) |
| RFC 7231 (HTTP Semantics) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc7231) |
| RFC 7540 (HTTP/2) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc7540) |
| RFC 6455 (WebSocket) | RFC | [IETF](https://www.rfc-editor.org/rfc/rfc6455) |

## Methodology

**Judge prompt:** For each (query, chunk) pair, gpt-4o-mini rates whether
the chunk alone is sufficient to answer the query:
- **full** — contains all key information for a complete answer
- **partial** — has relevant content but misses important parts
- **insufficient** — does not contain the needed information

**Statistical test:** Two-sided exact binomial test against p=0.5
(null hypothesis: both methods are equally likely to win).

**Chunking config:**
- chunkweaver: `target_size=600, min_size=100, overlap=2 sentences`,
  LEGAL_EU preset for regulations, RFC preset for specs
- Naive: `chunk_size=512, overlap=64 chars` (ragtune default)
