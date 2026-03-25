# Benchmark: chunkweaver vs baseline chunking

End-to-end retrieval quality comparison using **LLM-as-judge** evaluation
across three chunking strategies.

## Results

### 3-way comparison

**11 structured documents** (GDPR, EU AI Act, CCPA, 8 IETF RFCs),
**58 queries**, top-3 chunks judged per query, **gpt-4o-mini** judge.

| Metric | chunkweaver | Naive 600-char | LangChain RCTS |
|--------|-------------|----------------|----------------|
| Full-answer chunks | **23 / 174 (13%)** | 11 / 174 (6%) | 15 / 174 (9%) |
| Insufficient chunks | 51 / 174 (29%) | 52 / 174 (30%) | 61 / 174 (35%) |
| Per-query wins vs CW | — | 4 | 4 |
| Per-query losses vs CW | — | **15** | **11** |
| Per-query ties | — | 39 | 43 |

**Statistical significance (vs chunkweaver):**

| Baseline | Per-query p (binomial) | Permutation p (doc-clustered) | Significant? |
|----------|----------------------|------------------------------|--------------|
| Naive 600-char | **0.019** | **0.015** | Yes |
| LangChain RCTS | 0.119 | 0.290 | No |

### Interpretation

chunkweaver significantly outperforms naive fixed-size chunking
(p < 0.02 on both tests). Chunks are 2x more likely to contain a
complete answer (13% vs 6%).

Against LangChain RCTS — the most common production chunker —
the win ratio is similar (11:4) but does not reach significance.
RCTS's paragraph-aware splitting captures some structural signal
via `\n\n` separators, narrowing the gap. Where it falls short:
it can't distinguish a paragraph break *within* a section from a
boundary *between* sections, and its character-level overlap can
split mid-sentence.

### Scope and limitations

- **Structured documents only.** The advantage comes from explicit
  section markers (articles, numbered sections, clinical headers).
  Unstructured prose won't benefit.
- **Chunk-level quality, not document retrieval.** Document-level
  metrics (Recall@5, MRR) are near-identical across all three methods.
- **Single embedding model** (nomic-embed-text) and **single judge**
  (gpt-4o-mini). Results may vary with other models.
- **58 queries across 11 documents.** Sufficient for the naive
  comparison; the RCTS comparison would benefit from a larger corpus.

## Reproduce

### Prerequisites

- Python 3.9+ with `chunkweaver`, `openai`, and `langchain-text-splitters` installed
- [Qdrant](https://qdrant.tech/) running locally (port 6333/6334)
- [Ollama](https://ollama.ai/) running locally with `nomic-embed-text`
- [ragtune](https://github.com/metawake/ragtune) binary
- An OpenAI API key (for gpt-4o-mini judge calls, ~$0.15 total)

### 1. Export chunks (all three strategies)

```bash
# chunkweaver (structure-aware)
python benchmarks/llm_judge/export_chunks.py
# → benchmarks/llm_judge/chunks-chunkweaver/

# LangChain RecursiveCharacterTextSplitter baseline
python benchmarks/llm_judge/export_langchain.py
# → benchmarks/llm_judge/chunks-langchain/
```

Naive chunks are generated at ingest time by ragtune (no export needed).

### 2. Ingest all three collections

```bash
RAGTUNE=../ragtune/ragtune
CORPUS=benchmarks/corpus

# chunkweaver (pre-chunked)
$RAGTUNE ingest benchmarks/llm_judge/chunks-chunkweaver \
  --pre-chunked \
  --collection expanded-chunkweaver \
  --embedder ollama --store qdrant

# Naive baseline (600-char fixed-size)
$RAGTUNE ingest "$CORPUS" \
  --chunk-size 600 --chunk-overlap 80 \
  --collection expanded-naive-600 \
  --embedder ollama --store qdrant

# LangChain RCTS baseline (pre-chunked)
$RAGTUNE ingest benchmarks/llm_judge/chunks-langchain \
  --pre-chunked \
  --collection expanded-langchain \
  --embedder ollama --store qdrant
```

### 3. Run LLM-as-judge

```bash
export OPENAI_API_KEY=sk-...
QUERIES=../ragtune/benchmarks/hierarchical/queries-expanded.json

# chunkweaver vs naive
python benchmarks/llm_judge/llm_judge.py \
  --queries "$QUERIES" \
  --cw-collection expanded-chunkweaver \
  --naive-collection expanded-naive-600 \
  --output runs/llm-judge-cw-vs-naive600.json

# chunkweaver vs LangChain
python benchmarks/llm_judge/llm_judge.py \
  --queries "$QUERIES" \
  --cw-collection expanded-chunkweaver \
  --naive-collection expanded-langchain \
  --output runs/llm-judge-cw-vs-langchain.json
```

The script:
1. Embeds all 58 queries via Ollama
2. Searches both Qdrant collections (top-3 per query)
3. Sends each (query, chunk) pair to gpt-4o-mini for a sufficiency rating
   (`full` / `partial` / `insufficient`)
4. Compares ratings via binomial and document-clustered permutation tests
5. Saves detailed per-query results to `runs/`

### 4. Plug in your own chunker

To test a different chunking approach, write one chunk per `.txt` file
into a directory, then use `--pre-chunked` in step 2. The judge script
works with any Qdrant collection — just change `--naive-collection`.

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

**Statistical tests:**
- Two-sided exact binomial test against p=0.5 (null: both methods equally
  likely to win)
- Document-clustered permutation test (10k permutations) to control for
  non-independence of queries from the same document

**Chunking config:**
- **chunkweaver:** `target_size=600, min_size=100, overlap=2 sentences`,
  LEGAL_EU preset for regulations, RFC preset for specs
- **Naive:** `chunk_size=600, chunk_overlap=80` (character-level fixed-size)
- **LangChain RCTS:** `chunk_size=600, chunk_overlap=80`,
  separators `["\n\n", "\n", ". ", " ", ""]`

All three methods target comparable chunk sizes to avoid size confounders.
