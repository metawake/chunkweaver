# Benchmarks

Evaluation code and data live under **`benchmarks/`** in two areas:

| Path | What it does |
|------|----------------|
| **`corpus/`**, `needles.json`, `run_benchmark.py` | NeedleCoverage@5 retrieval benchmark (embeddings + vector search vs. ground-truth spans). |
| `run_hierarchical.py`, `hierarchical_results.json` | Hierarchical vs flat chunking comparison across the corpus (no ML). |
| **`llm_judge/`** | LLM-as-judge export scripts, legal corpus, and `llm_judge.py` for Qdrant + OpenAI evaluation. |

Shared text files for exports and tests are in **`corpus/`**.
