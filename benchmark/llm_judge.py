"""LLM-as-judge: compare answer sufficiency of chunkweaver vs naive chunks.

Queries Qdrant directly for chunk texts, then judges each chunk with GPT.
Runs a binomial significance test on the win/loss ratio.
"""

import argparse
import json
import os
import asyncio
import urllib.request
from math import comb
from openai import AsyncOpenAI

JUDGE_PROMPT = """You are evaluating whether a retrieved text chunk contains enough information to fully answer a question.

Question: {question}

Retrieved chunk:
---
{chunk}
---

Rate the chunk's sufficiency for answering the question. Respond with ONLY a JSON object:
{{"rating": "full"|"partial"|"insufficient", "reason": "<one sentence>"}}

- "full": The chunk contains all key information needed to answer the question completely.
- "partial": The chunk contains some relevant information but misses important parts.
- "insufficient": The chunk does not contain the information needed to answer."""

RANK = {"full": 3, "partial": 2, "insufficient": 1, "error": 0}


def embed_texts(texts, ollama_url="http://localhost:11434"):
    """Embed texts via Ollama API."""
    embeddings = []
    for text in texts:
        data = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
        req = urllib.request.Request(
            f"{ollama_url}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req).read())
        embeddings.append(resp["embedding"])
    return embeddings


def search_qdrant(collection, vector, top_k=3, qdrant_url="http://localhost:6333"):
    """Search Qdrant and return texts + scores."""
    data = json.dumps({
        "vector": vector,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }).encode()
    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/search",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    results = []
    for point in resp["result"]:
        results.append({
            "text": point["payload"]["text"],
            "source": point["payload"].get("source", ""),
            "score": point["score"],
        })
    return results


async def judge_one(client, query_text, chunk_text, model="gpt-4o-mini"):
    resp = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=150,
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(
                question=query_text, chunk=chunk_text[:3000])}
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"rating": "error", "reason": raw}


def binomial_p_value(wins, total):
    """Two-sided exact binomial test against p=0.5."""
    if total == 0:
        return 1.0
    k = min(wins, total - wins)
    p = 0.0
    for i in range(k + 1):
        p += comb(total, i) * 0.5**total
    return min(p * 2, 1.0)


def permutation_test_by_doc(query_ids, cw_by_query, naive_by_query, queries_lookup,
                            n_permutations=10000, seed=42):
    """Document-clustered permutation test.

    Groups queries by their source document, computes per-document win rate,
    then tests whether the observed mean win rate differs from chance by
    permuting document-level labels.
    """
    import random
    rng = random.Random(seed)

    doc_wins = {}
    for qid in query_ids:
        docs = queries_lookup.get(qid, {}).get("relevant_docs", ["unknown"])
        doc_key = docs[0] if docs else "unknown"

        cw_list = cw_by_query.get(qid, [])
        naive_list = naive_by_query.get(qid, [])
        cw_best = max((RANK.get(r["rating"], 0) for r in cw_list), default=0)
        naive_best = max((RANK.get(r["rating"], 0) for r in naive_list), default=0)

        doc_wins.setdefault(doc_key, []).append(1 if cw_best > naive_best else
                                                (-1 if naive_best > cw_best else 0))

    doc_scores = []
    for doc, outcomes in doc_wins.items():
        non_ties = [o for o in outcomes if o != 0]
        if non_ties:
            doc_scores.append(sum(non_ties) / len(non_ties))

    if not doc_scores:
        return 1.0, 0, len(doc_wins)

    observed = sum(doc_scores) / len(doc_scores)

    count_extreme = 0
    for _ in range(n_permutations):
        perm_scores = [s * rng.choice([1, -1]) for s in doc_scores]
        perm_mean = sum(perm_scores) / len(perm_scores)
        if abs(perm_mean) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return p_value, len(doc_scores), len(doc_wins)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Queries JSON file")
    parser.add_argument("--cw-collection", default="bench-chunkweaver")
    parser.add_argument("--naive-collection", default="bench-naive")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.queries) as f:
        data = json.load(f)
    queries = data.get("queries", data) if isinstance(data, dict) else data
    top_k = args.top_k

    print(f"Embedding {len(queries)} queries via Ollama...")
    query_texts = [q["text"] for q in queries]
    embeddings = embed_texts(query_texts)

    print(f"Searching Qdrant (top-{top_k}) on both collections...")
    cw_results = {}
    naive_results = {}
    for q, vec in zip(queries, embeddings):
        qid = q["id"]
        cw_results[qid] = search_qdrant(args.cw_collection, vec, top_k)
        naive_results[qid] = search_qdrant(args.naive_collection, vec, top_k)

    client = AsyncOpenAI()
    tasks = []
    task_meta = []

    for q in queries:
        qid = q["id"]
        query_text = q["text"]
        for i, res in enumerate(cw_results[qid]):
            tasks.append(judge_one(client, query_text, res["text"]))
            task_meta.append({"query_id": qid, "method": "cw", "rank": i})
        for i, res in enumerate(naive_results[qid]):
            tasks.append(judge_one(client, query_text, res["text"]))
            task_meta.append({"query_id": qid, "method": "naive", "rank": i})

    total_calls = len(tasks)
    print(f"Judging {total_calls} query-chunk pairs with gpt-4o-mini...")
    results = await asyncio.gather(*tasks)

    cw_by_query = {}
    naive_by_query = {}
    for meta, res in zip(task_meta, results):
        qid = meta["query_id"]
        bucket = cw_by_query if meta["method"] == "cw" else naive_by_query
        bucket.setdefault(qid, []).append({"rank": meta["rank"], **res})

    query_ids = [q["id"] for q in queries]

    # --- Per-chunk paired comparison ---
    cw_scores = {"full": 0, "partial": 0, "insufficient": 0, "error": 0}
    naive_scores = {"full": 0, "partial": 0, "insufficient": 0, "error": 0}
    pair_wins_cw = 0
    pair_wins_naive = 0
    pair_ties = 0

    for qid in query_ids:
        cw_list = sorted(cw_by_query.get(qid, []), key=lambda x: x["rank"])
        naive_list = sorted(naive_by_query.get(qid, []), key=lambda x: x["rank"])
        for cr, nr in zip(cw_list, naive_list):
            cw_scores[cr["rating"]] = cw_scores.get(cr["rating"], 0) + 1
            naive_scores[nr["rating"]] = naive_scores.get(nr["rating"], 0) + 1
            if RANK.get(cr["rating"], 0) > RANK.get(nr["rating"], 0):
                pair_wins_cw += 1
            elif RANK.get(nr["rating"], 0) > RANK.get(cr["rating"], 0):
                pair_wins_naive += 1
            else:
                pair_ties += 1

    # --- Per-query best-of-K comparison ---
    query_wins_cw = 0
    query_wins_naive = 0
    query_ties = 0

    print("\n" + "=" * 82)
    print(f"LLM-AS-JUDGE: Answer Sufficiency (top-{top_k} chunks, {len(query_ids)} queries)")
    print("=" * 82)
    print(f"\n{'Query':<14} {'CW best':<14} {'Naive best':<14} {'Winner':<8}")
    print("-" * 54)

    for qid in query_ids:
        cw_list = cw_by_query.get(qid, [])
        naive_list = naive_by_query.get(qid, [])
        cw_best = max((RANK.get(r["rating"], 0) for r in cw_list), default=0)
        naive_best = max((RANK.get(r["rating"], 0) for r in naive_list), default=0)
        cw_label = {3: "full", 2: "partial", 1: "insufficient"}.get(cw_best, "error")
        naive_label = {3: "full", 2: "partial", 1: "insufficient"}.get(naive_best, "error")
        if cw_best > naive_best:
            winner = "CW"
            query_wins_cw += 1
        elif naive_best > cw_best:
            winner = "Naive"
            query_wins_naive += 1
        else:
            winner = "tie"
            query_ties += 1
        print(f"{qid:<14} {cw_label:<14} {naive_label:<14} {winner:<8}")

    print("-" * 54)

    n_pairs = pair_wins_cw + pair_wins_naive
    n_queries = query_wins_cw + query_wins_naive
    p_pairs = binomial_p_value(pair_wins_cw, n_pairs)
    p_queries = binomial_p_value(query_wins_cw, n_queries)

    total_cw = sum(cw_scores.values())
    total_naive = sum(naive_scores.values())

    print("\n--- Rating Distribution ---")
    print(f"  Chunkweaver:  full={cw_scores['full']}  partial={cw_scores['partial']}  "
          f"insufficient={cw_scores['insufficient']}  (N={total_cw})")
    print(f"  Naive:        full={naive_scores['full']}  partial={naive_scores['partial']}  "
          f"insufficient={naive_scores['insufficient']}  (N={total_naive})")
    print(f"  CW full rate:    {cw_scores['full']/max(total_cw,1):.0%}")
    print(f"  Naive full rate: {naive_scores['full']/max(total_naive,1):.0%}")

    print("\n--- Paired Comparison (by rank) ---")
    print(f"  CW wins: {pair_wins_cw}   Naive wins: {pair_wins_naive}   "
          f"Ties: {pair_ties}   (out of {pair_wins_cw+pair_wins_naive+pair_ties})")
    print(f"  Binomial p-value (two-sided): {p_pairs:.4f}"
          f"  {'*** SIGNIFICANT' if p_pairs < 0.05 else '(not significant)'}")

    print(f"\n--- Per-Query Best-of-{top_k} ---")
    print(f"  CW wins: {query_wins_cw}   Naive wins: {query_wins_naive}   "
          f"Ties: {query_ties}   (out of {len(query_ids)})")
    print(f"  Binomial p-value (two-sided): {p_queries:.4f}"
          f"  {'*** SIGNIFICANT' if p_queries < 0.05 else '(not significant)'}")

    queries_lookup = {q["id"]: q for q in queries}
    perm_p, n_docs_tested, n_docs_total = permutation_test_by_doc(
        query_ids, cw_by_query, naive_by_query, queries_lookup)

    print(f"\n--- Document-Clustered Permutation Test ---")
    print(f"  Documents with non-tie queries: {n_docs_tested} / {n_docs_total}")
    print(f"  Permutation p-value (two-sided, 10k permutations): {perm_p:.4f}"
          f"  {'*** SIGNIFICANT' if perm_p < 0.05 else '(not significant)'}")

    out_path = args.output or os.path.join(
        os.path.dirname(__file__), f"../runs/llm-judge-top{top_k}-40q-results.json")
    out = {
        "experiment": "llm-as-judge-answer-sufficiency",
        "model": "gpt-4o-mini",
        "top_k": top_k,
        "num_queries": len(queries),
        "total_judge_calls": total_calls,
        "collections": {
            "chunkweaver": args.cw_collection,
            "naive": args.naive_collection,
        },
        "summary": {
            "cw_scores": cw_scores,
            "naive_scores": naive_scores,
            "pair_wins_cw": pair_wins_cw,
            "pair_wins_naive": pair_wins_naive,
            "pair_ties": pair_ties,
            "pair_p_value": p_pairs,
            "query_wins_cw": query_wins_cw,
            "query_wins_naive": query_wins_naive,
            "query_ties": query_ties,
            "query_p_value": p_queries,
            "permutation_p_value": perm_p,
            "n_docs_tested": n_docs_tested,
        },
        "per_query": [
            {
                "query_id": qid,
                "cw_judgments": sorted(cw_by_query.get(qid, []), key=lambda x: x["rank"]),
                "naive_judgments": sorted(naive_by_query.get(qid, []), key=lambda x: x["rank"]),
            }
            for qid in query_ids
        ],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
