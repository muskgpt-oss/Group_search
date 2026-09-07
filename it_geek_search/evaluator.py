from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import pandas as pd

from .retrieval import search_messages, SearchIndex
from .generator import generate_benchmark_queries


def evaluate_retrieval(
    index: SearchIndex,
    queries: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    use_auto_parse: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates the retrieval pipeline against benchmark queries.
    Computes Hit@1, Hit@3, Hit@5, and Mean Reciprocal Rank (MRR).
    Also provides segmented metrics for zero-overlap queries and query types.
    """
    if queries is None:
        queries = generate_benchmark_queries()

    detailed_results = []
    total_queries = len(queries)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    latencies = []

    for q in queries:
        prompt = q["prompt"]
        target = q.get("target_content", "").lower()
        q_type = q.get("type", "unknown")
        is_zero_overlap = q.get("zero_overlap", False)

        start_time = time.perf_counter()
        
        if use_auto_parse:
            # Let query parser extract sender and dates from prompt
            results = search_messages(
                index,
                query=prompt,
                top_k=top_k,
                auto_parse=True,
            )
        else:
            # Use explicit metadata filters if present in query definition
            results = search_messages(
                index,
                query=prompt,
                sender=q.get("expected_sender"),
                start_date=q.get("start_date"),
                end_date=q.get("end_date"),
                top_k=top_k,
                auto_parse=False,
            )
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        # Check for target in results
        hit_rank = None
        for rank, res in enumerate(results, start=1):
            text = res.get("text", "").lower()
            context_texts = [
                m.get("text", "").lower()
                for m in res.get("context", [])
            ]
            all_text = text + " " + " ".join(context_texts)
            if target and target in all_text:
                hit_rank = rank
                break

        if hit_rank == 1:
            hits_at_1 += 1
        if hit_rank and hit_rank <= 3:
            hits_at_3 += 1
        if hit_rank and hit_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / hit_rank) if hit_rank else 0.0
        reciprocal_ranks.append(rr)

        detailed_results.append({
            "id": q.get("id", ""),
            "prompt": prompt,
            "type": q_type,
            "zero_overlap": is_zero_overlap,
            "target": target,
            "hit_rank": hit_rank if hit_rank is not None else -1,
            "reciprocal_rank": rr,
            "latency_ms": round(elapsed_ms, 2),
            "top_result_snippet": results[0]["text"][:80] + "..." if results else "No result"
        })

    mrr = sum(reciprocal_ranks) / total_queries if total_queries else 0.0
    hit_1_rate = hits_at_1 / total_queries if total_queries else 0.0
    hit_3_rate = hits_at_3 / total_queries if total_queries else 0.0
    hit_5_rate = hits_at_5 / total_queries if total_queries else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Zero-overlap metrics
    zero_overlap_queries = [r for r in detailed_results if r["zero_overlap"]]
    zero_overlap_mrr = (
        sum(r["reciprocal_rank"] for r in zero_overlap_queries) / len(zero_overlap_queries)
        if zero_overlap_queries else 0.0
    )

    summary = {
        "total_queries": total_queries,
        "mrr": round(mrr, 4),
        "hit@1": round(hit_1_rate, 4),
        "hit@3": round(hit_3_rate, 4),
        "hit@5": round(hit_5_rate, 4),
        "zero_overlap_mrr": round(zero_overlap_mrr, 4),
        "zero_overlap_count": len(zero_overlap_queries),
        "avg_latency_ms": round(avg_latency, 2),
    }

    return {
        "summary": summary,
        "details": detailed_results,
    }

