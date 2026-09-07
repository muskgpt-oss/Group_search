import pytest
import tempfile
import os
from it_geek_search.generator import generate_chat_dataset, generate_benchmark_queries
from it_geek_search.retrieval import build_search_index
from it_geek_search.evaluator import evaluate_retrieval


def test_40_benchmark_evaluation():
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_dir = os.path.join(tmpdir, "chroma")
        db_path = os.path.join(tmpdir, "test.db")
        # Generate full dataset
        data = generate_chat_dataset(seed=42, n_messages=4200)
        index = build_search_index(
            data,
            chroma_path=chroma_dir,
            db_path=db_path,
            force_rebuild=True
        )
        
        queries = generate_benchmark_queries()
        assert len(queries) == 40
        
        zero_overlap = [q for q in queries if q.get("zero_overlap")]
        assert len(zero_overlap) >= 8

        # Run evaluation
        eval_result = evaluate_retrieval(index, queries=queries, top_k=5, use_auto_parse=True)
        summary = eval_result["summary"]
        
        # Verify MRR and Hit rates are strong
        assert summary["total_queries"] == 40
        assert summary["hit@5"] >= 0.70  # At least 70% of benchmark queries found in top 5
        assert summary["mrr"] >= 0.50    # Strong MRR score across diverse intents

