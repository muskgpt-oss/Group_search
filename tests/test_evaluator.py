import pytest
import tempfile
import os
from it_geek_search.generator import generate_chat_dataset, generate_benchmark_queries
from it_geek_search.retrieval import build_search_index
from it_geek_search.evaluator import evaluate_retrieval


@pytest.fixture(scope="module")
def shared_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_dir = os.path.join(tmpdir, "chroma")
        db_path = os.path.join(tmpdir, "test.db")
        data = generate_chat_dataset(seed=42, n_messages=500)
        index = build_search_index(
            data,
            chroma_path=chroma_dir,
            db_path=db_path,
            force_rebuild=True
        )
        yield index


def test_evaluate_retrieval_metrics(shared_index):
    queries = generate_benchmark_queries()[:5]
    eval_results = evaluate_retrieval(shared_index, queries=queries, top_k=5)
    
    assert "summary" in eval_results
    assert "details" in eval_results
    summary = eval_results["summary"]
    assert summary["total_queries"] == 5
    assert "mrr" in summary
    assert "hit@1" in summary
    assert "avg_latency_ms" in summary
    assert len(eval_results["details"]) == 5

