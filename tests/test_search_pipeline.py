from datetime import datetime

from it_geek_search.generator import generate_chat_dataset, generate_benchmark_queries
from it_geek_search.retrieval import build_search_index, search_messages


def test_dataset_generation_creates_valid_chat_corpus():
    data = generate_chat_dataset(seed=42)

    assert len(data) >= 4000
    assert data["sender"].nunique() == 8
    assert data["timestamp"].min() < data["timestamp"].max()
    assert data["text"].str.len().gt(0).all()
    assert data["text"].str.contains("hackathon|server|meeting|final|cost|schedule|code", case=False, regex=True).any()


def test_benchmark_queries_cover_zero_overlap_and_diverse_intents():
    queries = generate_benchmark_queries()

    assert len(queries) >= 40
    assert sum(1 for q in queries if q["zero_overlap"] is True) >= 8
    assert {"semantic", "attributed", "temporal", "hybrid"}.issubset({q["type"] for q in queries})


def test_hybrid_search_can_filter_by_sender_and_time():
    data = generate_chat_dataset(seed=7)
    chunks = build_search_index(data)
    results = search_messages(
        chunks,
        query="weekend hackathon final karlo",
        sender="Rohan",
        start_date="2025-02-01",
        end_date="2025-03-31",
    )

    assert results
    assert all(r["sender"] == "Rohan" for r in results)
    assert any("hackathon" in r["text"].lower() or "final" in r["text"].lower() for r in results)


def test_semantic_trip_query_surfaces_manali_decision():
    data = generate_chat_dataset(seed=42)
    texts = " ".join(data["text"].str.lower())

    assert "manali" in texts
    chunks = build_search_index(data)
    results = search_messages(chunks, "when did we decide on Manali?", top_k=5)

    assert results
    assert any("manali" in r["text"].lower() for r in results)
