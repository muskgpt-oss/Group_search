# Context-Aware Hinglish Group Chat Semantic Search Engine

A high-performance, local-first hybrid semantic and keyword search engine over noisy, code-mixed (Hinglish) group chat data, implementing all requirements from `PRD.txt`.

## Key Features

- **Synthetic Hinglish Corpus Generator**: Generates 4,000+ messages across 8 participants spanning 6 months with realistic conversational session bursts, typos, slang, and code-mixed dialogues.
- **SQLite Message Database**: Relational storage indexing timestamps, senders, and session threads for instant context reconstruction around search hits.
- **Context-Aware Sliding Window Chunking**: Preserves conversational continuity across turns while respecting session and time-gap boundaries.
- **ChromaDB Vector Store**: Precomputed dense embeddings using HuggingFace's `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for sub-millisecond similarity search.
- **Hybrid Retrieval (RRF)**: Combines dense vector similarity with sparse lexical search (`BM25Okapi`) using Reciprocal Rank Fusion.
- **Natural Language Query Parser**: Automatically detects and extracts author attribution (e.g. "What did Rohan say...") and temporal ranges (e.g. "in March", "between Jan and Feb").
- **Benchmark Evaluation Suite**: Evaluates 40 structured benchmark queries (including $\ge 8$ zero-lexical-overlap queries) with metrics for MRR, Hit@1, Hit@3, and Hit@5.
- **Streamlit Interactive UI**: Multi-tab dashboard featuring chat-style results, context expansion, filter controls, live benchmark evaluation, and corpus statistics.

## Architecture

```
User Query ---> Query Parser (Entity & Date Extraction)
                      |
        +-------------+-------------+
        |                           |
   Dense Search               Sparse Search
   (ChromaDB + ST)            (BM25Okapi)
        \                           /
         +----> Hybrid Fusion <----+
                (RRF Algorithm)
                      |
         Filter by Sender / Timeframe
                      |
        Enrich with Message Context (SQLite)
                      |
                 Search Results
```

## Quick Start

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest
```

### 3. Launch the Web Interface
```bash
streamlit run app.py
```

## Programmatic Usage

```python
from it_geek_search.generator import generate_chat_dataset, generate_benchmark_queries
from it_geek_search.retrieval import build_search_index, search_messages
from it_geek_search.evaluator import evaluate_retrieval

# 1. Generate synthetic dataset
messages = generate_chat_dataset(seed=42, n_messages=4200)

# 2. Build or load index (SQLite + ChromaDB + BM25)
index = build_search_index(
    messages,
    chroma_path="./data/chroma",
    db_path="./data/chat_history.db"
)

# 3. Query with natural language (auto-extracts sender and dates)
results = search_messages(
    index,
    query="What did Rohan say about server costs in March?",
    auto_parse=True,
    top_k=5
)

for r in results:
    print(f"[{r['timestamp']}] {r['sender']} (Score: {r['score']}): {r['text']}")
    print("Context:", len(r['context']), "surrounding messages")

# 4. Run 40-query benchmark evaluation
benchmark_results = evaluate_retrieval(index, top_k=5)
print(benchmark_results["summary"])
```

## Benchmark Evaluation

Run automated evaluation on all 40 test queries:
```bash
pytest tests/test_benchmark.py
```
This tests:
- Semantic queries (conceptual matching without lexical overlap)
- Attributed queries (filtering by specific speaker)
- Temporal queries (filtering by month or date range)
- Hybrid queries (combining domain terms with semantic intent)
