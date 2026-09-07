"""Context-aware Hinglish chat search package."""

from .generator import generate_chat_dataset, generate_benchmark_queries
from .chunking import chunk_messages
from .retrieval import build_search_index, search_messages, SearchIndex
from .db import MessageDatabase
from .query_parser import parse_query, ParsedQuery
from .evaluator import evaluate_retrieval

__all__ = [
    "generate_chat_dataset",
    "generate_benchmark_queries",
    "chunk_messages",
    "build_search_index",
    "search_messages",
    "SearchIndex",
    "MessageDatabase",
    "parse_query",
    "ParsedQuery",
    "evaluate_retrieval",
]

