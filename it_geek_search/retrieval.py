from __future__ import annotations

import os
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

from .chunking import chunk_messages
from .db import MessageDatabase
from .query_parser import parse_query, ParsedQuery

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL = None


def _get_embedding_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        try:
            _MODEL = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _MODEL = SentenceTransformer(model_name)
    return _MODEL



def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().replace("-", " ").split() if token]


class SearchIndex(Sequence):
    """
    SearchIndex encapsulates the SQLite message database,
    ChromaDB persistent vector store, and BM25 inverted index.
    Implements Sequence to maintain backward compatibility with list-of-chunks.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        collection: Optional[chromadb.Collection] = None,
        bm25: Optional[BM25Okapi] = None,
        db: Optional[MessageDatabase] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.chunks = chunks
        self.collection = collection
        self.bm25 = bm25
        self.db = db
        self.embeddings = embeddings
        self.chunk_by_id = {c["chunk_id"]: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: Any) -> Any:
        return self.chunks[index]

    def __iter__(self):
        return iter(self.chunks)


def build_search_index(
    messages: Union[pd.DataFrame, List[Dict[str, Any]]],
    chroma_path: Optional[str] = "./data/chroma",
    db_path: Optional[str] = "./data/chat_history.db",
    window_size: int = 7,
    stride: int = 3,
    force_rebuild: bool = False,
    batch_size: int = 64,
) -> SearchIndex:
    """
    Builds or loads the search index including:
    1. SQLite message database for context retrieval.
    2. Sliding window conversational chunks.
    3. ChromaDB vector database with precomputed embeddings.
    4. BM25 keyword index for fast lexical search.
    """
    if messages is None:
        if db_path and os.path.exists(db_path):
            db = MessageDatabase(db_path)
            messages_df = db.get_all_messages(as_df=True)
        else:
            raise ValueError("messages must be provided if database does not exist.")
    elif isinstance(messages, list):
        messages_df = pd.DataFrame(messages)
    else:
        messages_df = messages.copy()

    # 1. Initialize and populate Message Database
    db: Optional[MessageDatabase] = None
    if db_path:
        db = MessageDatabase(db_path)
        if force_rebuild or db.count() == 0:
            if force_rebuild:
                db.clear()
            db.insert_messages(messages_df)

    # 2. Chunk messages
    chunks = chunk_messages(messages_df, window_size=window_size, stride=stride)
    for c in chunks:
        c["text_lower"] = c["text"].lower()
        if "timestamp_iso" not in c:
            c["timestamp_iso"] = pd.Timestamp(c["timestamp"]).isoformat()

    if not chunks:
        return SearchIndex(chunks=[], db=db)

    # 3. Build BM25 index
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. ChromaDB vector store
    collection = None
    embeddings = None

    if chroma_path:
        os.makedirs(chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_path)
    else:
        client = chromadb.EphemeralClient()

    collection_name = "chat_chunks"

    if force_rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count == len(chunks) and not force_rebuild:
        # Already indexed
        pass
    else:
        # Reset and generate embeddings
        if existing_count > 0:
            client.delete_collection(collection_name)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        model = _get_embedding_model()
        texts = [c["text"] for c in chunks]
        embeddings_list = []

        # Batch encode
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embs = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings_list.append(batch_embs)

        if embeddings_list:
            embeddings = np.vstack(embeddings_list)
        else:
            embeddings = np.array([])

        # Add to ChromaDB in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size].tolist()
            ids = [str(c["chunk_id"]) for c in batch_chunks]
            docs = [c["text"] for c in batch_chunks]
            metadatas = [
                {
                    "chunk_id": int(c["chunk_id"]),
                    "session_id": str(c.get("session_id", "")),
                    "sender": str(c.get("sender", "")),
                    "participants": ",".join(c.get("participants", [])),
                    "start_time": str(c.get("start_time", c["timestamp_iso"])),
                    "end_time": str(c.get("end_time", c["timestamp_iso"])),
                    "start_message_id": int(c.get("start_message_id", 0)),
                    "end_message_id": int(c.get("end_message_id", 0)),
                    "topic": str(c.get("topic", "")),
                }
                for c in batch_chunks
            ]
            collection.add(ids=ids, embeddings=batch_embs, documents=docs, metadatas=metadatas)

    return SearchIndex(chunks=chunks, collection=collection, bm25=bm25, db=db, embeddings=embeddings)


def search_messages(
    index_or_chunks: Union[SearchIndex, List[Dict[str, Any]]],
    query: str,
    sender: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: int = 5,
    auto_parse: bool = False,
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Hybrid search across chat chunks combining Dense (ChromaDB/SentenceTransformers)
    and Sparse (BM25) retrieval using Reciprocal Rank Fusion (RRF).
    Supports metadata filtering on sender and date range.
    """
    if not index_or_chunks:
        return []

    parsed: Optional[ParsedQuery] = None
    if auto_parse:
        parsed = parse_query(query)
        if not sender and parsed.sender:
            sender = parsed.sender
        if not start_date and parsed.start_date:
            start_date = parsed.start_date
        if not end_date and parsed.end_date:
            end_date = parsed.end_date
        query = parsed.clean_query

    q = query.strip()
    if not q:
        q = parsed.raw_query if parsed else ""

    if isinstance(index_or_chunks, SearchIndex):
        search_idx = index_or_chunks
        chunks = search_idx.chunks
    else:
        chunks = index_or_chunks
        search_idx = None

    if not chunks:
        return []

    # Filter chunks based on sender and date range
    filtered_chunks = []
    filtered_indices = []

    for idx, chunk in enumerate(chunks):
        ts = pd.Timestamp(chunk["timestamp"]).to_pydatetime()

        if sender:
            sender_lower = sender.strip().lower()
            chunk_participants = [p.lower() for p in chunk.get("participants", [])]
            chunk_sender = chunk.get("sender", "").lower()
            if sender_lower not in chunk_participants and sender_lower != chunk_sender:
                continue

        if start_date:
            start_dt = pd.to_datetime(start_date).to_pydatetime()
            chunk_end = pd.to_datetime(chunk.get("end_time", chunk["timestamp"])).to_pydatetime()
            if chunk_end < start_dt:
                continue

        if end_date:
            end_dt = pd.to_datetime(end_date).to_pydatetime()
            chunk_start = pd.to_datetime(chunk.get("start_time", chunk["timestamp"])).to_pydatetime()
            if chunk_start > end_dt:
                continue

        filtered_chunks.append(chunk)
        filtered_indices.append(idx)

    if not filtered_chunks:
        return []

    num_candidates = len(filtered_chunks)
    actual_k = min(top_k, num_candidates)

    # 1. BM25 Scoring
    tokenized_q = _tokenize(q)
    if search_idx and search_idx.bm25:
        all_bm25_scores = search_idx.bm25.get_scores(tokenized_q)
        bm25_scores = [all_bm25_scores[idx] for idx in filtered_indices]
    else:
        bm25 = BM25Okapi([_tokenize(chunk["text"]) for chunk in filtered_chunks])
        bm25_scores = bm25.get_scores(tokenized_q)

    bm25_ranked = np.argsort(bm25_scores)[::-1]
    bm25_ranks = {filtered_indices[pos]: rank for rank, pos in enumerate(bm25_ranked)}

    # 2. Semantic (ChromaDB or Model) Scoring
    model = _get_embedding_model()
    query_emb = model.encode(q, normalize_embeddings=True)

    semantic_scores = np.zeros(num_candidates)
    if search_idx and search_idx.collection:
        try:
            n_search = min(len(chunks), max(top_k * 20, 100))
            chroma_res = search_idx.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=n_search,
            )
            retrieved_ids = [int(cid) for cid in chroma_res["ids"][0]]
            distances = chroma_res["distances"][0] if "distances" in chroma_res else [0.0] * len(retrieved_ids)
            id_to_score = {cid: (1.0 - dist) for cid, dist in zip(retrieved_ids, distances)}
            for i, chunk in enumerate(filtered_chunks):
                semantic_scores[i] = id_to_score.get(chunk["chunk_id"], 0.0)
        except Exception:
            chunk_texts = [chunk["text"] for chunk in filtered_chunks]
            chunk_embs = model.encode(chunk_texts, normalize_embeddings=True)
            semantic_scores = np.dot(chunk_embs, query_emb)
    else:
        chunk_texts = [chunk["text"] for chunk in filtered_chunks]
        chunk_embs = model.encode(chunk_texts, normalize_embeddings=True)
        semantic_scores = np.dot(chunk_embs, query_emb)


    vector_ranked = np.argsort(semantic_scores)[::-1]
    vector_ranks = {filtered_indices[pos]: rank for rank, pos in enumerate(vector_ranked)}

    # 3. Hybrid Fusion via Reciprocal Rank Fusion (RRF)
    combined = []
    for i, chunk in enumerate(filtered_chunks):
        orig_idx = filtered_indices[i]
        r_bm25 = bm25_ranks[orig_idx]
        r_vec = vector_ranks[orig_idx]
        
        rrf_score = (sparse_weight / (rrf_k + r_bm25)) + (dense_weight / (rrf_k + r_vec))
        
        norm_sem = max(0.0, float(semantic_scores[i]))
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = float(bm25_scores[i] / max_bm25)
        blended_score = dense_weight * norm_sem + sparse_weight * norm_bm25

        out_chunk = dict(chunk)
        out_chunk["score"] = round(float(rrf_score * 100), 4)
        out_chunk["semantic_score"] = round(norm_sem, 4)
        out_chunk["bm25_score"] = round(norm_bm25, 4)
        out_chunk["blended_score"] = round(blended_score, 4)
        
        if sender:
            out_chunk["sender"] = sender
            
        combined.append(out_chunk)

    # Deduplicate by session_id to prevent overlapping chunks from dominating results
    combined = sorted(combined, key=lambda item: item["score"], reverse=True)
    deduped = []
    seen_sessions = set()
    for item in combined:
        sid = item.get("session_id", "")
        if sid and sid in seen_sessions:
            continue
        seen_sessions.add(sid)
        deduped.append(item)
        if len(deduped) >= actual_k:
            break
            
    combined = deduped

    results = []
    for item in combined:
        ts_val = item["timestamp"]
        ts_str = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
        
        context = item.get("context", [])
        if search_idx and search_idx.db and "start_message_id" in item:
            mid = item.get("start_message_id")
            db_ctx = search_idx.db.get_context_around(mid, window_before=2, window_after=4)
            if db_ctx:
                context = db_ctx

        results.append(
            {
                "chunk_id": item["chunk_id"],
                "session_id": item.get("session_id", ""),
                "sender": item["sender"],
                "participants": item.get("participants", [item["sender"]]),
                "timestamp": ts_str,
                "text": item["text"],
                "score": item["score"],
                "semantic_score": item.get("semantic_score", 0.0),
                "bm25_score": item.get("bm25_score", 0.0),
                "context": context,
            }
        )

    return results
