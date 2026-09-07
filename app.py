from __future__ import annotations

import os
import re
from datetime import datetime
import pandas as pd
import streamlit as st

from it_geek_search.loader import load_chat_from_txt
from it_geek_search.generator import generate_benchmark_queries
from it_geek_search.retrieval import build_search_index, search_messages, SearchIndex
from it_geek_search.query_parser import parse_query
from it_geek_search.evaluator import evaluate_retrieval

# Configure Streamlit page
st.set_page_config(
    page_title="Hinglish Chat Semantic Search",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .chat-bubble {
        padding: 10px 14px;
        border-radius: 12px;
        margin-bottom: 8px;
        line-height: 1.4;
        font-size: 0.95rem;
    }
    .chat-user-1 { background-color: #f0f4f8; border-left: 4px solid #3b82f6; }
    .chat-user-2 { background-color: #fdf2f8; border-left: 4px solid #ec4899; }
    .chat-user-3 { background-color: #ecfdf5; border-left: 4px solid #10b981; }
    .chat-user-4 { background-color: #fffbeb; border-left: 4px solid #f59e0b; }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 6px;
    }
    .badge-semantic { background: #dbeafe; color: #1e40af; }
    .badge-keyword { background: #fef3c7; color: #92400e; }
    .badge-score { background: #dcfce7; color: #166534; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = "./data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
DB_PATH = os.path.join(DATA_DIR, "chat_history.db")


@st.cache_resource(show_spinner="Initializing Database & ChromaDB Vector Store...")
def get_search_index(force_rebuild: bool = False) -> SearchIndex:
    os.makedirs(DATA_DIR, exist_ok=True)
    
    txt_path = os.path.join(DATA_DIR, "chat_history.txt")
    if os.path.exists(txt_path):
        df = load_chat_from_txt(txt_path)
    else:
        st.error("chat_history.txt not found in data folder!")
        return None
        
    index = build_search_index(
        messages=df,
        chroma_path=CHROMA_DIR,
        db_path=DB_PATH,
        force_rebuild=force_rebuild,
    )
    return index


# Load Index
index = get_search_index()

st.markdown('<div class="main-header">💬 Context-Aware Hinglish Chat Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid Semantic & Lexical (ChromaDB + BM25) Retrieval over 4,000+ Hinglish Group Chat Messages</div>',
    unsafe_allow_html=True,
)

tab_search, tab_benchmarks, tab_browse, tab_data = st.tabs(["🔍 Search & Chat", "📊 Benchmark & Evaluation", "💬 Browse Conversations", "🗄️ Database & Stats"])

with tab_search:
    # Sidebar Filters
    with st.sidebar:
        st.header("Search Controls")
        auto_detect = st.checkbox("Auto-detect Sender & Date from Query", value=True, help="Extracts participant names and dates automatically from the prompt.")
        
        st.subheader("Manual Filters")
        participants = []
        if index and index.db:
            participants = sorted(index.db.get_all_messages(as_df=True)["sender"].unique().tolist())
        selected_sender = st.selectbox("Sender / Participant", ["All"] + participants, index=0)
        
        col_sd, col_ed = st.columns(2)
        with col_sd:
            filter_start = st.date_input("Start Date", value=None)
        with col_ed:
            filter_end = st.date_input("End Date", value=None)

        st.subheader("Retrieval Tuning")
        top_k = st.slider("Top Results", min_value=1, max_value=20, value=5)
        dense_weight = st.slider("Vector (Semantic) Weight", 0.0, 1.0, 0.7, 0.05)
        sparse_weight = round(1.0 - dense_weight, 2)
        st.caption(f"BM25 (Keyword) Weight: **{sparse_weight}**")
        
        if st.button("🔄 Force Rebuild Index"):
            st.cache_resource.clear()
            get_search_index(force_rebuild=True)
            st.success("Rebuilt ChromaDB index & SQLite database!")

    # Search Bar & Preset Examples
    sample_queries = [
        "What did Rohan say about the server costs?",
        "when did we decide on Manali?",
        "Where did the group settle on having the coding marathon meetup?",
        "What database indexing optimizations were completed in April?",
        "502 bad gateway auth-worker memory leak container OOM",
    ]
    
    col_q, col_sample = st.columns([3, 1])
    with col_sample:
        selected_sample = st.selectbox("Sample Queries", ["Custom"] + sample_queries)
        
    with col_q:
        default_val = selected_sample if selected_sample != "Custom" else "weekend hackathon final karlo"
        query_input = st.text_input("Enter search query (Hinglish or English):", value=default_val)

    # Determine query parameters
    effective_sender = None if selected_sender == "All" else selected_sender
    effective_start = filter_start.isoformat() if filter_start else None
    effective_end = filter_end.isoformat() if filter_end else None

    # Show parser insights if auto_detect is enabled
    if auto_detect and query_input:
        parsed_q = parse_query(query_input)
        if parsed_q.sender or parsed_q.start_date or parsed_q.end_date:
            st.info(
                f"**Auto-Detected Intent**: `{parsed_q.inferred_intent}` | "
                f"**Sender**: `{parsed_q.sender or 'Any'}` | "
                f"**Date Range**: `{parsed_q.start_date or 'Any'} → {parsed_q.end_date or 'Any'}` | "
                f"**Search Keywords**: *\"{parsed_q.clean_query}\"*"
            )

    if query_input:
        with st.spinner("Searching..."):
            results = search_messages(
                index,
                query=query_input,
                sender=effective_sender,
                start_date=effective_start,
                end_date=effective_end,
                top_k=top_k,
                auto_parse=auto_detect,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )

        st.markdown(f"#### Results ({len(results)} matches)")

        if not results:
            st.warning("No messages matched the query and filters.")
        else:
            for i, res in enumerate(results, 1):
                score_badge = f"<span class='badge badge-score'>RRF Score: {res['score']}</span>"
                sem_badge = f"<span class='badge badge-semantic'>Semantic: {res['semantic_score']}</span>"
                bm25_badge = f"<span class='badge badge-keyword'>BM25: {res['bm25_score']}</span>"
                
                with st.container():
                    st.markdown(
                        f"**#{i} | Session: `{res['session_id']}` | Participants: {', '.join(res['participants'])}** "
                        f"<br/>{score_badge} {sem_badge} {bm25_badge}",
                        unsafe_allow_html=True,
                    )
                    
                    # Display the conversational chunk
                    chunk_title = f"**Conversation Window (Chunk {res['chunk_id']}) - {res['timestamp'][:19]}**"
                    
                    # Highlight words from the search query
                    highlighted_text = res["text"]
                    if query_input:
                        # Extract core keywords, ignoring common stop words
                        stop_words = {"the", "and", "for", "with", "about", "what", "where", "when", "why", "who", "how", "this", "that", "there", "their", "are", "was", "were"}
                        words = [w.lower() for w in query_input.split() if len(w) > 2 and w.lower() not in stop_words]
                        
                        # Use partial word matching to highlight related forms (e.g. exam -> exams)
                        for w in words:
                            highlighted_text = re.sub(f"(?i)(\b\w*{re.escape(w)}\w*\b)", r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", highlighted_text)
                            
                    st.markdown(f"{chunk_title}<br><div style='background-color: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef; margin-top: 5px; margin-bottom: 15px; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.5;'>{highlighted_text}</div>", unsafe_allow_html=True)

                    # Show database-backed context
                    with st.expander("💬 View Extended Thread Context (from SQLite DB)"):
                        for msg in res.get("context", []):
                            is_target = msg.get("text") in res["text"]
                            color_style = "border-left: 4px solid #3b82f6;" if is_target else "border-left: 2px solid #ccc; opacity: 0.85;"
                            st.markdown(
                                f"""<div style="background-color: #f8fafc; padding: 6px 12px; margin: 4px 0; border-radius: 6px; {color_style}">
                                    <strong style="color: #1e293b;">{msg['sender']}</strong> 
                                    <span style="font-size: 0.8em; color: #64748b;">({msg['timestamp']})</span>: 
                                    <span style="color: #334155;">{msg['text']}</span>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                    st.divider()

with tab_benchmarks:
    st.header("Benchmark Test Suite (40 Queries)")
    st.write(
        "Evaluates the hybrid search system across 40 predefined benchmark queries "
        "encompassing semantic search (including zero-lexical-overlap), author-attributed queries, "
        "temporal queries, and hybrid queries."
    )

    if st.button("🚀 Run Full Benchmark Suite"):
        with st.spinner("Evaluating 40 benchmark queries..."):
            eval_output = evaluate_retrieval(index, top_k=5, use_auto_parse=True)
            summary = eval_output["summary"]
            details = pd.DataFrame(eval_output["details"])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean Reciprocal Rank (MRR)", f"{summary['mrr']:.3f}")
        col2.metric("Hit@1 Rate", f"{summary['hit@1'] * 100:.1f}%")
        col3.metric("Hit@3 Rate", f"{summary['hit@3'] * 100:.1f}%")
        col4.metric("Hit@5 Rate", f"{summary['hit@5'] * 100:.1f}%")
        col5.metric("Zero-Overlap MRR", f"{summary['zero_overlap_mrr']:.3f}")

        st.subheader("Detailed Query Results")
        st.dataframe(
            details[["id", "type", "zero_overlap", "prompt", "hit_rank", "reciprocal_rank", "latency_ms"]],
            width='stretch',
        )


with tab_browse:
    st.header("Browse Generated Conversations")
    st.write("View the actual full conversation threads generated in the backend.")
    if index.db:
        all_df = index.db.get_all_messages(as_df=True)
        if not all_df.empty:
            topics = ["All"] + sorted(all_df["topic"].unique().tolist())
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                filter_topic = st.selectbox("Filter by Topic", topics)
            
            # Filter logic
            filtered_df = all_df
            if filter_topic != "All":
                filtered_df = filtered_df[filtered_df["topic"] == filter_topic]
            
            sessions = filtered_df["session_id"].unique().tolist()
            with col_b2:
                selected_session = st.selectbox("Select Session to View", sessions)
            
            if selected_session:
                session_df = all_df[all_df["session_id"] == selected_session].sort_values("timestamp")
                st.subheader(f"Session: {selected_session} (Topic: {session_df['topic'].iloc[0]})")
                st.write(f"**Date:** {session_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')}")
                
                st.markdown("---")
                for _, row in session_df.iterrows():
                    st.markdown(
                        f"""<div style="background-color: #f8fafc; padding: 10px 14px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <strong style="color: #1e293b;">{row['sender']}</strong> 
                            <span style="font-size: 0.8em; color: #64748b;">({row['timestamp'].strftime('%H:%M:%S')})</span>: 
                            <span style="color: #334155; font-size: 1.05em;">{row['text']}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No messages in database.")
    else:
        st.info("Database not connected.")

with tab_data:
    st.header("Dataset & SQLite Storage Statistics")
    
    if index.db:
        total_msgs = index.db.count()
        st.write(f"**Total Stored Messages:** {total_msgs}")
        st.write(f"**Total Indexed Chunks:** {len(index.chunks)}")
        
        all_df = index.db.get_all_messages(as_df=True)
        if not all_df.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Messages per Participant")
                st.bar_chart(all_df["sender"].value_counts())
            with col_b:
                st.subheader("Messages per Topic")
                st.bar_chart(all_df["topic"].value_counts())
            
            st.subheader("Raw Message Sample")
            st.dataframe(all_df.head(50), width='stretch')
    else:
        st.info("Database not connected.")

