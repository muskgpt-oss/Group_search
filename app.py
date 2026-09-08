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

# 1. Global Styling & Layout Config
st.set_page_config(
    page_title="AI Context Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '''
    <style>
    /* Hide header and footer for demo cleanliness */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean Typography & Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    h1, h2, h3 { color: #1e293b; font-weight: 700; letter-spacing: -0.5px; }
    .stat-strip {
        display: flex; gap: 20px; align-items: center; 
        background: #ffffff; padding: 12px 20px; 
        border-radius: 8px; border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .stat-item {
        font-size: 0.9rem; color: #475569;
    }
    .stat-value {
        font-size: 1.1rem; font-weight: 600; color: #4f46e5;
    }
    .match-meaning-badge {
        background-color: #f3e8ff; color: #7e22ce;
        padding: 4px 10px; border-radius: 99px;
        font-size: 0.75rem; font-weight: 600; display: inline-block;
        margin-bottom: 8px;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Backend Initialization (Untouched Logic)
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
    return build_search_index(
        messages=df, chroma_path=CHROMA_DIR, db_path=DB_PATH, force_rebuild=force_rebuild
    )

index = get_search_index()

# 2. Header & Stats Strip
st.title("🔍 Context-Aware Semantic Search")

if index and index.db:
    all_df = index.db.get_all_messages(as_df=True)
    if not all_df.empty:
        total_msgs = len(all_df)
        total_participants = all_df['sender'].nunique()
        min_date = all_df['timestamp'].min().strftime('%b %d, %Y')
        max_date = all_df['timestamp'].max().strftime('%b %d, %Y')
        
        st.markdown(
            f'''
            <div class="stat-strip">
                <div class="stat-item">Messages Indexed: <span class="stat-value">{total_msgs:,}</span></div>
                <div class="stat-item">Participants: <span class="stat-value">{total_participants}</span></div>
                <div class="stat-item">Timeframe: <span class="stat-value">{min_date} — {max_date}</span></div>
            </div>
            ''',
            unsafe_allow_html=True
        )

# Standardized Tabs
tab_search, tab_benchmarks, tab_browse, tab_data = st.tabs([
    "🔍 Search & Chat", "📊 Benchmark & Evaluation", "💬 Browse Conversations", "🗄️ Database & Stats"
])

# 5. Sidebar Restructuring
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Settings")
    
    with st.container(border=True):
        st.markdown("#### 🎯 Search Controls")
        auto_detect = st.checkbox("✨ Auto-detect Sender/Date", value=True, help="Extracts filters automatically from prompt.")
        
        st.divider()
        st.markdown("#### 👤 Manual Filters")
        participants = []
        if index and index.db:
            participants = sorted(index.db.get_all_messages(as_df=True)["sender"].unique().tolist())
        selected_sender = st.selectbox("Participant", ["All"] + participants, index=0)
        col_sd, col_ed = st.columns(2)
        with col_sd:
            filter_start = st.date_input("From", value=None)
        with col_ed:
            filter_end = st.date_input("To", value=None)

    with st.container(border=True):
        st.markdown("#### 🎛️ Retrieval Tuning")
        top_k = st.slider("Top Results Count", min_value=1, max_value=20, value=5)
        dense_weight = st.slider("Semantic Weight (Vectors)", 0.0, 1.0, 0.7, 0.05)
        sparse_weight = round(1.0 - dense_weight, 2)
        st.caption(f"Keyword Weight (BM25): **{sparse_weight}**")
        
    with st.container(border=True):
        st.markdown("#### 🔄 Maintenance")
        if st.button("Force Rebuild Index", use_container_width=True):
            st.cache_resource.clear()
            get_search_index(force_rebuild=True)
            st.success("Rebuilt successfully!")

# Tab 1: Search
with tab_search:
    # 3. Search Box Container
    with st.container(border=True):
        st.markdown("### Search the Corpus")
        
        sample_queries = [
            "What did Rohan say about the server costs?",
            "when did we decide on Manali?",
            "Where did the group settle on having the coding marathon meetup?",
            "What database indexing optimizations were completed in April?",
            "502 bad gateway auth-worker memory leak container OOM"
        ]
        
        # Grid layout for inputs inside container
        search_col, preset_col = st.columns([3, 1])
        with preset_col:
            selected_sample = st.selectbox("Or try an example...", ["Custom"] + sample_queries, label_visibility="collapsed")
        
        with search_col:
            default_val = selected_sample if selected_sample != "Custom" else ""
            query_input = st.text_input(
                "Enter search query", 
                value=default_val, 
                placeholder="Type your search here e.g., 'weekend hackathon plans'...",
                label_visibility="collapsed"
            )

    # Retrieval Execution
    effective_sender = None if selected_sender == "All" else selected_sender
    effective_start = filter_start.isoformat() if filter_start else None
    effective_end = filter_end.isoformat() if filter_end else None

    if auto_detect and query_input:
        parsed_q = parse_query(query_input)
        if parsed_q.sender or parsed_q.start_date or parsed_q.end_date:
            st.info(
                f"✨ **Smart Filter Active** • Sender: `{parsed_q.sender or 'Any'}` | "
                f"Dates: `{parsed_q.start_date or 'Any'} → {parsed_q.end_date or 'Any'}`",
                icon="🪄"
            )

    if query_input:
        # 7. Add a proper spinner
        with st.spinner("Analyzing semantic and lexical matches..."):
            results = search_messages(
                index, query=query_input, sender=effective_sender,
                start_date=effective_start, end_date=effective_end,
                top_k=top_k, auto_parse=auto_detect,
                dense_weight=dense_weight, sparse_weight=sparse_weight,
            )

        st.divider()
        st.markdown(f"### 📑 Results ({len(results)} found)")

        # 7. Proper empty state
        if not results:
            st.warning("No messages matched the query and filters. Try adjusting your parameters.", icon="⚠️")
        else:
            for i, res in enumerate(results, 1):
                # 4. Result cards with horizontal progress bars & semantic match detection
                with st.container(border=True):
                    header_col, score_col = st.columns([2, 1])
                    with header_col:
                        st.markdown(f"**Chat Session:** `{res['session_id']}`")
                        st.caption(f"👥 Participants: {', '.join(res['participants'])}")
                    with score_col:
                        # Normalize scores purely for visual bar rendering (backend logic untouched)
                        # RRF is usually very small. BM25/Semantic vary. We cap for visual display.
                        sem_val = min(1.0, res['semantic_score'])
                        bm25_val = min(1.0, res['bm25_score'] / 10.0 if res['bm25_score'] > 0 else 0)
                        
                        st.markdown(f"<span style='font-size: 0.8rem; font-weight: 600;'>Semantic Meaning (Score: {res['semantic_score']:.2f})</span>", unsafe_allow_html=True)
                        st.progress(sem_val)
                        st.markdown(f"<span style='font-size: 0.8rem; font-weight: 600;'>Exact Keyword (Score: {res['bm25_score']:.2f})</span>", unsafe_allow_html=True)
                        st.progress(bm25_val)

                    st.markdown("---")
                    
                    raw_text = res["text"]
                    highlighted_text = raw_text
                    
                    # Apply regex highlights
                    stop_words = {"the", "and", "for", "with", "about", "what", "where", "when", "why", "who", "how", "this", "that", "there", "their", "are", "was", "were"}
                    words = [w.lower() for w in query_input.split() if len(w) > 2 and w.lower() not in stop_words]
                    for w in words:
                        highlighted_text = re.sub(
                            f"(?i)(\\b\\w*{re.escape(w)}\\w*\\b)", 
                            r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", 
                            highlighted_text
                        )
                    
                    # 4. Detect pure semantic match (no literal word overlap highlighted)
                    is_semantic_only = (highlighted_text == raw_text)
                    if is_semantic_only:
                        st.markdown("<div class='match-meaning-badge'>✨ Matched by Meaning</div>", unsafe_allow_html=True)
                    
                    chunk_title = f"**Conversation Window (Chunk {res['chunk_id']}) - {res['timestamp'][:19]}**"
                    st.markdown(f"{chunk_title}<br><div style='background-color: #f8fafc; padding: 16px; border-radius: 8px; font-size: 0.95rem; line-height: 1.6;'>{highlighted_text}</div>", unsafe_allow_html=True)
                    
                    st.write("")
                    with st.expander("💬 View Extended Thread Context (SQLite)"):
                        for msg in res.get("context", []):
                            is_target = msg.get("text") in raw_text
                            color_style = "border-left: 4px solid #4f46e5; background: #f1f5f9;" if is_target else "border-left: 2px solid #cbd5e1; opacity: 0.85;"
                            st.markdown(
                                f'''<div style="padding: 8px 14px; margin: 6px 0; border-radius: 6px; {color_style}">
                                    <strong style="color: #0f172a;">{msg['sender']}</strong> 
                                    <span style="font-size: 0.8em; color: #64748b;">({msg['timestamp']})</span>: 
                                    <span style="color: #334155;">{msg['text']}</span>
                                </div>''',
                                unsafe_allow_html=True
                            )

# Tab 2: Benchmarks (Untouched logic, spaced out)
with tab_benchmarks:
    st.header("Benchmark Test Suite (40 Queries)")
    st.write("Evaluates the hybrid search system across predefined queries.")
    st.divider()

    if st.button("🚀 Run Full Benchmark Suite", type="primary"):
        with st.spinner("Running complex evaluation metrics..."):
            eval_output = evaluate_retrieval(index, top_k=5, use_auto_parse=True)
            summary = eval_output["summary"]
            details = pd.DataFrame(eval_output["details"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean Reciprocal Rank", f"{summary['mrr']:.3f}")
        c2.metric("Hit@1 Rate", f"{summary['hit@1'] * 100:.1f}%")
        c3.metric("Hit@3 Rate", f"{summary['hit@3'] * 100:.1f}%")
        c4.metric("Hit@5 Rate", f"{summary['hit@5'] * 100:.1f}%")
        c5.metric("Zero-Overlap MRR", f"{summary['zero_overlap_mrr']:.3f}")
        
        st.divider()
        st.subheader("Detailed Query Results")
        st.dataframe(details[["id", "type", "zero_overlap", "prompt", "hit_rank", "latency_ms"]], use_container_width=True)

# Tab 3: Browse
with tab_browse:
    st.header("Browse Conversation Threads")
    st.divider()
    
    if index.db:
        all_df = index.db.get_all_messages(as_df=True)
        if not all_df.empty:
            topics = ["All"] + sorted(all_df["topic"].unique().tolist())
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                filter_topic = st.selectbox("Filter by Topic", topics)
            
            filtered_df = all_df if filter_topic == "All" else all_df[all_df["topic"] == filter_topic]
            sessions = filtered_df["session_id"].unique().tolist()
            
            with col_b2:
                selected_session = st.selectbox("Select Session to View", sessions)
            
            if selected_session:
                st.divider()
                session_df = all_df[all_df["session_id"] == selected_session].sort_values("timestamp")
                st.subheader(f"Session: {selected_session}")
                st.caption(f"Date: {session_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} | Topic: {session_df['topic'].iloc[0]}")
                
                for _, row in session_df.iterrows():
                    st.markdown(
                        f'''<div style="background-color: #f8fafc; padding: 10px 14px; margin: 6px 0; border-radius: 6px; border-left: 3px solid #64748b;">
                            <strong style="color: #1e293b;">{row['sender']}</strong> 
                            <span style="font-size: 0.8em; color: #94a3b8;">({row['timestamp'].strftime('%H:%M:%S')})</span>: 
                            <span style="color: #334155; font-size: 1rem;">{row['text']}</span>
                        </div>''',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No messages in database.")

# Tab 4: Data & Stats
with tab_data:
    st.header("Storage & Analytics")
    st.divider()
    
    if index.db:
        all_df = index.db.get_all_messages(as_df=True)
        if not all_df.empty:
            # 6. Enhance the Stats tab with a timeline chart
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Stored Messages", f"{len(all_df):,}")
            with c2:
                st.metric("Total Indexed Vectors (Chunks)", f"{len(index.chunks):,}")
            
            st.divider()
            st.subheader("📊 Message Activity Over Time")
            # Convert timestamp to date for aggregation
            timeline_df = all_df.copy()
            timeline_df['Date'] = pd.to_datetime(timeline_df['timestamp']).dt.date
            daily_counts = timeline_df.groupby('Date').size()
            st.line_chart(daily_counts)
            
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("🗣️ Messages per Participant")
                st.bar_chart(all_df["sender"].value_counts())
            with c4:
                st.subheader("📌 Messages per Topic")
                st.bar_chart(all_df["topic"].value_counts())
                
            st.divider()
            st.subheader("Raw Message Sample")
            st.dataframe(all_df.head(50), use_container_width=True)
    else:
        st.info("Database not connected.")
