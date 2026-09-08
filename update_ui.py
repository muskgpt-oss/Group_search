import re

with open("app.py", "r") as f:
    content = f.read()

# 1. Update Headers
new_header = """
st.markdown('<div class="main-header">💬 Group Chat Search Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Instantly search through thousands of group chat messages using AI-powered semantic search.</div>',
    unsafe_allow_html=True,
)

tab_search, tab_browse, tab_benchmarks, tab_data = st.tabs(["🔍 Search", "💬 Browse Chats", "⚙️ Evaluation", "🗄️ Database Stats"])
"""
content = re.sub(r"st.markdown\('<div class=\"main-header\">.*?🗄️ Database & Stats\"\]\)", new_header, content, flags=re.DOTALL)

# 2. Update Sidebar (hide tuning)
new_sidebar = """
    with st.sidebar:
        st.header("Search Controls")
        auto_detect = st.checkbox("Auto-detect Context", value=True, help="Extracts participant names and dates automatically from your search prompt.")
        
        with st.expander("🎯 Manual Filters", expanded=False):
            participants = []
            if index and index.db:
                participants = sorted(index.db.get_all_messages(as_df=True)["sender"].unique().tolist())
            selected_sender = st.selectbox("Sender / Participant", ["All"] + participants, index=0)
            
            col_sd, col_ed = st.columns(2)
            with col_sd:
                filter_start = st.date_input("Start Date", value=None)
            with col_ed:
                filter_end = st.date_input("End Date", value=None)

        with st.expander("⚙️ Advanced Tuning", expanded=False):
            top_k = st.slider("Top Results", min_value=1, max_value=20, value=5)
            dense_weight = st.slider("AI Semantic Weight", 0.0, 1.0, 0.7, 0.05)
            sparse_weight = round(1.0 - dense_weight, 2)
            st.caption(f"Keyword Weight: **{sparse_weight}**")
        
        st.divider()
        if st.button("🔄 Force Rebuild Index", use_container_width=True):
            st.cache_resource.clear()
            get_search_index(force_rebuild=True)
            st.success("Rebuilt index!")
"""
content = re.sub(r"    with st.sidebar:.*?st.success\(\"Rebuilt ChromaDB index & SQLite database!\"\)", new_sidebar, content, flags=re.DOTALL)

# 3. Update Search Result Cards
new_results = """
        if not results:
            st.warning("No messages matched your query.")
        else:
            for i, res in enumerate(results, 1):
                with st.container(border=True):
                    col_title, col_score = st.columns([3, 1])
                    with col_title:
                        st.markdown(f"**Session:** `{res['session_id']}` &nbsp;|&nbsp; 👥 {', '.join(res['participants'])}")
                    with col_score:
                        st.markdown(f"<div style='text-align: right; color: #10b981; font-weight: bold;'>Match: {round((res['score']/2)*100)}%</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Split the chunk into individual messages and render as chat bubbles
                    for msg in res.get("context", []):
                        # Highlight the exact matched text
                        is_match = msg.get("text") in res["text"]
                        
                        with st.chat_message(name=msg['sender']):
                            st.markdown(f"**{msg['sender']}** <span style='color: gray; font-size: 0.8em;'>{msg['timestamp'][11:16]}</span>", unsafe_allow_html=True)
                            if is_match:
                                st.markdown(f"<div style='background-color: #fef9c3; display: inline-block; padding: 2px 5px; border-radius: 4px;'>{msg['text']}</div>", unsafe_allow_html=True)
                            else:
                                st.write(msg['text'])
"""
content = re.sub(r"        if not results:.*?st.divider\(\)", new_results, content, flags=re.DOTALL)

# 4. Update Browse Tab
new_browse = """
with tab_browse:
    st.header("Browse Chats")
    st.write("Read full conversation threads straight from the database.")
    if index.db:
        all_df = index.db.get_all_messages(as_df=True)
        if not all_df.empty:
            sessions = all_df["session_id"].unique().tolist()
            selected_session = st.selectbox("Select a Chat Session", sessions)
            
            if selected_session:
                session_df = all_df[all_df["session_id"] == selected_session].sort_values("timestamp")
                st.subheader(f"Chat History: {selected_session}")
                st.caption(f"Date: {session_df['timestamp'].iloc[0].strftime('%B %d, %Y')}")
                st.divider()
                
                for _, row in session_df.iterrows():
                    with st.chat_message(name=row['sender']):
                        st.markdown(f"**{row['sender']}** <span style='color: gray; font-size: 0.8em;'>{row['timestamp'].strftime('%I:%M %p')}</span>", unsafe_allow_html=True)
                        st.write(row['text'])
        else:
            st.info("No messages found.")
    else:
        st.error("Database not connected.")
"""
content = re.sub(r"with tab_browse:.*?st.info\(\"Database not connected.\"\)", new_browse, content, flags=re.DOTALL)

with open("app.py", "w") as f:
    f.write(content)
