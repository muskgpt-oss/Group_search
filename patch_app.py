import re

with open("app.py", "r") as f:
    content = f.read()

# 1. Add Legend above Results
legend_html = """
        st.divider()
        st.markdown(f"### 📑 Results ({len(results)} found)")
        
        # 4. Small one-time legend
        st.markdown(
            \"\"\"
            <div style='margin-bottom: 1rem;'>
                <span style='font-size: 0.85rem; color: #475569; font-weight: 600; margin-right: 15px;'>Highlight Legend:</span>
                <span style='background-color: #fef08a; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; color: #854d0e; margin-right: 10px;'>Exact keyword match</span>
                <span style='background-color: #e0e7ff; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; color: #3730a3;'>Matched by meaning</span>
            </div>
            \"\"\", unsafe_allow_html=True
        )

        # 7. Proper empty state"""

content = content.replace("""        st.divider()
        st.markdown(f"### 📑 Results ({len(results)} found)")

        # 7. Proper empty state""", legend_html)


# 2. Modify Result Card rendering
old_render = """                    raw_text = res["text"]
                    highlighted_text = raw_text
                    
                    # Apply regex highlights
                    stop_words = {"the", "and", "for", "with", "about", "what", "where", "when", "why", "who", "how", "this", "that", "there", "their", "are", "was", "were"}
                    words = [w.lower() for w in query_input.split() if len(w) > 2 and w.lower() not in stop_words]
                    for w in words:
                        highlighted_text = re.sub(
                            f"(?i)(\\\\\\b\\\\\\w*{re.escape(w)}\\\\\\w*\\\\\\b)", 
                            r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", 
                            highlighted_text
                        )
                    
                    # 4. Detect pure semantic match (no literal word overlap highlighted)
                    is_semantic_only = (highlighted_text == raw_text)
                    if is_semantic_only:
                        st.markdown("<div class='match-meaning-badge'>✨ Matched by Meaning</div>", unsafe_allow_html=True)
                    
                    chunk_title = f"**Conversation Window (Chunk {res['chunk_id']}) - {res['timestamp'][:19]}**"
                    st.markdown(f"{chunk_title}<br><div style='background-color: #f8fafc; padding: 16px; border-radius: 8px; font-size: 0.95rem; line-height: 1.6;'>{highlighted_text}</div>", unsafe_allow_html=True)
                    
                    st.write("")"""

new_render = """                    raw_text = res["text"]
                    
                    stop_words = {"the", "and", "for", "with", "about", "what", "where", "when", "why", "who", "how", "this", "that", "there", "their", "are", "was", "were"}
                    query_words = [w.lower() for w in query_input.split() if len(w) > 2 and w.lower() not in stop_words]
                    
                    lines = raw_text.strip().split('\\n')
                    
                    # 2. Show only ~2-3 messages before and after (the chunk is already ~7, so we show it line-by-line tightly)
                    # We will format each line
                    formatted_lines = []
                    has_any_keyword_match = False
                    
                    colors = ["#2563eb", "#db2777", "#16a34a", "#ea580c", "#8b5cf6", "#0d9488", "#b91c1c"]
                    
                    for line in lines:
                        if ":" not in line:
                            formatted_lines.append(f"<div style='color: #475569; padding: 2px 0;'>{line}</div>")
                            continue
                            
                        sender_part, msg_part = line.split(":", 1)
                        sender_part = sender_part.strip()
                        msg_part = msg_part.strip()
                        
                        # 1. Apply keyword highlighting
                        highlighted_msg = msg_part
                        for w in query_words:
                            if re.search(f"(?i)(\\b\\w*{re.escape(w)}\\w*\\b)", highlighted_msg):
                                has_any_keyword_match = True
                            highlighted_msg = re.sub(
                                f"(?i)(\\b\\w*{re.escape(w)}\\w*\\b)", 
                                r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", 
                                highlighted_msg
                            )
                        
                        # Consistent sender color
                        s_color = colors[hash(sender_part) % len(colors)]
                        line_html = f"<div style='padding: 3px 0;'><strong style='color: {s_color};'>{sender_part}:</strong> <span style='color: #1e293b;'>{highlighted_msg}</span></div>"
                        formatted_lines.append(line_html)
                        
                    is_semantic_only = not has_any_keyword_match
                    
                    # 3. Add second highlight style for semantic-only matches
                    if is_semantic_only and formatted_lines:
                        # Vector match pointed to this chunk. We highlight the center message to visually anchor the semantic meaning.
                        center_idx = len(formatted_lines) // 2
                        center_line = formatted_lines[center_idx]
                        formatted_lines[center_idx] = f"<div style='background-color: #e0e7ff; padding: 4px 8px; border-radius: 6px; border-left: 3px solid #4f46e5; margin: 4px 0;'>{center_line}</div>"

                    preview_html = "".join(formatted_lines)
                    
                    chunk_title = f"**Conversation Window (Chunk {res['chunk_id']}) - {res['timestamp'][:19]}**"
                    st.markdown(f"{chunk_title}<br><div style='background-color: #f8fafc; padding: 12px 16px; border-radius: 8px; font-size: 0.95rem; line-height: 1.5; border: 1px solid #e2e8f0; margin-bottom: 6px;'>{preview_html}</div>", unsafe_allow_html=True)
                    
                    # 5. Caption for semantic-only
                    if is_semantic_only:
                        st.markdown("<div style='font-size: 0.85rem; color: #64748b; font-style: italic; margin-bottom: 12px; padding-left: 4px;'>No shared words with your query — matched by meaning.</div>", unsafe_allow_html=True)
                    else:
                        st.write("")"""

content = content.replace(old_render, new_render)

with open("app.py", "w") as f:
    f.write(content)
