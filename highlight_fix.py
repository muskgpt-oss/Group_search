import re

with open("app.py", "r") as f:
    content = f.read()

# Make sure 'import re' is at the top
if "import re" not in content:
    content = content.replace("import os", "import os\nimport re")

old_block = """                    # Display the conversational chunk
                    st.text_area(
                        f"Conversation Window (Chunk {res['chunk_id']}) - {res['timestamp'][:19]}",
                        value=res["text"],
                        height=140,
                        key=f"chunk_{res['chunk_id']}_{i}",
                    )"""

new_block = """                    # Display the conversational chunk
                    chunk_title = f"**Conversation Window (Chunk {res['chunk_id']}) - {res['timestamp'][:19]}**"
                    
                    # Highlight words from the search query
                    highlighted_text = res["text"]
                    if query_input:
                        words = [w for w in query_input.split() if len(w) > 2]
                        for w in words:
                            highlighted_text = re.sub(f"(?i)({re.escape(w)})", r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", highlighted_text)
                            
                    st.markdown(f"{chunk_title}<br><div style='background-color: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef; margin-top: 5px; margin-bottom: 15px; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.5;'>{highlighted_text}</div>", unsafe_allow_html=True)"""

content = content.replace(old_block, new_block)

with open("app.py", "w") as f:
    f.write(content)
