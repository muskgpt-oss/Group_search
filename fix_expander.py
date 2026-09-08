import re

with open("app.py", "r") as f:
    content = f.read()

old_block = """                    st.divider()
                    
                    # Split the chunk into individual messages and render as chat bubbles
                    for msg in res.get("context", []):
                        # Highlight the exact matched text
                        is_match = msg.get("text") in res["text"]
                        
                        with st.chat_message(name=msg['sender']):
                            st.markdown(f"**{msg['sender']}** <span style='color: gray; font-size: 0.8em;'>{msg['timestamp'][11:16]}</span>", unsafe_allow_html=True)
                            if is_match:
                                st.markdown(f"<div style='background-color: #fef9c3; display: inline-block; padding: 2px 5px; border-radius: 4px;'>{msg['text']}</div>", unsafe_allow_html=True)
                            else:
                                st.write(msg['text'])"""

new_block = """                    st.divider()
                    st.write(f"**Matched text:** *{res['text']}*")
                    
                    # Split the chunk into individual messages and render as chat bubbles
                    with st.expander("💬 View Extended Thread Context (from SQLite DB)"):
                        for msg in res.get("context", []):
                            # Highlight the exact matched text
                            is_match = msg.get("text") in res["text"]
                            
                            with st.chat_message(name=msg['sender']):
                                st.markdown(f"**{msg['sender']}** <span style='color: gray; font-size: 0.8em;'>{msg['timestamp'][11:16]}</span>", unsafe_allow_html=True)
                                if is_match:
                                    st.markdown(f"<div style='background-color: #fef9c3; display: inline-block; padding: 2px 5px; border-radius: 4px;'>{msg['text']}</div>", unsafe_allow_html=True)
                                else:
                                    st.write(msg['text'])"""

content = content.replace(old_block, new_block)

with open("app.py", "w") as f:
    f.write(content)
