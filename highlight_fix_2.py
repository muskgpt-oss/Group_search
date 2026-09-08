import re

with open("app.py", "r") as f:
    content = f.read()

old_block = """                    if query_input:
                        words = [w for w in query_input.split() if len(w) > 2]
                        for w in words:
                            highlighted_text = re.sub(f"(?i)({re.escape(w)})", r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", highlighted_text)"""

new_block = """                    if query_input:
                        # Extract core keywords, ignoring common stop words
                        stop_words = {"the", "and", "for", "with", "about", "what", "where", "when", "why", "who", "how", "this", "that", "there", "their", "are", "was", "were"}
                        words = [w.lower() for w in query_input.split() if len(w) > 2 and w.lower() not in stop_words]
                        
                        # Use partial word matching to highlight related forms (e.g. exam -> exams)
                        for w in words:
                            highlighted_text = re.sub(f"(?i)(\\b\\w*{re.escape(w)}\\w*\\b)", r"<mark style='background-color: #fef08a; padding: 0 4px; border-radius: 4px; color: #854d0e;'>\g<1></mark>", highlighted_text)"""

content = content.replace(old_block, new_block)

with open("app.py", "w") as f:
    f.write(content)
