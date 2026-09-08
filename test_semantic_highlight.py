import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "food and snacks"
text = "Canteen momos are fire today. I am so hungry."

start = time.time()
q_emb = model.encode([query])

# Extract words
words = list(set(re.findall(r'\b\w{3,}\b', text)))
w_embs = model.encode(words)

sims = cosine_similarity(q_emb, w_embs)[0]

highlight_words = []
for w, sim in zip(words, sims):
    print(f"Word: {w}, Sim: {sim}")
    if sim > 0.35: # Threshold
        highlight_words.append(w)

print("Words to highlight:", highlight_words)
print(f"Time: {time.time() - start:.3f}s")
