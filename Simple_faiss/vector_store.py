from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("/Simple_faiss/input.txt", "r") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]

vectors = model.encode(chunks)

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))
print(f"Indexed {len(chunks)} text chunks.")

def retrieve (quary, top_k=2):
    quary_vec = model.encode([quary])
    D,I = index.search(quary_vec,top_k)
    results = [chunks[i] for i in I[0]]
    return results

question = "what is Guido ?"
context = retrieve(question)
for c in context:
    print("_",c)