from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("/Users/mdsaibhossain/code/python/End-to-End RAG System/input.txt", "r") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]

vectors = model.encode(chunks)

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))
print(f"Indexed {len(chunks)} text chunks.")