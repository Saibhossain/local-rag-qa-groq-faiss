from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("/Users/mdsaibhossain/code/python/End-to-End RAG System/input.txt", "r") as f:
    text = f.read()