# query_engine.py
from llama_index.core import StorageContext, load_index_from_storage
from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.faiss import FaissVectorStore

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

query_engine = index.as_query_engine()
