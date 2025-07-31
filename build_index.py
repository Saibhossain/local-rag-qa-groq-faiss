from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.faiss import FAISSVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# Load financial documents
documents = SimpleDirectoryReader("data").load_data()

# Chunking
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# Embed model (free)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model = LangchainEmbedding(sbert_model)

# Indexing with FAISS
vector_store = FAISSVectorStore.from_documents(nodes, embed_model=embed_model)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Save index for reuse
index.storage_context.persist()
