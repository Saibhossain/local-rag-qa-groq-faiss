# Local RAG QA System with FAISS and Groq LLaMA 70B

This project builds a simple **Retrieval-Augmented Generation (RAG)** system that:
- Reads text documents
- Splits them into chunks
- Converts them into vector embeddings
- Stores them in the FAISS index
- Uses a local or hosted LLaMA 70B model (via Groq API) to answer your questions

## 🔧 How it works
1. **Load Documents**: From `data/` folder
2. **Chunk & Embed**: Split into pieces and create vector embeddings
3. **FAISS Index**: Stores embeddings for similarity search
4. **Query Engine**: Takes a user question, searches relevant chunks
5. **LLM Response**: Sends context + question to LLaMA-70B and returns an answer

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

## Setup .env

```bash
GROQ_API_URL=https://console.groq.com/home { create your own api key }
```

### Run

```bash
api_generate_ans.py
```
