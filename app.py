# app.py
from fastapi import FastAPI
from query_engine import query_engine

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Financial RAG System is running."}

@app.get("/ask/")
def ask(query: str):
    response = query_engine.query(query)
    return {"question": query, "answer": str(response)}
