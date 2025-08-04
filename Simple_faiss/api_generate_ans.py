from groq import Groq
from dotenv import load_dotenv
from vector_store import retrieve
import openai
import os

load_dotenv()
api_key = os.getenv("GROQ_api_key")
client = Groq(api_key=api_key)


question = "What is Guido?"
context = retrieve(question)

# Build prompt for LLaMA-3
prompt = f"""Context: {' '.join(context)}

Question: {question}
Answer:"""

# Call Groq's LLaMA-3-70B Versatile
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant that answers questions based on context."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=150,
)

# Print generated answer
print("Answer:", response.choices[0].message.content)