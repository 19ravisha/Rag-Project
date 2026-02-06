from fastapi import FastAPI
from openai import OpenAI
import faiss, pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

import os
from openai import OpenAI

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/query")
def query_rag(q: str):
    query_vec = model.encode([q])
    D, I = index.search(query_vec, 3)

    context = "\n".join([chunks[i]["text"] for i in I[0]])

    prompt = f"Answer only using context:\n{context}\n\nQuestion:{q}"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )

    return {"answer": response.choices[0].message.content}
