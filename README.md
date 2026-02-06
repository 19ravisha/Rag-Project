📘 Enterprise RAG System (Policy & Handbook Q&A)

🧩 Problem Statement
Large organizations maintain extensive employee handbooks, HR policies, and compliance documents in PDF format.
Employees and HR teams struggle to quickly retrieve accurate answers from these long documents.

This project builds a Retrieval-Augmented Generation (RAG) system that allows users to ask natural language questions and receive precise answers grounded in company policy documents.


💡 Solution Overview
We implement an end-to-end RAG pipeline:
Parse and chunk enterprise PDF documents (employee handbooks & policies)
Generate embeddings using SentenceTransformers
Store embeddings in FAISS vector database
Retrieve relevant chunks for a query
Use OpenAI LLM to generate grounded answers

🏗️ Architecture :

User Query
   |
FastAPI (/query endpoint)
   |
FAISS Vector Search
   |
Top-k Relevant Chunks
   |
OpenAI LLM (Answer Generation)
   |
Final Answer + Source Documents

🛠️ Tech Stack :

Python 3.13.9
FastAPI – API framework
FAISS – vector similarity search
SentenceTransformers – embedding model
OpenAI API – LLM for response generation
pdfplumber – PDF parsing
python-dotenv – environment variable management
Docker – containerization
GitHub – version control

Project Structure
project/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
├── data/
│   └── pdfs/
├── config/
│   └── .env
├── faiss.index
└── chunks.pkl



