# 📚 DocQA RAG — Chat with Your PDFs

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions in natural language. The system retrieves relevant context using vector search and generates grounded answers using an LLM.

## 🚀 Features

- Upload and process PDF documents
- Semantic search using vector embeddings and FAISS
- LLM-powered question answering
- Source-aware responses with citations
- Streamlit-based interface
- Reindex documents after upload

## 🧠 Architecture

PDFs → Chunking → Embeddings → FAISS Vector Store  
User Query → Similarity Search → Retrieved Context → LLM → Answer + Sources

## 🛠️ Tech Stack

- Python
- LangChain
- FAISS
- OpenAI API
- Streamlit
- PyPDF
- python-dotenv

## 📂 Project Structure

```text
docqa-rag-openai/
├── app.py
├── ingest.py
├── query.py
├── requirements.txt
├── .gitignore
├── .env.example
├── docs/
└── vectorstore/
