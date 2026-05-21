# 📚 PDF Study Buddy

> **AIU Coursera Lab — Track A | Student ID: 21100867**  
> LangChain & LangGraph Specialization — Document Processing & RAG Basics

A Retrieval-Augmented Generation (RAG) system that lets students upload a university lecture PDF and ask natural-language questions about its content, with **page-level citations** for every answer.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/rawan03ayman/pdf-study-buddy.git
cd pdf-study-buddy

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🧱 Architecture 
PDF File
│
▼
PyPDFLoader               ← Loads each page as a Document
│
▼
RecursiveCharacterTextSplitter   ← chunk_size=1000, overlap=200
│
▼
HuggingFaceEmbeddings     ← all-MiniLM-L6-v2 (free, local)
│
▼
ChromaDB (persistent)     ← keyed by SHA-256 file hash
│
▼
RetrievalQA Chain         ← top-4 chunks → Groq LLaMA 3.3 → answer + citations

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| PDF Upload | Any lecture / textbook PDF via Streamlit file uploader |
| Smart Splitting | RecursiveCharacterTextSplitter, configurable chunk size & overlap |
| Free Embeddings | `all-MiniLM-L6-v2` via HuggingFace (no API key needed) |
| Persistent Vector DB | ChromaDB on disk — re-runs skip re-embedding |
| Page Citations | Every answer shows which page(s) the information came from |
| Streamlit Chat UI | Multi-turn conversation with history |
| Groq Models | Choose between LLaMA 3.3-70B, Mixtral, Gemma2 |

---

## 🔑 API Keys

You only need one key:

| Service | Purpose | Where to get |
|---------|---------|-------------|
| Groq | LLaMA 3.3-70B answer generation — **FREE** | [console.groq.com](https://console.groq.com) |

Embeddings are **free** — generated locally using HuggingFace `sentence-transformers`.

---

## 🤖 Available Groq Models

| Model | Speed | Best For |
|-------|-------|----------|
| `llama-3.3-70b-versatile` | Fast | Best quality — recommended |
| `llama-3.1-8b-instant` | Fastest | Quick responses |
| `mixtral-8x7b-32768` | Fast | Long documents |
| `gemma2-9b-it` | Fast | Lightweight tasks |

---

## 📁 Project Structure
pdf-study-buddy/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── chroma_db/          # Persisted vector store (auto-created)
└── <file_hash>/

---

## 📦 Dependencies
langchain
langchain-community
langchain-groq
pypdf
chromadb
sentence-transformers
streamlit

---

## ⚠️ Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: langchain_community` | `pip install langchain-community` |
| `ModuleNotFoundError: langchain_groq` | `pip install langchain-groq` |
| `AuthenticationError` | Check your Groq API key — must start with `gsk_` |
| `sqlite3` version error | `pip install pysqlite3-binary` |
| Slow first load | Downloading ~90 MB embedding model — wait once |
| Empty answers | Increase Top-K chunks in the sidebar |

---

## 👩‍🎓 Author

**Name:** Rawan Ayman  
**Student ID:** 21100867  
**Track:** A — LangChain & LangGraph Specialization  
**Institution:** American International University (AIU)  
**Course:** LangChain MasterClass — Document Processing & RAG Basics
