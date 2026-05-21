"""
PDF Study Buddy — RAG-based Document Q&A System
Student ID: 21100867 | Track A | AIU Coursera Lab
Powered by: Groq API (llama-3.3-70b) + HuggingFace Embeddings + ChromaDB
"""

import os
import hashlib
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Study Buddy",
    page_icon="📚",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .groq-badge {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 6px;
    }
    .answer-box {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .source-box {
        background: #e3f2fd;
        border-left: 4px solid #1565c0;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #f3f4f6;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card .number { font-size: 1.6rem; font-weight: bold; color: #1a237e; }
    .metric-card .label  { font-size: 0.78rem; color: #555; }

    .step-header {
        background: #1a237e;
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 PDF Study Buddy</h1>
    <p>Upload a lecture PDF · Ask questions · Get cited answers</p>
    <span class="groq-badge">⚡ Powered by Groq — Ultra-Fast AI</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GROQ MODELS AVAILABLE
# ─────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile   (Best — Recommended":    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant      (Fastest":               "llama-3.1-8b-instant",
    "mixtral-8x7b-32768        (long context)":            "mixtral-8x7b-32768",
    "gemma2-9b-it              (Lightweight and Fast)":           "gemma2-9b-it",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_file_hash(file_bytes: bytes) -> str:
    """SHA-256 hash → unique Chroma collection name per PDF."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def save_uploaded_file(uploaded_file) -> str:
    """Write uploaded bytes to /tmp and return path."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    return tmp.name


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load HuggingFace embedding model (cached once per session)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def ingest_pdf(pdf_path: str, file_hash: str) -> Chroma:
    """
    Full RAG ingestion pipeline:
      1. Load PDF pages
      2. Split into chunks
      3. Embed with HuggingFace (FREE — no API key needed)
      4. Store in persistent ChromaDB (keyed by file hash)
    Returns a Chroma vectorstore.
    """
    persist_dir = f"./chroma_db/{file_hash}"
    embeddings  = load_embeddings()

    # ── Reload from disk if already embedded ──────────────────────────────
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        st.toast("✅ Loaded cached embeddings from disk", icon="💾")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=file_hash,
        )

    # ── Step 1: Document Loader ────────────────────────────────────────────
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ── Step 2: Text Splitter ──────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # ── Step 3 + 4: Embed → Store in ChromaDB ─────────────────────────────
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=file_hash,
    )
    st.toast(f"✅ Embedded {len(chunks)} chunks into ChromaDB", icon="🗂️")
    return vectorstore


def build_qa_chain(vectorstore: Chroma, groq_api_key: str, model_id: str) -> RetrievalQA:
    """Build a RetrievalQA chain using Groq LLM."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model_name=model_id,
        temperature=0,
        groq_api_key=groq_api_key,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Groq Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your FREE key from: console.groq.com",
    )

    # Link to get free key
    st.markdown(
        "🔑 [احصل على مفتاح مجاني من Groq](https://console.groq.com)",
        unsafe_allow_html=False,
    )

    st.markdown("---")
    st.markdown("### 🤖 اختر النموذج")
    selected_label = st.selectbox(
        "Groq Model",
        options=list(GROQ_MODELS.keys()),
        index=0,
    )
    selected_model = GROQ_MODELS[selected_label]
    st.caption(f"Model ID: `{selected_model}`")

    st.markdown("---")
    st.markdown("### 📖 كيف يعمل التطبيق")
    st.markdown("""
1. **ارفع** ملف PDF المحاضرة  
2. **معالجة** — تقسيم النص وتحويله لـ embeddings  
3. **اسأل** سؤالك في الدردشة  
4. التطبيق يـ**بحث** عن أقرب 4 أجزاء ذات صلة  
5. **Groq** يولد الإجابة مع أرقام الصفحات  
""")

    st.markdown("---")
    st.markdown("### 🔧 إعدادات RAG")
    chunk_size    = st.slider("Chunk Size",    500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0,   500,  200,  50)
    top_k         = st.slider("Top-K Chunks",  1,   8,    4,    1)

    st.markdown("---")
    st.caption("Student ID: 21100867 · Track A · AIU Coursera Lab")


# ─────────────────────────────────────────────
# MAIN AREA — TWO COLUMNS
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

# ══════════════════════════════════════════════
# LEFT COLUMN — Upload & Stats
# ══════════════════════════════════════════════
with col_left:
    st.markdown('<div class="step-header">STEP 1 — Upload PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a lecture PDF",
        type=["pdf"],
        help="Upload any university lecture or textbook PDF.",
    )

    if uploaded:
        file_bytes = uploaded.getvalue()
        file_hash  = get_file_hash(file_bytes)

        st.success(f"**{uploaded.name}**  \n`{len(file_bytes):,}` bytes · hash: `{file_hash[:8]}…`")

        st.markdown('<div class="step-header">STEP 2 — Ingesting…</div>', unsafe_allow_html=True)
        with st.spinner("Loading, splitting & embedding PDF…"):
            pdf_path    = save_uploaded_file(uploaded)
            vectorstore = ingest_pdf(pdf_path, file_hash)

        # Stats
        col_a, col_b = st.columns(2)
        collection   = vectorstore._collection
        n_chunks     = collection.count()
        n_pages      = len(PyPDFLoader(pdf_path).load())

        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="number">{n_pages}</div>
                <div class="label">Pages</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="number">{n_chunks}</div>
                <div class="label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warning-box">
            📐 <b>Chunk config:</b> size={chunk_size}, overlap={chunk_overlap}<br>
            🔍 <b>Retrieval:</b> top-{top_k} chunks per query<br>
            🧠 <b>Embeddings:</b> all-MiniLM-L6-v2 (HuggingFace — Free)<br>
            ⚡ <b>LLM:</b> {selected_model} via Groq
        </div>""", unsafe_allow_html=True)

        st.session_state["vectorstore"] = vectorstore
        st.session_state["pdf_ready"]   = True
    else:
        st.info("⬆️ Upload a PDF to get started.")
        st.session_state["pdf_ready"] = False


# ══════════════════════════════════════════════
# RIGHT COLUMN — Chat Q&A
# ══════════════════════════════════════════════
with col_right:
    st.markdown('<div class="step-header">STEP 3 — Ask Questions</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render existing chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Source Citations", expanded=False):
                    for src in msg["sources"]:
                        page_num = src["page"] + 1
                        snippet  = src["snippet"][:300].replace("\n", " ")
                        st.markdown(f"""
                        <div class="source-box">
                            📄 <b>Page {page_num}</b><br>
                            <i>{snippet}…</i>
                        </div>""", unsafe_allow_html=True)

    # Guard checks
    if not st.session_state.get("pdf_ready"):
        st.markdown('<div class="warning-box">⚠️ Please upload a PDF first.</div>',
                    unsafe_allow_html=True)
    elif not groq_key:
        st.markdown(
            '<div class="warning-box">⚠️ أدخل Groq API Key في الشريط الجانبي.'
            ' احصل عليه مجاناً من <a href="https://console.groq.com" target="_blank">console.groq.com</a></div>',
            unsafe_allow_html=True,
        )
    else:
        query = st.chat_input("Ask a question about your lecture…")
        if query:
            st.session_state["messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("⚡ Groq is thinking…"):
                    qa_chain = build_qa_chain(
                        st.session_state["vectorstore"],
                        groq_key,
                        selected_model,
                    )
                    result = qa_chain.invoke({"query": query})

                answer      = result["result"]
                source_docs = result["source_documents"]

                st.markdown(f'<div class="answer-box">{answer}</div>',
                            unsafe_allow_html=True)

                sources_data = []
                if source_docs:
                    with st.expander("📎 Source Citations", expanded=True):
                        for doc in source_docs:
                            page_num = doc.metadata.get("page", 0) + 1
                            snippet  = doc.page_content[:300].replace("\n", " ")
                            st.markdown(f"""
                            <div class="source-box">
                                📄 <b>Page {page_num}</b><br>
                                <i>{snippet}…</i>
                            </div>""", unsafe_allow_html=True)
                            sources_data.append({
                                "page":    doc.metadata.get("page", 0),
                                "snippet": doc.page_content,
                            })

            st.session_state["messages"].append({
                "role":    "assistant",
                "content": answer,
                "sources": sources_data,
            })

    if st.session_state.get("messages"):
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
