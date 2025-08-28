"""
RAG pipeline for Tanzania Data Bot
- Indexes a single Census 2022 PDF
- Stores embeddings in Chroma (in-memory, no SQLite needed)
- Supports retrieval-augmented generation
"""

import os
import pickle
from typing import List
import pdfplumber  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import chromadb  # type: ignore
import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv  # type: ignore

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file")
genai.configure(api_key=API_KEY)

# Paths / Settings
PDF_FILE = "additional_report.pdf"  # only 2022 Census
CHROMA_COLLECTION = "census_2022"
CACHE_FILE = "pdf_indexed.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# -----------------------------
# Environment tweaks
# -----------------------------
os.environ["CHROMA_TELEMETRY"] = "false"

# -----------------------------
# Globals
# -----------------------------
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ FIX: Use EphemeralClient (in-memory Chroma) → avoids sqlite3 requirement
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"source": "census_2022"}
)

# -----------------------------
# Helpers
# -----------------------------
def _is_indexed() -> bool:
    return os.path.exists(CACHE_FILE) and collection.count() > 0

def _mark_indexed():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(True, f)

def _extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pieces: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pieces.append(page_text)
    return "\n".join(pieces).strip()

def _split_text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Indexing
# -----------------------------
def load_and_index_pdf(force_reindex: bool = False):
    """Load and index the 2022 Census PDF"""
    global collection
    if _is_indexed() and not force_reindex:
        return

    if force_reindex:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"source": "census_2022"}
        )

    print(f"Indexing: {PDF_FILE} ...")
    text = _extract_text_from_pdf(PDF_FILE)
    if not text:
        print(f"Warning: {PDF_FILE} yielded no text, skipping.")
        return

    chunks = _split_text_to_chunks(text)
    embeddings = _embedder.encode(chunks).tolist()
    ids = [f"{os.path.basename(PDF_FILE)}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(PDF_FILE), "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    _mark_indexed()
    print(f"Completed indexing. Total chunks: {len(chunks)}")

# ✅ FIX: Wrapper to match app.py import
def load_and_index_all_pdfs(force_reindex: bool = False):
    """Wrapper for app.py compatibility (calls load_and_index_pdf)."""
    return load_and_index_pdf(force_reindex=force_reindex)

# -----------------------------
# Query
# -----------------------------
def query_pdf(question: str, top_k: int = TOP_K, max_context_chars: int = 3000) -> str:
    if collection.count() == 0:
        return "No documents indexed. Please ensure the 2022 Census PDF is present."

    q_emb = _embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        return "I couldn't find relevant information in the 2022 Census. Try rephrasing your question."

    selected_parts = []
    total_len = 0
    for doc, meta, dist in zip(docs, metas, dists):
        if total_len + len(doc) > max_context_chars:
            break
        source_tag = meta.get("source", "") if meta else ""
        snippet = f"[{source_tag}] {doc}" if source_tag else doc
        selected_parts.append(snippet)
        total_len += len(doc)

    context = "\n\n".join(selected_parts).strip()
    if not context:
        return "I couldn't find a focused context for that question."

    prompt = f"""
You are TANZANIA DATA BOT. Use ONLY the provided Context (from Census 2022 report).
If the answer cannot be found in the Context, say you cannot find it in the documents.
Be concise and exact (2-4 sentences). Do not add unrelated information.

Context:
{context}

Question: {question}
Answer:
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
    except Exception:
        return "There was an error querying the language model. Please try again."

    if response and getattr(response, "text", None):
        return response.text.strip()
    return "No answer generated by the model."

# -----------------------------
# Force reindex
# -----------------------------
def force_reindex():
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
    except Exception:
        pass
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    global collection
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"source": "census_2022"}
    )
    load_and_index_pdf(force_reindex=True)
