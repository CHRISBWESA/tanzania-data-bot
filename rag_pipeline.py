# rag_pipeline.py
import os
import pickle
import pdfplumber  # type: ignore
from typing import Optional, List
from dotenv import load_dotenv # type: ignore
from tqdm import tqdm # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np # type: ignore

import google.generativeai as genai  # type: ignore # Gemini

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file")

genai.configure(api_key=API_KEY)

# -----------------------------
# Globals
# -----------------------------
PERSIST_FILE = "pdf_indexed.pkl"
VECTOR_FILE = "vectorstore.pkl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_vectorstore: Optional[List[dict]] = None
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Helpers
# -----------------------------
def _is_indexed() -> bool:
    return os.path.exists(PERSIST_FILE)

def _mark_indexed():
    with open(PERSIST_FILE, "wb") as f:
        pickle.dump(True, f)

def load_and_index_pdf(pdf_path: str = "nbs_report.pdf"):
    """
    Load PDF, split into chunks, embed, store in memory.
    """
    global _vectorstore
    if _is_indexed() and _vectorstore:
        return  # already loaded

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                full_text.append(txt)

    all_text = "\n".join(full_text).strip()
    if not all_text:
        raise ValueError("PDF empty, nothing to index.")

    # Split into chunks
    chunks = []
    start = 0
    while start < len(all_text):
        end = min(start + CHUNK_SIZE, len(all_text))
        chunks.append(all_text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    # Embed chunks
    embeddings = _embedder.encode(chunks).tolist()
    _vectorstore = [{"chunk": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]

    _mark_indexed()
    print(f"Indexed {len(chunks)} chunks from {pdf_path}")

def query_pdf(question: str, top_k: int = 5) -> str:
    """
    Retrieve top chunks using cosine similarity and query Gemini.
    """
    if not _vectorstore:
        raise RuntimeError("Vectorstore not loaded. Run load_and_index_pdf first.")

    # Embed query
    q_vec = _embedder.encode([question])
    chunk_embs = np.array([v["embedding"] for v in _vectorstore])
    sims = np.dot(chunk_embs, q_vec.T).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    context = "\n\n".join([_vectorstore[i]["chunk"] for i in top_idx if sims[i] > 0.4])

    if not context:
        return "I can't find that in the NBS (2002 & 2012) sources."

    prompt = f"""
You are TANZANIA DATA BOT. Answer using ONLY Tanzania National Bureau of Statistics census data (2002 & 2012).
Answer briefly and exactly (2-4 sentences). Do NOT add extra info.

Context:
{context}

Question: {question}
Answer:
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip() if response and response.text else "No answer generated."
