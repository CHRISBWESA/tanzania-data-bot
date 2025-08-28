"""
RAG pipeline for Tanzania Data Bot
- Focused on 2022 Tanzania Population & Housing Census
- Uses FAISS and SentenceTransformers for retrieval
- Provides simple QA + conversation handling
"""

from typing import List
import numpy as np # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore

# -----------------------------
# Dummy Census Data for demonstration
# -----------------------------
CENSUS_DOCS = [
    "The total population of Tanzania in 2022 is 63 million.",
    "Dar es Salaam has the largest population of over 5 million.",
    "Dodoma has around 3 million people.",
    "Mbeya City has around 1.2 million people.",
    "Iringa region has around 1.5 million people.",
    "The 2022 Census covers population, housing, and basic demographics.",
]

# -----------------------------
# Globals
# -----------------------------
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_index = None
_documents: List[str] = []

# -----------------------------
# Indexing
# -----------------------------
def load_index():
    global _index, _documents
    _documents = CENSUS_DOCS
    embeddings = _embedder.encode(_documents).astype("float32")
    _index = faiss.IndexFlatL2(embeddings.shape[1])
    _index.add(embeddings)

# -----------------------------
# Query
# -----------------------------
def query_census(user_input: str) -> str:
    """
    Returns answer from Census docs or handles greetings/other interactions
    """
    global _index, _documents
    if _index is None or not _documents:
        load_index()

    # conversation handling
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    thanks = ["thank you", "thanks", "thx", "thank you very much", "thanks a lot"]
    bye = ["bye", "goodbye", "see you", "see you later", "farewell"]
    who = ["who are you", "who is he", "who is this", "what is your name"]

    text_lower = user_input.lower()

    # Greetings
    if any(g in text_lower for g in greetings):
        return "Hello! I'm Tanzania Data Bot, your assistant for the 2022 Census."

    # Thanks
    if any(t in text_lower for t in thanks):
        return "You're welcome! 😊"

    # Goodbye
    if any(b in text_lower for b in bye):
        return "Goodbye! Have a nice day."

    # Who/identity
    if any(w in text_lower for w in who):
        return "I'm Tanzania Data Bot, here to provide insights from the 2022 Tanzania Census."

    # Handle out-of-scope
    if "census" not in text_lower and "population" not in text_lower and "housing" not in text_lower:
        return "I can only provide answers about the 2022 Tanzania Census (population & housing). Other data may be available in the next version."

    # Vector search for relevant Census info
    q_emb = _embedder.encode([user_input]).astype("float32")
    D, I = _index.search(q_emb, k=3)
    answers = [f"{_documents[i]}" for i in I[0] if i >= 0]

    if not answers:
        return "I couldn't find an exact answer in the Census report."
    return answers[0]
