import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
from functools import lru_cache
from langdetect import detect
import pandas as pd
import re

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

# Gemini config
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

gen_model = genai.GenerativeModel('gemini-1.5-flash')
emb_model = SentenceTransformer('sentence-transformers/LaBSE')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="census_data",
    metadata={"hnsw:space": "cosine"}
)

# Greeting detection
GREETING_PATTERNS = re.compile(r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b', re.I)

# Extract PDF content (include tables as readable text)
def extract_pdf_content(pdf_path):
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                # Extract tables and convert to readable text
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        for _, row in df.iterrows():
                            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                            text += row_text + "\n"
        logger.info(f"Extracted content from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        raise

# Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

# Index PDF
def index_pdf(pdf_path):
    try:
        if collection.count() > 0:
            logger.info("Collection already populated, skipping re-indexing.")
            return
        text = extract_pdf_content(pdf_path)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No content extracted from PDF")
        batch_size = 16
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings.extend(emb_model.encode(batch).tolist())
        ids = [str(i) for i in range(len(chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks
        )
        logger.info(f"Indexed {len(chunks)} chunks into Chroma")
    except Exception as e:
        logger.error(f"Error indexing PDF: {e}")
        raise

# Rebuild index
def rebuild_index(pdf_path):
    try:
        chroma_client.delete_collection("census_data")
        global collection
        collection = chroma_client.get_or_create_collection(
            name="census_data",
            metadata={"hnsw:space": "cosine"}
        )
        index_pdf(pdf_path)
        logger.info("Index rebuilt successfully")
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise

# Retrieve chunks
def retrieve_chunks(query, top_k=5, distance_threshold=0.7):
    try:
        if collection.count() == 0:
            logger.error("Chroma collection is empty. Please rebuild the index.")
            return [], []
        query_emb = emb_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        documents = results['documents'][0]
        distances = results['distances'][0]
        relevant = [(doc, dist) for doc, dist in zip(documents, distances) if dist < distance_threshold]
        return [doc for doc, _ in relevant], [dist for _, dist in relevant]
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return [], []

# Generate answer using Gemini
def generate_answer(query, chunks):
    if not chunks:
        return "I'm sorry, I can only answer questions about the Tanzania Population & Housing Census 2022. Please ensure the index is built."

    # Handle greetings
    if GREETING_PATTERNS.search(query):
        return "Hello! I’m Tanzania Data Bot. I can answer questions about the Population & Housing Census 2022. How can I help you today?"

    context = "\n\n".join(chunks)
    prompt = (
        "Answer in English only. Summarize any table content clearly in text form. "
        "Provide a concise 2-4 sentence answer based on the Tanzania Population & Housing Census 2022. "
        f"Question: {query}\nContext: {context}"
    )
    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"Fallback response:\n" + "\n".join(chunks[:2])

# Cached pipeline
@lru_cache(maxsize=15)
def query_pipeline(query):
    try:
        chunks, distances = retrieve_chunks(query)
        answer = generate_answer(query, chunks)
        return answer, chunks
    except Exception as e:
        logger.error(f"Error in query pipeline: {e}")
        return "Sorry, an error occurred while processing your query.", []
