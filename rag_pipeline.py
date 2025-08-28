# rag_pipeline.py
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
from functools import lru_cache

# Set up logging for backend errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize generative model
gen_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize embedding model (multilingual support for English and Kiswahili)
emb_model = SentenceTransformer('sentence-transformers/LaBSE')

# Initialize Chroma persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection with cosine distance for better semantic similarity
collection = chroma_client.get_or_create_collection(
    name="census_data",
    metadata={"hnsw:space": "cosine"}  # Cosine distance metric
)

# Function to extract text and tables from PDF using pdfplumber
def extract_pdf_content(pdf_path):
    """
    Extracts text and tables from the given PDF file.
    :param pdf_path: Path to the PDF file.
    :return: Combined text content including tables.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                # Extract tables and convert to text
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += "\t".join([str(cell) if cell else "" for cell in row]) + "\n"
                    text += "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        raise

# Function to chunk the text into smaller pieces for embedding
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the text into chunks with overlap for better context retention.
    :param text: Input text.
    :param chunk_size: Size of each chunk.
    :param overlap: Overlap between chunks.
    :return: List of text chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Function to index the PDF content into Chroma
def index_pdf(pdf_path):
    """
    Processes the PDF, chunks it, generates embeddings, and stores in Chroma.
    :param pdf_path: Path to the PDF file.
    """
    try:
        text = extract_pdf_content(pdf_path)
        chunks = chunk_text(text)
        embeddings = emb_model.encode(chunks).tolist()  # Convert to list for Chroma
        ids = [str(i) for i in range(len(chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks
        )
        logger.info("PDF indexed successfully.")
    except Exception as e:
        logger.error(f"Error indexing PDF: {e}")
        raise

# Function to rebuild the index by deleting and re-indexing
def rebuild_index(pdf_path):
    """
    Deletes the existing collection and rebuilds the index from scratch.
    :param pdf_path: Path to the PDF file.
    """
    try:
        chroma_client.delete_collection("census_data")
        global collection
        collection = chroma_client.get_or_create_collection(
            name="census_data",
            metadata={"hnsw:space": "cosine"}
        )
        index_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise

# Function to retrieve relevant chunks based on query embedding
def retrieve_chunks(query, top_k=5, distance_threshold=0.8):
    """
    Embeds the query and retrieves top_k relevant chunks from Chroma.
    Filters by distance threshold to ensure relevance.
    :param query: User query.
    :param top_k: Number of chunks to retrieve.
    :param distance_threshold: Max cosine distance for relevance (lower is better).
    :return: List of relevant documents and their distances.
    """
    try:
        query_emb = emb_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        documents = results['documents'][0]
        distances = results['distances'][0]
        # Filter relevant chunks
        relevant = [(doc, dist) for doc, dist in zip(documents, distances) if dist < distance_threshold]
        return [doc for doc, _ in relevant], [dist for _, dist in relevant]
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return [], []

# Function to generate answer using Gemini
def generate_answer(query, chunks):
    """
    Generates a concise answer using the generative model based on retrieved chunks.
    Falls back to raw chunks if model fails.
    :param query: User query.
    :param chunks: List of relevant document chunks.
    :return: Generated answer.
    """
    if not chunks:
        return "I'm sorry, but I can only answer questions about the Tanzania Population & Housing Census 2022."
    
    context = "\n\n".join(chunks)
    prompt = (
        "Answer the question concisely in 2-5 sentences about the Tanzania Population & Housing Census 2022 "
        "using the following context. Use bullet points for exact figures if present. "
        f"Question: {query}\nContext: {context}"
    )
    
    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        # Fallback to extracted text
        return "Fallback response (model unavailable):\n" + "\n".join(chunks[:2])

# Cached query pipeline for last 5 queries
@lru_cache(maxsize=5)
def query_pipeline(query):
    """
    Full RAG pipeline: Retrieve relevant chunks and generate answer.
    :param query: User query (supports English, Kiswahili, mixed).
    :return: Answer and list of source excerpts.
    """
    try:
        chunks, _ = retrieve_chunks(query)
        answer = generate_answer(query, chunks)
        return answer, chunks  # Return answer and sources
    except Exception as e:
        logger.error(f"Error in query pipeline: {e}")
        return "Sorry, an error occurred while processing your query.", []