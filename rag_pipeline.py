import os
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_openai import OpenAIEmbeddings # type: ignore

# Function to load and index PDFs
def load_and_index_all_pdfs(pdf_folder="pdfs", db_path="vectorstore"):
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"Folder '{pdf_folder}' does not exist.")

    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    # Create FAISS index
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Save index
    vectorstore.save_local(db_path)

    return vectorstore


# Function to load FAISS vectorstore
def load_vectorstore(db_path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
