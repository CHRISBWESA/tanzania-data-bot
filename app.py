import streamlit as st # type: ignore
import subprocess

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📘 RAG Chatbot with FAISS")

st.write("Navigate to **1_Chat.py** in the sidebar to start chatting.")

# Button to re-index PDFs
if st.button("Rebuild Vectorstore"):
    with st.spinner("Re-indexing PDFs..."):
        subprocess.run(["python", "rag_pipeline.py"])
    st.success("Vectorstore rebuilt successfully!")
