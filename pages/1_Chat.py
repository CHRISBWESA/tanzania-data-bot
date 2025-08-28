import streamlit as st # type: ignore
from rag_pipeline import load_vectorstore
from langchain.chains import RetrievalQA # type: ignore
from langchain_openai import ChatOpenAI # type: ignore

# Load FAISS vectorstore
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Setup RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

st.title("📘 RAG Chatbot")

# Input area with Send button
user_input = st.text_area("Enter your question:", key="user_input")
if st.button("Send"):
    if user_input.strip():
        response = qa_chain({"query": user_input})
        st.markdown(f"**Answer:** {response['result']}")

        # Show sources
        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.markdown(f"- {doc.metadata['source']}")
