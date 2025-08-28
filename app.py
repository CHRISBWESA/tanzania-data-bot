# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import query_pipeline, rebuild_index

# Load environment variables
load_dotenv()

# Path to the fixed PDF file
PDF_PATH = "nbs.pdf"

# Initialize session state for messages if not present
if "messages" not in st.session_state:
    greeting = "Hello! I’m Tanzania Data Bot. I can answer questions about Population & Housing Census 2022. How can I help you today?"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

# Set page title
st.title("Tanzania Data Bot")

# Sidebar for example questions and rebuild index
st.sidebar.title("Example Questions")
example_questions = [
    "What is the total population of Tanzania in 2022?",
    "Idadi ya watu wa Tanzania ni ngapi mwaka 2022?",
    "How many households are there in Dar es Salaam?",
    "Viwango vya makazi katika mikoa ya Tanzania ni vipi?",
    "What are the key statistics on dwelling units?"
]

for example in example_questions:
    if st.sidebar.button(example):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": example})
        # Display user message
        with st.chat_message("user"):
            st.markdown(example)
        # Process query with loading indicator
        with st.spinner("Thinking..."):
            try:
                answer, sources = query_pipeline(example)
            except Exception as e:
                answer = "Sorry, an error occurred. Please try again."
                sources = []
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        # Rerun to update the UI and auto-scroll
        st.rerun()

# Sidebar button to rebuild index
if st.sidebar.button("Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        rebuild_index(PDF_PATH)
    st.sidebar.success("Index rebuilt successfully!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If sources are available, show them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    st.text(f"Excerpt {idx}: {source[:200]}...")  # Truncate for display

# Chat input box
user_input = st.chat_input("Ask a question about the Tanzania Population & Housing Census 2022")

if user_input:
    # Basic input sanitization (strip whitespace)
    user_input = user_input.strip()
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        # Process query with loading indicator
        with st.spinner("Thinking..."):
            try:
                answer, sources = query_pipeline(user_input)
            except Exception as e:
                answer = "Sorry, an error occurred. Please try again."
                sources = []
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        # Rerun to update the UI and auto-scroll
        st.rerun()

# Reset conversation button
if st.button("Reset Conversation"):
    greeting = "Hello! I’m Tanzania Data Bot. I can answer questions about Population & Housing Census 2022. How can I help you today?"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    st.rerun()

# Note: Streamlit handles scrollable chat window, auto-scroll to latest message, and basic mobile responsiveness by default.