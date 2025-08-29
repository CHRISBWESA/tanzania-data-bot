import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import query_pipeline, rebuild_index

# Load environment variables
load_dotenv()

# Path to PDF
PDF_PATH = "nbs.pdf"

# Initialize session state
if "messages" not in st.session_state:
    greeting = "Hello! I’m Tanzania Data Bot. I can answer questions about Population & Housing Census 2022. How can I help you today?"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

st.set_page_config(page_title="Tanzania Data Bot", page_icon="📊")
st.title("Tanzania Data Bot")

# Sidebar examples and actions
st.sidebar.title("Example Questions")
example_questions = [
    "Hello",
    "Hi",
    "What is the total population of Tanzania in 2022?",
    "How many households are there in Dar es Salaam?",
    "What are the key statistics on dwelling units?"
]

for example in example_questions:
    if st.sidebar.button(example):
        st.session_state.messages.append({"role": "user", "content": example})
        with st.chat_message("user"):
            st.markdown(example)
        with st.spinner("Thinking..."):
            answer, sources = query_pipeline(example)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

# Sidebar button to rebuild index
if st.sidebar.button("Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        try:
            rebuild_index(PDF_PATH)
            st.sidebar.success("Index rebuilt successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to rebuild index: {str(e)}")

# WhatsApp feedback
st.sidebar.markdown("---")
st.sidebar.subheader("Feedback")
st.sidebar.markdown(
    """
    <a href="https://wa.me/+255746044144?text=Feedback%20for%20Tanzania%20Data%20Bot" target="_blank">
        <button style="background-color: #25D366; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            Send Feedback via WhatsApp
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    st.text(f"Excerpt {idx}: {source[:200]}...")

# Chat input
user_input = st.chat_input("Ask a question about the Tanzania Population & Housing Census 2022")

if user_input:
    user_input = user_input.strip()
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            answer, sources = query_pipeline(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

# Reset conversation
if st.button("Reset Conversation"):
    greeting = "Hello! I’m Tanzania Data Bot. I can answer questions about Population & Housing Census 2022. How can I help you today?"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    st.rerun()
