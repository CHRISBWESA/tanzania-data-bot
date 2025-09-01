import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import process_query, initialize

# Load environment variables
load_dotenv()

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to reset conversation
def reset_conversation():
    st.session_state.messages = []

# Streamlit UI Configuration
st.set_page_config(page_title="Tanzania Data Bot", page_icon="🇹🇿", layout="wide")

# Sidebar for controls
with st.sidebar:
    st.title("Tanzania Data Bot")
    st.markdown("Powered by Tanzania Population & Housing Census 2022 Data")
    if st.button("Reset Conversation"):
        reset_conversation()
    if st.button("Index Data"):
        try:
            initialize(force=True)
            st.success("Data indexed successfully!")
        except Exception as e:
            st.error(f"Error indexing data: {str(e)}")
    st.markdown("---")
    st.markdown("### Example Questions:")
    st.markdown("- What is the population of Dar es Salaam?")
    st.markdown("- List top 10 regions by population")
    st.markdown("- Idadi ya watu katika Dodoma?")
    st.markdown("- Which region has more buildings: Arusha or Mwanza?")
    st.markdown("- Top 10 regions by schools")
    st.markdown("---")
    feedback_url = "https://wa.me/255746044144"
    st.markdown(f"[Provide Feedback]({feedback_url})")

# Main chat interface
st.title("Tanzania Data Bot MVP")
st.markdown("Ask questions about the 2022 Population & Housing Census in English or Kiswahili.")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources/Excerpts"):
                st.json(message["sources"])

# User input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the query
    try:
        response, sources = process_query(prompt)
    except Exception as e:
        response = f"Error processing query: {str(e)}"
        sources = None
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
    with st.chat_message("assistant"):
        st.markdown(response)
        if sources:
            with st.expander("Sources/Excerpts"):
                st.json(sources)

# Footer credit
st.markdown("---")
st.markdown("<center>Developed by Chris Bwesa & Fedelika Maxmus</center>", unsafe_allow_html=True)
