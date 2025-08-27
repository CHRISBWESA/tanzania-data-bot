import streamlit as st  # type: ignore
import altair as alt  # type: ignore
import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np  # type: ignore
import os
import json
from rag_pipeline import load_and_index_all_pdfs, query_pdf, force_reindex

HISTORY_DIR = "session_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

st.set_page_config(
    page_title="Tanzania Data Bot - Chat",
    page_icon="💬",
    layout="wide",
)

# -----------------------------
# Load & index PDF
# -----------------------------
with st.spinner("📂 Loading & indexing Census 2022 report..."):
    load_and_index_all_pdfs()

# -----------------------------
# Fixed Title
# -----------------------------
st.markdown("""
    <div style='position:sticky; top:0; background:#fff; z-index:1000; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
        <h2>💬 Tanzania Data Bot Chat</h2>
        <p>Ask me questions about the 2022 Tanzania Census or request charts.</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar for actions
# -----------------------------
with st.sidebar:
    st.header("📊 About")
    st.write("I provide answers grounded in the 2022 Census report only.")

    if "history" not in st.session_state:
        st.session_state.history = []

    session_file = os.path.join(HISTORY_DIR, "session_default.json")
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            st.session_state.history = json.load(f)

    if st.button("🆕 New Chat"):
        st.session_state.history = []
        st.rerun()

    if st.button("🔄 Re-index Data"):
        with st.spinner("Re-indexing..."):
            force_reindex()
        st.success("Re-index complete. Restart to reload.")

# -----------------------------
# Chart helper
# -----------------------------
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
DATA = {"Iringa": 1500000, "Mbeya City": 1200000, "Mbeya District": 1300000, "Dar es Salaam": 5000000, "Dodoma": 3000000}

def plot_user_chart(prompt_text: str):
    input_vec = EMBEDDER.encode([prompt_text])
    region_names = list(DATA.keys())
    region_vecs = EMBEDDER.encode(region_names)
    sims = cosine_similarity(input_vec, region_vecs)[0]
    chosen = [r for r, s in zip(region_names, sims) if s >= 0.45] or [region_names[int(np.argmax(sims))]]

    df = pd.DataFrame({"Region": chosen, "Value": [DATA[r] for r in chosen]})
    chart = alt.Chart(df).mark_bar(cornerRadius=6).encode(
        x=alt.X("Region:N", sort=None),
        y=alt.Y("Value:Q"),
        tooltip=["Region", "Value"]
    ).properties(height=320, width="container")
    return chart

# -----------------------------
# Custom CSS for chat bubbles + input focus
# -----------------------------
st.markdown("""
    <style>
    .chat-area {
        max-height: 65vh;
        overflow-y: auto;
        padding: 10px;
        scroll-behavior: smooth;
    }
    .user-msg {
        background-color: #4f46e5;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 6px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .bot-msg {
        background-color: #f6f9fc;
        color: #003366;
        padding: 12px;
        border-radius: 15px;
        margin: 6px;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .chat-input {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    input:focus {
        outline: 2px solid #4f46e5;
        border-radius: 12px;
    }
    </style>

    <script>
    const inputBox = window.parent.document.querySelector('input[placeholder="Type your question here..."]');
    if(inputBox) { inputBox.focus(); }
    </script>
""", unsafe_allow_html=True)

# -----------------------------
# Chat Loop
# -----------------------------
def render_history():
    st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
    for role, txt in st.session_state.history:
        bubble_class = "user-msg" if role == "user" else "bot-msg"
        st.markdown(f"<div class='{bubble_class}'>{txt}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

render_history()

# -----------------------------
# User input fixed at bottom
# -----------------------------
def process_input(user_input: str):
    st.session_state.history.append(("user", user_input))
    render_history()

    if any(k in user_input.lower() for k in ["chart", "plot", "graph", "compare"]):
        chart = plot_user_chart(user_input)
        st.altair_chart(chart, use_container_width=True)
        st.session_state.history.append(("assistant", f"📊 Chart generated for: {user_input}"))
    else:
        answer = query_pdf(user_input)
        st.session_state.history.append(("assistant", answer))
        st.markdown(f"<div class='bot-msg'>{answer}</div>", unsafe_allow_html=True)

    # Save session
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)

# Text input with "Enter to send"
user_input = st.text_input("", placeholder="Type your question here...", key="chat_input")
if user_input:
    process_input(user_input)
    st.experimental_rerun()  # Refresh so input clears after Enter
