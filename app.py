# app.py
import streamlit as st # type: ignore
import altair as alt # type: ignore
import pandas as pd # type: ignore
from rag_pipeline import load_and_index_pdf, query_pdf
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np # type: ignore

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="🇹🇿 TANZANIA DATA BOT", page_icon="🤖", layout="wide")

# -----------------------------
# Embeddings
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* Page background */
.stApp { background: linear-gradient(120deg, #e6f2ff, #ffffff); }

/* Chat bubbles */
.chat-bubble-user {
    background-color: #004080; color: white; padding: 12px;
    border-radius: 18px; max-width: 70%; margin-bottom: 10px;
    float:right; clear:both; font-size: 0.95em;
}
.chat-bubble-assistant {
    background-color: #dbe6f1; color: #004080; padding: 12px;
    border-radius: 18px; max-width: 70%; margin-bottom: 10px;
    float:left; clear:both; font-size: 0.95em;
}

/* Sticky header */
.title {
    text-align: center; font-size: 2.2em; font-weight: bold;
    color: white; position: sticky; top: 0; z-index: 1000;
    background: linear-gradient(120deg, #004080, #007bff);
    padding: 12px; border-radius: 0 0 15px 15px;
}

/* Sidebar headers */
.sidebar-header { font-weight: bold; font-size: 1.2em; margin-top: 15px; }

/* Charts container */
.chart-container { max-height: 400px; overflow-y: auto; margin-bottom: 15px; }

/* Footer */
.footer { text-align:center; color: gray; font-size:0.9em; margin-top:20px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sticky Title
# -----------------------------
st.markdown('<div class="title">🤖 TANZANIA DATA BOT</div>', unsafe_allow_html=True)
st.write("Ask questions about Tanzania's statistics or request visual comparisons across regions.")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 About TANZANIA DATA BOT</div>', unsafe_allow_html=True)
    st.write("""
    Insights based on Tanzania National Bureau of Statistics (Census 2002 & 2012).
    Ask questions or request visual comparisons.
    """)
    st.info("💡 Friendly & supportive bot.")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown('<div class="sidebar-header">🕘 Previous Chats</div>', unsafe_allow_html=True)
    for i, (role, msg) in enumerate(st.session_state.history):
        if role == "user":
            if st.button(f"{msg[:40]}...", key=f"hist_{i}"):
                st.session_state.selected_history = i

# -----------------------------
# Load & index multiple PDFs
# -----------------------------
with st.spinner("📂 Loading & indexing NBS reports..."):
    pdf_files = ["nbs_report.pdf", "additional_report.pdf"]
    for pdf_file in pdf_files:
        load_and_index_pdf(pdf_file)

# -----------------------------
# Multi-turn chat
# -----------------------------
if "selected_history" in st.session_state:
    idx = st.session_state.selected_history
    st.chat_input(st.session_state.history[idx][1])
    del st.session_state.selected_history

# Display chat history
for role, msg in st.session_state.history:
    style = "chat-bubble-user" if role=="user" else "chat-bubble-assistant"
    st.markdown(f"<div class='{style}'>{msg}</div>", unsafe_allow_html=True)

# -----------------------------
# Sample dataset for charts
# -----------------------------
DATA = {
    "Iringa": 1500000,
    "Mbeya City": 1200000,
    "Mbeya District": 1300000,
    "Dar es Salaam": 5000000,
    "Dodoma": 3000000
}
CHART_HISTORY = st.session_state.get("chart_history", [])

# -----------------------------
# Utilities
# -----------------------------
def is_greeting(msg: str) -> bool:
    return any(msg.lower().startswith(g) for g in [
        "hi","hello","hey","how are you","who are you","what can you do","how can you assist"
    ])

def greeting_response(msg: str) -> str:
    import random
    msg_lower = msg.lower()
    if "how are you" in msg_lower:
        return "😊 I'm calm and ready to assist you. How can I help?"
    elif "who are you" in msg_lower:
        return "🤖 I am Tanzania Data Bot, your assistant for National Bureau of Statistics (2002 & 2012) data."
    elif "what can you do" in msg_lower or "how can you assist" in msg_lower:
        return "I can answer questions about Tanzania's population, regions, and trends, generate charts, and guide you through statistics."
    return random.choice([
        "👋 Hello! I provide insights from Tanzania statistics.",
        "🤝 Welcome! Ask anything about population, regions, or trends.",
        "😊 Hi there! Ready to explore Tanzania’s data together."
    ])

def is_chart_request(msg: str) -> bool:
    keywords = ["draw", "chart", "plot", "compare", "graph"]
    return any(word in msg.lower() for word in keywords)

def plot_user_chart(msg: str):
    input_vec = embedder.encode([msg])
    region_vecs = embedder.encode(list(DATA.keys()))
    sims = cosine_similarity(input_vec, region_vecs)[0]
    regions = [r for r, s in zip(DATA.keys(), sims) if s > 0.5]

    if not regions:
        return "⚠️ Could not identify regions. Try specifying clearly, e.g., 'Iringa', 'Mbeya'."

    df = pd.DataFrame({
        "Region": regions,
        "Value": [DATA[r] for r in regions]
    })

    chart = alt.Chart(df).mark_bar(color="#1f77b4").encode(
        x="Region",
        y="Value",
        tooltip=["Region","Value"]
    ).interactive()

    CHART_HISTORY.append((df.copy(), msg))
    st.session_state.chart_history = CHART_HISTORY

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return "📊 Chart created!"

# -----------------------------
# Handle user input
# -----------------------------
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.history.append(("user", user_input))
    st.markdown(f"<div class='chat-bubble-user'>{user_input}</div>", unsafe_allow_html=True)

    with st.spinner("🤔 Thinking..."):
        if is_greeting(user_input):
            answer = greeting_response(user_input)
        elif is_chart_request(user_input):
            answer = plot_user_chart(user_input)
        else:
            # Queries now search across both PDFs
            raw_answer = query_pdf(user_input)
            answer = f"{raw_answer}\n\n(From NBS Census 2002 & 2012 & additional report)"

    st.markdown(f"<div class='chat-bubble-assistant'>{answer}</div>", unsafe_allow_html=True)
    st.session_state.history.append(("assistant", answer))

# -----------------------------
# Show chart history
# -----------------------------
if CHART_HISTORY:
    st.markdown('<hr>')
    st.subheader("📊 Chart History")
    for i, (df_chart, prompt) in enumerate(CHART_HISTORY):
        st.write(f"Prompt: {prompt}")
        chart = alt.Chart(df_chart).mark_bar(color="#ff7f0e").encode(
            x="Region",
            y="Value",
            tooltip=["Region", "Value"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        if st.button(f"Delete Chart {i}", key=f"del_chart_{i}"):
            CHART_HISTORY.pop(i)
            st.session_state.chart_history = CHART_HISTORY
            st.experimental_rerun()

# -----------------------------
# Footer credit
# -----------------------------
st.markdown("""
<div class="footer">
Created by Chris Bwesa & Fedelika Maxmus
</div>
""", unsafe_allow_html=True)
