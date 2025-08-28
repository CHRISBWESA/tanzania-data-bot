import streamlit as st # type: ignore
from rag_pipeline import query_census

st.set_page_config(
    page_title="🇹🇿 Tanzania Data Bot",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Chat Page Title
# -----------------------------
st.markdown("""
<div style='position:sticky; top:0; background:#fff; z-index:1000; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
    <h2>💬 Tanzania Data Bot</h2>
    <p>Ask me questions about the 2022 Tanzania Census (population & housing).</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Custom CSS for chat
# -----------------------------
st.markdown("""
<style>
.chat-area {
    max-height: 70vh;
    overflow-y: auto;
    padding: 10px;
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
input:focus {
    outline: 2px solid #4f46e5;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# -----------------------------
# Render chat history
# -----------------------------
def render_history():
    st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
    for role, text in st.session_state.history:
        cls = "user-msg" if role == "user" else "bot-msg"
        st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

render_history()

# -----------------------------
# Send message function
# -----------------------------
def send_message():
    user_text = st.session_state.chat_input.strip()
    if not user_text:
        return
    st.session_state.history.append(("user", user_text))
    reply = query_census(user_text)
    st.session_state.history.append(("bot", reply))
    st.session_state.chat_input = ""
    st.experimental_rerun()

# -----------------------------
# Input box + send button
# -----------------------------
st.text_input(
    "Your question:",
    key="chat_input",
    placeholder="Type your question here..."
)
st.button("Send", on_click=send_message)
