import streamlit as st  # type: ignore

st.set_page_config(
    page_title="🇹🇿 Tanzania Data Bot",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Landing Page Styling
# -----------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #4f46e5, #8b5cf6); /* softer indigo → violet */
        color: white;
        padding: 80px;
        text-align: center;
        border-radius: 20px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
    }
    .hero h1 {
        font-size: 3.5em;
        margin-bottom: 15px;
        font-weight: 800;
    }
    .hero p {
        font-size: 1.3em;
        margin-bottom: 40px;
        font-weight: 400;
    }
    .btn {
        background-color: #ffd700;
        color: #4f46e5;
        padding: 15px 32px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1em;
        text-decoration: none;
        transition: 0.3s;
        margin: 8px;
        display: inline-block;
    }
    .btn:hover {
        background-color: #ffcc00;
        transform: scale(1.05);
    }
    .features {
        display: flex; justify-content: center; flex-wrap: wrap; margin-top: 50px;
    }
    .feature-card {
        background-color: #f6f9fc;
        padding: 25px;
        margin: 12px;
        border-radius: 15px;
        width: 260px;
        text-align: center;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0px 10px 20px rgba(0,0,0,0.15);
    }
    .feature-card h3 {
        color: #4f46e5;
        margin-bottom: 12px;
    }
    .footer {
        text-align:center;
        color: gray;
        font-size:0.95em;
        margin-top:60px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Landing Content
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🤖 Tanzania Data Bot</h1>
        <p>Your assistant for exploring the 2022 Population & Housing Census of Tanzania.</p>
        <a href="/Chat" class="btn">💬 Start Chat</a>
    </div>

    <div class="features">
        <div class="feature-card">
            <h3>📊 Accurate Answers</h3>
            <p>Grounded in official 2022 Census data.</p>
        </div>
        <div class="feature-card">
            <h3>📈 Charts & Visuals</h3>
            <p>Generate graphs and comparisons easily.</p>
        </div>
        <div class="feature-card">
            <h3>💾 History Tracking</h3>
            <p>Revisit your previous sessions anytime.</p>
        </div>
        <div class="feature-card">
            <h3>⚡ Fast & Reliable</h3>
            <p>Smart RAG retrieval for quick results.</p>
        </div>
    </div>

    <div class="footer">✨ Created by <b>Chris Bwesa</b> & <b>Fedelika Maxmus</b></div>
    """,
    unsafe_allow_html=True,
)
