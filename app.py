"""
app.py - RSET Academic Assistant web interface
Powered by Groq LLaMA 3 (free, fast) + local embeddings
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_rag_chain, ask, format_sources

load_dotenv()

st.set_page_config(
    page_title="RSET Academic Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .user-msg {
        background: #8B0000; color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px; margin: 8px 0;
        max-width: 80%; margin-left: auto; font-size: 0.95rem;
    }
    .bot-msg {
        background: white; color: #1a1a1a;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px; margin: 8px 0;
        max-width: 85%; border: 1px solid #e0e0e0;
        font-size: 0.95rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .source-box {
        background: #fff8e1; border-left: 3px solid #f9a825;
        border-radius: 4px; padding: 8px 12px;
        margin-top: 6px; font-size: 0.82rem; color: #5d4037;
    }
    .main-header { text-align: center; padding: 1.5rem 0 1rem 0; }
    .main-header h1 { color: #8B0000; font-size: 2rem; margin-bottom: 4px; }
    .main-header p { color: #666; font-size: 0.95rem; }
    .stButton > button {
        background-color: #8B0000; color: white;
        border-radius: 25px; border: none;
        padding: 10px 28px; font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background-color: #6d0000; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 RSET Academic Assistant")
    st.markdown("*Powered by Groq LLaMA 3 — Free & Fast!*")
    st.markdown("---")

    if not os.getenv("GROQ_API_KEY"):
        st.markdown("**🔑 Enter your Groq API Key:**")
        st.markdown("Get it free at [console.groq.com](https://console.groq.com)")
        api_key = st.text_input("API Key", type="password", placeholder="gsk_...")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        st.success("✅ Groq API Key loaded!")

    st.markdown("---")
    st.markdown("**📚 Loaded Documents:**")
    st.markdown("📘 Academic Handbook 2025-26")
    st.markdown("📗 B.Tech Regulations 2023")
    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    for q in [
        "What is the minimum attendance?",
        "How is CGPA calculated?",
        "What happens with an FE grade?",
        "What are pass criteria for practicals?",
        "Can I apply for Honours?",
        "What are grace marks rules?",
        "How many credits to graduate?",
        "Explain the internship requirements.",
    ]:
        st.markdown(f"• *{q}*")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages  = []
        st.session_state.history   = []
        st.rerun()
    st.markdown(
        "<div style='font-size:0.78rem;color:#999;text-align:center'>"
        "Answers based on official RSET documents only.<br>"
        "Verify critical decisions with the Academic Office."
        "</div>", unsafe_allow_html=True
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🎓 RSET Academic Assistant</h1>
    <p>Ask anything about your college regulations, attendance, grades, exams & more.</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []   # list of (question, answer) tuples
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None

@st.cache_resource(show_spinner=False)
def get_chain():
    return build_rag_chain()   # returns (retriever, chain)

if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to get started.")
    st.stop()

if st.session_state.chain is None:
    with st.spinner("🔄 Loading knowledge base (local model)..."):
        try:
            retriever, chain = get_chain()
            st.session_state.retriever = retriever
            st.session_state.chain     = chain
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.info("Run `python ingest.py` first, then refresh this page.")
            st.stop()
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧑‍🎓 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
        if msg.get("sources"):
            st.markdown(f"<div class='source-box'>📚 <b>Sources:</b><br>{msg['sources']}</div>", unsafe_allow_html=True)

# Input
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "question",
        placeholder="e.g. What is the minimum attendance to sit for exams?",
        label_visibility="collapsed",
        key="user_input"
    )
with col2:
    send_btn = st.button("Ask →", use_container_width=True)

if (send_btn or user_input) and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f"<div class='user-msg'>🧑‍🎓 {question}</div>", unsafe_allow_html=True)

    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            result       = ask(
                st.session_state.retriever,
                st.session_state.chain,
                question,
                st.session_state.history
            )
            answer       = result["answer"]
            sources      = format_sources(result.get("source_documents", []))
            sources_html = sources.replace("\n", "<br>")

            st.markdown(f"<div class='bot-msg'>🤖 {answer}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='source-box'>📚 <b>Sources:</b><br>{sources_html}</div>", unsafe_allow_html=True)

            # Save to history for follow-up questions
            st.session_state.history.append((question, answer))
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources_html
            })
        except Exception as e:
            err = f"Sorry, something went wrong: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
