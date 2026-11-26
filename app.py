"""Standalone Streamlit application. It does not call or require FastAPI."""
from html import escape
from pathlib import Path
import streamlit as st
from rag_core import RAGEngine, build_corpus_from_files

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CORPUS_PATH = DATA_DIR / "corpus.csv"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- THEME -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Manrope:wght@400;500;600;700&family=Cairo:wght@400;600;700&display=swap');

:root {
    --bg: #0a0716;
    --panel: #130d29;
    --panel2: #1a1236;
    --violet: #8b5cf6;
    --violet-deep: #6d28d9;
    --pink: #ff4d8d;
    --text: #f1ecff;
    --muted: #9b92b8;
    --line: rgba(139,92,246,.30);
    --grad: linear-gradient(100deg, #8b5cf6 0%, #ff4d8d 100%);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Manrope', 'Cairo', sans-serif;
}

/* animated aurora behind everything */
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none;
    background:
        radial-gradient(40% 35% at 15% 10%, rgba(139,92,246,.30), transparent 70%),
        radial-gradient(35% 30% at 85% 15%, rgba(255,77,141,.20), transparent 70%),
        radial-gradient(45% 40% at 60% 95%, rgba(109,40,217,.28), transparent 70%);
    animation: drift 26s ease-in-out infinite alternate;
}
/* faint grid, fading toward the bottom */
[data-testid="stAppViewContainer"]::after {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(139,92,246,.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(139,92,246,.07) 1px, transparent 1px);
    background-size: 46px 46px;
    -webkit-mask-image: linear-gradient(180deg, #000 0%, transparent 70%);
            mask-image: linear-gradient(180deg, #000 0%, transparent 70%);
}
@keyframes drift {
    from { transform: translate3d(-2%, -1%, 0) scale(1); }
    to   { transform: translate3d(3%, 2%, 0) scale(1.08); }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

[data-testid="stHeader"] { background: transparent; }
.block-container { position: relative; z-index: 1; max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { font-family: 'Sora', 'Cairo', sans-serif; color: var(--text); }
p, label, li, span { color: inherit; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden;
    padding: 38px 42px 34px; margin: 0 0 26px;
    border-radius: 26px; border: 1px solid var(--line);
    background: linear-gradient(120deg, rgba(26,18,54,.94), rgba(12,8,28,.92));
    box-shadow: 0 0 60px rgba(139,92,246,.14), inset 0 0 40px rgba(139,92,246,.05);
}
.hero .orb {
    position: absolute; right: -70px; top: -90px; width: 340px; height: 340px; border-radius: 50%;
    background: conic-gradient(from 0deg, #8b5cf6, #ff4d8d, #6d28d9, #8b5cf6);
    filter: blur(52px); opacity: .55; animation: spin 18s linear infinite;
}
.hero .ring {
    position: absolute; right: 36px; top: 26px; width: 190px; height: 190px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,.22);
    box-shadow: 0 0 0 24px rgba(255,255,255,.035), 0 0 0 52px rgba(255,255,255,.02);
}
.hero-title {
    position: relative; font-family: 'Sora', sans-serif; font-weight: 800;
    font-size: clamp(32px, 4.6vw, 54px); letter-spacing: -1.5px; line-height: 1.05; margin: 0 0 12px;
    background: linear-gradient(100deg, #ffffff 10%, #d8c8ff 48%, #ff9cc0 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { position: relative; color: #bdb2dd; font-size: 15px; line-height: 1.7; max-width: 560px; margin: 0; }
.pill {
    position: relative; display: inline-flex; align-items: center; gap: 9px; margin-top: 20px;
    padding: 8px 14px; border-radius: 30px; font-size: 12px; font-weight: 600; color: #e6dcff;
    border: 1px solid var(--line); background: rgba(139,92,246,.12);
}
.pill i { width: 7px; height: 7px; border-radius: 50%; background: var(--pink); box-shadow: 0 0 12px var(--pink); animation: pulse 2.2s ease-in-out infinite; }

/* ---------- section headings ---------- */
.sec-title { font-family: 'Sora', sans-serif; font-size: 21px; font-weight: 700; margin: 28px 0 4px; color: var(--text); }
.sec-desc { color: var(--muted); font-size: 13px; margin: 0 0 16px; }
.tag { color: var(--pink); font-size: 12px; font-weight: 700; letter-spacing: .6px; margin-bottom: 8px; }

/* ---------- stat tiles ---------- */
.stat {
    padding: 16px 18px; border-radius: 16px; border: 1px solid var(--line);
    background: linear-gradient(145deg, rgba(26,18,54,.92), rgba(15,10,34,.92));
}
.stat .k { color: var(--muted); font-size: 12px; font-weight: 600; }
.stat .v { font-family: 'Sora', sans-serif; font-size: 26px; font-weight: 700; margin-top: 4px;
    background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }

/* ---------- answer card (gradient border) ---------- */
.answer {
    position: relative; padding: 22px 26px; border-radius: 18px; line-height: 1.85; color: var(--text);
    background: linear-gradient(var(--panel), var(--panel)) padding-box, var(--grad) border-box;
    border: 1px solid transparent;
    box-shadow: 0 0 40px rgba(139,92,246,.16);
}

/* ---------- agent plan timeline ---------- */
.plan { margin: 6px 0 0 6px; border-left: 2px solid var(--line); }
.plan-step { position: relative; padding: 0 0 14px 22px; color: #d9d0f3; font-size: 14px; line-height: 1.6; }
.plan-step:last-child { padding-bottom: 0; }
.plan-step::before {
    content: ""; position: absolute; left: -7px; top: 5px; width: 12px; height: 12px; border-radius: 50%;
    background: var(--violet); box-shadow: 0 0 0 3px rgba(139,92,246,.22), 0 0 14px var(--violet);
}
.plan-step:last-child::before { background: var(--pink); box-shadow: 0 0 0 3px rgba(255,77,141,.22), 0 0 14px var(--pink); }

/* ---------- chips and evidence ---------- */
.chip {
    display: inline-block; margin: 0 8px 8px 0; padding: 6px 11px; border-radius: 9px;
    border: 1px solid var(--line); background: rgba(139,92,246,.12);
    color: #d8c8ff; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px;
}
.chip.score { border-color: rgba(255,77,141,.4); background: rgba(255,77,141,.10); color: #ffc2d8; }
.source {
    padding: 16px 18px; margin: 10px 0; border-radius: 14px; line-height: 1.7; font-size: 14px;
    border: 1px solid var(--line); border-left: 3px solid var(--violet);
    background: rgba(19,13,41,.85); color: #e6dfff;
}
.source b { font-family: 'Sora', sans-serif; color: #fff; }
.source code { color: #ffc2d8; background: rgba(255,77,141,.10); padding: 2px 6px; border-radius: 6px; }

/* ---------- empty state ---------- */
.empty {
    padding: 26px 28px; border-radius: 18px; border: 1px dashed rgba(139,92,246,.5);
    background: rgba(19,13,41,.7);
}
.empty h3 { margin: 0 0 12px; font-size: 18px; }
.empty ol { margin: 0; padding-left: 20px; color: #cfc5ee; line-height: 2; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120b28 0%, #0b0719 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h2 { font-size: 18px; }

/* ---------- inputs ---------- */
div[data-testid="stTextArea"] label, div[data-testid="stFileUploader"] label, div[data-testid="stSlider"] label {
    color: #ddd3f7 !important; font-weight: 600 !important; font-size: 13px !important;
}
div[data-testid="stTextArea"] textarea {
    background: #0e0922 !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 14px !important;
    font-family: 'Manrope', 'Cairo', sans-serif; font-size: 15px;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 1px var(--violet), 0 0 22px rgba(139,92,246,.28) !important;
}
div[data-testid="stFileUploader"] section {
    background: rgba(14,9,34,.8) !important; border: 1px dashed rgba(139,92,246,.5) !important; border-radius: 14px !important;
}
div[data-testid="stFileUploader"] section:hover { border-color: var(--pink) !important; }
div[data-testid="stFileUploader"] section * { color: #cfc5ee !important; }
div[data-testid="stFileUploader"] button {
    color: #fff !important; border: 1px solid var(--line) !important;
    background: rgba(139,92,246,.2) !important; border-radius: 9px !important;
}
div[data-testid="stSlider"] [role="slider"] { background: var(--pink) !important; box-shadow: 0 0 14px var(--pink); }
div[data-testid="stSlider"] [data-testid="stTickBarMin"], div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSliderThumbValue"] { color: var(--muted) !important; }

.stButton > button {
    width: 100%; min-height: 50px; border: none !important; border-radius: 13px !important;
    color: #fff !important; background: var(--grad) !important;
    font-family: 'Sora', sans-serif !important; font-weight: 700 !important; letter-spacing: .4px;
    box-shadow: 0 0 22px rgba(139,92,246,.40), 0 6px 24px rgba(255,77,141,.18);
    transition: transform .2s ease, box-shadow .2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px); color: #fff !important;
    box-shadow: 0 0 30px rgba(139,92,246,.6), 0 10px 34px rgba(255,77,141,.3);
}
.stButton > button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }

/* ---------- misc widgets ---------- */
div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: 14px !important; background: rgba(19,13,41,.7) !important;
}
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * { color: #ece5ff !important; font-weight: 600; }
div[data-testid="stAlert"] { border-radius: 13px !important; background: rgba(139,92,246,.12) !important; border: 1px solid var(--line) !important; }
div[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stSpinner"] * { color: var(--muted) !important; }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
hr { border-color: var(--line) !important; }
.foot { text-align: center; color: #6f6590; font-size: 12px; padding-top: 28px; margin-top: 36px; border-top: 1px solid rgba(139,92,246,.16); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>

<div class="hero">
  <div class="orb"></div><div class="ring"></div>
  <div class="hero-title">Agentic RAG Assistant</div>
  <p class="hero-sub">Ask a question about your documents. The agent plans its search, pulls the best evidence, and shows you exactly which chunks the answer came from.</p>
  <div class="pill"><i></i> Answers backed by your sources</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------- HELPERS -----------------------------
def safe(text):
    """Escape text for HTML and keep line breaks."""
    return escape(str(text)).replace("\n", "<br>")


def get_engine():
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine.from_csv(CORPUS_PATH)
    return st.session_state.rag_engine


# ----------------------------- SIDEBAR -----------------------------
engine = None
engine_error = None

with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    if st.button("Build knowledge base", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one document first.")
        else:
            saved_paths = []
            for uploaded in uploads:
                destination = UPLOAD_DIR / Path(uploaded.name).name
                destination.write_bytes(uploaded.getbuffer())
                saved_paths.append(destination)
            try:
                frame = build_corpus_from_files(saved_paths, CORPUS_PATH)
                st.session_state.pop("rag_engine", None)
                st.success(f"Knowledge base created: {len(frame)} chunks.")
            except Exception as error:
                st.error(str(error))

    st.divider()
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)

    if CORPUS_PATH.exists():
        try:
            engine = get_engine()
            st.caption(f"Loaded: {len(engine.corpus)} chunks")
        except Exception as error:
            engine_error = error

# ----------------------------- EMPTY / ERROR STATES -----------------------------
if not CORPUS_PATH.exists():
    st.markdown("""
    <div class="empty">
      <h3>Start by adding your documents</h3>
      <ol>
        <li>Upload PDF, DOCX, TXT or Markdown files in the sidebar.</li>
        <li>Press <b>Build knowledge base</b>.</li>
        <li>Ask your question here.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if engine_error is not None:
    st.error(f"Could not load the knowledge base: {engine_error}")
    st.stop()

# ----------------------------- STATS -----------------------------
corpus = engine.corpus
doc_count = corpus["doc_id"].nunique() if "doc_id" in corpus.columns else None

stats = [("Chunks indexed", f"{len(corpus):,}")]
if doc_count is not None:
    stats.append(("Documents", f"{doc_count:,}"))
stats.append(("Chunks per answer", str(top_k)))

for col, (label, value) in zip(st.columns(len(stats)), stats):
    with col:
        st.markdown(
            f'<div class="stat"><div class="k">{label}</div><div class="v">{value}</div></div>',
            unsafe_allow_html=True
        )

# ----------------------------- ASK -----------------------------
st.markdown(
    '<div class="sec-title">Ask your documents</div>'
    '<div class="sec-desc">Write a question in plain language. Arabic and English both work.</div>',
    unsafe_allow_html=True
)
question = st.text_area(
    "Your question",
    placeholder="What does the report say about...?",
    height=110
)

if st.button("Ask the agent", type="primary"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Searching your documents..."):
            result = engine.ask(question, top_k=top_k)

        st.markdown('<div class="sec-title">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer" dir="auto">{safe(result["answer"])}</div>', unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="sec-title" style="font-size:17px;">How the agent got here</div>', unsafe_allow_html=True)
            steps_html = "".join(f'<div class="plan-step">{safe(step)}</div>' for step in result["plan"])
            st.markdown(f'<div class="plan">{steps_html}</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="sec-title" style="font-size:17px;">Sources cited</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="chip">{safe(cid)}</span>' for cid in result["cited_chunk_ids"])
            st.markdown(chips or '<span class="sec-desc">No sources were cited.</span>', unsafe_allow_html=True)

        with st.expander("Retrieved evidence", expanded=True):
            for _, row in result["retrieved_chunks"].iterrows():
                score_chip = ""
                if "score" in row.index:
                    try:
                        score_chip = f' <span class="chip score">match {float(row["score"]):.3f}</span>'
                    except (TypeError, ValueError):
                        pass
                st.markdown(
                    f'<div class="source" dir="auto"><b>{safe(row["title"])}</b> '
                    f'<code>{safe(row["chunk_id"])}</code>{score_chip}<br>{safe(row["chunk_text"])}</div>',
                    unsafe_allow_html=True
                )

st.markdown('<div class="foot">Agentic RAG Assistant · runs standalone, FastAPI is optional</div>', unsafe_allow_html=True)
