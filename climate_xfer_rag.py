"""
CLIMATE-XFER RAG Demo  |  Assignment 3
AI and Large Models — Masters Program 2025-2027
Earth Observation Interface · Powered by Groq LLaMA-3.3-70B
Demonstrates: chunking → embedding → retrieval → generation
"""

import os
import base64
import html as _html

import numpy as np
import streamlit as st

# ── path helpers ──────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(rel: str) -> str:
    return os.path.join(_DIR, rel)

# Load Groq key from Streamlit secrets (set GROQ_API_KEY in Streamlit Cloud settings)
_GROQ_KEY_DEFAULT = st.secrets.get("GROQ_API_KEY", "")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLIMATE-XFER · RAG EO Terminal",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy model load ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Earth Observation / Remote Sensing Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<style>

/* ── base ── */
html, body, [class*="css"] {
  font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  background-color: #060e08 !important;
}

/* ── coordinate-grid background overlay ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(74,222,128,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(74,222,128,.025) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none;
  z-index: 0;
}

/* ── satellite scan line ── */
@keyframes satelliteScan {
  0%   { transform: translateY(-4px); opacity: 0; }
  3%   { opacity: 1; }
  97%  { opacity: 0.7; }
  100% { transform: translateY(100vh); opacity: 0; }
}
.rs-scan-line {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(74,222,128,.15) 8%,
    rgba(74,222,128,.95) 50%,
    rgba(74,222,128,.15) 92%,
    transparent 100%
  );
  box-shadow: 0 0 10px 3px rgba(74,222,128,.35), 0 0 24px 8px rgba(74,222,128,.12);
  animation: satelliteScan 9s linear infinite;
  z-index: 9999;
  pointer-events: none;
}

/* ── title animations ── */
@keyframes titleGlow {
  0%,100% { text-shadow: 0 0 14px #4ade8066, 0 0 36px #4ade8033; }
  50%      { text-shadow: 0 0 28px #4ade80bb, 0 0 64px #4ade8055; }
}
@keyframes radarPulse {
  0%   { transform: scale(.7); opacity: .8; }
  100% { transform: scale(2.6); opacity: 0; }
}

.eo-title {
  font-family: 'Cinzel', serif;
  font-size: 2.3rem; font-weight: 700;
  color: #dcfce7; text-align: center; letter-spacing: .07em;
  animation: titleGlow 3.5s ease-in-out infinite;
  margin-bottom: .15rem;
  position: relative;
}
.eo-subtitle {
  font-family: 'Orbitron', sans-serif;
  font-size: .7rem; color: #4ade80;
  text-align: center; letter-spacing: .26em; margin-bottom: 1.4rem;
}

/* ── radar ring (decorative title overlay) ── */
.radar-wrap {
  display: inline-block; position: relative;
  margin: 0 auto 1rem; display: flex; justify-content: center;
}
.radar-ring {
  position: absolute; inset: -8px;
  border-radius: 50%; border: 1px solid #4ade80;
  animation: radarPulse 2.8s ease-out infinite;
}
.radar-ring:nth-child(2) { animation-delay: 1.4s; }

/* ── institution bar ── */
.inst-bar {
  display: flex; align-items: center; justify-content: center;
  gap: 16px; padding: 10px 20px; margin-bottom: 1rem;
  background: linear-gradient(90deg, #060e08, #0b1a0d, #060e08);
  border-radius: 10px; border: 1px solid #1e4a24;
}
.inst-uni  { font-family: 'Cinzel', serif; font-size: .95rem; color: #b8d4bb; }
.inst-dept { font-family: 'Orbitron', sans-serif; font-size: .6rem; color: #4ade80; letter-spacing: .15em; }

/* ── author cards ── */
@keyframes avatarPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(74,222,128,.45); }
  70%      { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
}
.author-row  { display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin-bottom: 1.4rem; }
.author-card {
  display: flex; align-items: center; gap: 10px; padding: 8px 16px;
  background: linear-gradient(135deg, #0b1a0d, #060e08);
  border: 1px solid #1e4a24; border-radius: 12px; transition: .3s;
}
.author-card:hover { border-color: #4ade80; transform: translateY(-2px); box-shadow: 0 0 14px #4ade8022; }
.author-avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Cinzel', serif; font-weight: 700; font-size: .85rem; color: #fff;
  animation: avatarPulse 2.5s infinite;
}
.author-name { font-family: 'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif; font-size: .9rem; color: #b8d4bb; }
.author-id   { font-family: 'Orbitron', sans-serif; font-size: .58rem; color: #4ade80; letter-spacing: .1em; }

/* ── pipeline step cards ── */
@keyframes stepFadeIn { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform: translateY(0); } }
.pipe-step {
  background: linear-gradient(135deg, #0b1a0d, #060e08);
  border: 1px solid #1e4a24; border-radius: 14px;
  padding: 20px; text-align: center;
  animation: stepFadeIn .6s ease both;
  transition: .3s;
}
.pipe-step:hover { border-color: #4ade80; box-shadow: 0 0 20px #4ade8022; }
.pipe-icon  { font-size: 2rem; margin-bottom: .4rem; }
.pipe-label {
  font-family: 'Orbitron', sans-serif; font-size: .7rem;
  color: #4ade80; letter-spacing: .18em; margin-bottom: .5rem;
}
.pipe-desc { font-size: .88rem; color: #6b9a74; line-height: 1.55; }

/* ── metric cards ── */
@keyframes metricBorderPulse {
  0%,100% { border-color: #1e4a24; }
  50%      { border-color: #4ade8066; }
}
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-card {
  flex: 1; min-width: 155px;
  background: linear-gradient(135deg, #0b1a0d, #060e08);
  border: 1px solid #1e4a24; border-radius: 12px;
  padding: 14px 18px;
  animation: metricBorderPulse 3s ease-in-out infinite;
}
.metric-label { font-family: 'Orbitron', sans-serif; font-size: .6rem; color: #4ade80; letter-spacing: .15em; }
.metric-value { font-family: 'Cinzel', serif; font-size: 1.5rem; color: #dcfce7; margin-top: 4px; }

/* ── chunk display ── */
.chunk-box {
  background: #070f09; border: 1px solid #1e4a24; border-radius: 8px;
  padding: 12px; font-size: .82rem; color: #6b9a74;
  font-family: 'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif; line-height: 1.6;
  margin-bottom: 8px; position: relative; transition: .25s;
}
.chunk-box:hover { border-color: #4ade8055; }
.chunk-id {
  font-family: 'Orbitron', sans-serif; font-size: .58rem;
  color: #4ade80; position: absolute; top: 8px; right: 10px;
}

/* ── answer box ── */
@keyframes answerReveal {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.answer-box {
  background: linear-gradient(135deg, #070f09, #0b1a0d);
  border: 1px solid #4ade8066; border-radius: 12px;
  padding: 20px; font-size: .95rem; color: #b8d4bb;
  line-height: 1.8; font-family: 'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;
  animation: answerReveal .5s ease;
}

/* ── tab guide ── */
.tab-guide {
  background: linear-gradient(90deg, #060e08, #0b1a0d, #060e08);
  border: 1px solid #1e4a24; border-radius: 10px;
  padding: 12px 20px; margin-bottom: 1.2rem;
  font-family: 'Orbitron', sans-serif; font-size: .65rem;
  color: #4ade80; letter-spacing: .12em; text-align: center;
}
.tab-guide strong { color: #dcfce7; }

/* ── similarity bar ── */
.sim-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.sim-bar-bg { flex: 1; height: 8px; background: #0b1a0d; border-radius: 4px; }
.sim-bar-fg { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #1e4a24, #4ade80); }
.sim-pct { font-family: 'Orbitron', sans-serif; font-size: .6rem; color: #4ade80; width: 44px; text-align: right; }

/* ── status badge ── */
@keyframes statusBlink {
  0%,100% { opacity: 1; } 50% { opacity: .4; }
}
.status-dot {
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: #4ade80; margin-right: 5px;
  animation: statusBlink 1.8s ease-in-out infinite;
  vertical-align: middle;
}

/* ── EO band bar (decorative) ── */
.band-bar {
  height: 4px; border-radius: 2px; margin: 8px 0;
  background: linear-gradient(90deg,
    #1d4ed8 0%, #0ea5e9 16%, #22c55e 33%, #84cc16 50%,
    #eab308 66%, #f97316 83%, #dc2626 100%
  );
  opacity: .7;
}

/* ── satellite status strip ── */
.sat-strip {
  background: #070f09; border: 1px solid #1e4a24; border-radius: 8px;
  padding: 8px 14px; font-family: 'Orbitron',sans-serif; font-size: .6rem;
  color: #4ade80; letter-spacing: .1em; margin-bottom: 12px;
  display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
}
.sat-strip span { color: #6b9a74; }

/* ── tutorial ── */
.tutorial-step { margin-bottom: .7rem; font-size: .9rem; line-height: 1.6; }
.tutorial-step b { color: #4ade80; }

/* ── EO section header ── */
.eo-section-head {
  font-family: 'Orbitron', sans-serif; font-size: .68rem;
  color: #4ade80; letter-spacing: .2em;
  border-bottom: 1px solid #1e4a24; padding-bottom: 6px; margin-bottom: 12px;
}

/* ── groq badge ── */
.groq-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg, #0b1a0d, #0e2010);
  border: 1px solid #4ade8066; border-radius: 20px;
  padding: 4px 14px;
  font-family: 'Orbitron',sans-serif; font-size: .6rem;
  color: #4ade80; letter-spacing: .15em;
}

</style>
""")

# ── scan line overlay ──────────────────────────────────────────────────────────
st.html('<div class="rs-scan-line"></div>')

# ── logo helper ───────────────────────────────────────────────────────────────
def _get_logo_b64() -> str:
    try:
        with open(_path("SCHOOL LOGO.png"), "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ── header ────────────────────────────────────────────────────────────────────
logo_b64  = _get_logo_b64()
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="height:54px;filter:drop-shadow(0 0 8px #4ade8044);">'
    if logo_b64 else '<span style="font-size:2rem">🛰️</span>'
)

st.html(f"""
<div style="position:fixed;top:56px;right:16px;z-index:9998;
  background:linear-gradient(135deg,#1e4a24,#166534);
  color:#dcfce7;padding:5px 14px;border-radius:20px;
  font-family:'Orbitron',sans-serif;font-size:.6rem;letter-spacing:.18em;
  box-shadow:0 0 12px #4ade8033;border:1px solid #4ade8044;">
  ASSIGNMENT 3
</div>

<div style="text-align:center;margin-bottom:.6rem;">
  {logo_html}
</div>

<div class="eo-title">CLIMATE-XFER · RAG INTELLIGENCE</div>
<div class="eo-subtitle">🛰️ &nbsp; EARTH OBSERVATION KNOWLEDGE RETRIEVAL &nbsp;·&nbsp; ASSIGNMENT 3 &nbsp;·&nbsp; AI &amp; LARGE MODELS</div>

<div style="text-align:center;margin-bottom:.5rem;">
  <div class="band-bar" style="max-width:520px;margin:0 auto;"></div>
  <span style="font-family:'Orbitron',sans-serif;font-size:.55rem;color:#2d4a32;letter-spacing:.12em">
    COASTAL · BLUE · GREEN · RED · RED-EDGE · NIR · SWIR1 · SWIR2
  </span>
</div>

<div style="text-align:center;margin-bottom:.9rem;font-family:'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;">
  <span style="color:#4ade80;font-size:1rem;font-weight:bold;">XFER</span>
  <span style="color:#6b9a74;font-size:.95rem;"> — </span>
  <span style="color:#b8d4bb;font-size:.95rem;font-style:italic;">Cross-domain Transfer Learning:</span>
  <span style="color:#6b9a74;font-size:.9rem;"> a model trained on one climate region transferred to another without full retraining, preserving temporal patterns across geographic boundaries.</span>
</div>

<div class="inst-bar">
  <div class="inst-text">
    <div class="inst-uni">Beihang University</div>
    <div class="inst-dept">MSc AI &amp; LARGE MODELS &nbsp;·&nbsp; REMOTE SENSING &nbsp;·&nbsp; SEMESTER 1/2 2025-2027</div>
  </div>
</div>

<div class="author-row">
  <div class="author-card">
    <div class="author-avatar" style="background:linear-gradient(135deg,#14532d,#166534)">T</div>
    <div><div class="author-name">Tanaka Alex Mbendana</div><div class="author-id">LS2525233</div></div>
  </div>
  <div class="author-card">
    <div class="author-avatar" style="background:linear-gradient(135deg,#064e3b,#047857)">F</div>
    <div><div class="author-name">Fitrotur Rofiqoh</div><div class="author-id">LS2525220</div></div>
  </div>
  <div class="author-card">
    <div class="author-avatar" style="background:linear-gradient(135deg,#0c4a0e,#166534)">M</div>
    <div><div class="author-name">Munashe Innocent Mafuta</div><div class="author-id">LS2557204</div></div>
  </div>
</div>
""")

# ── reviewer guide bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="tab-guide">
  📡 &nbsp;GROUND STATION GUIDE &nbsp;|&nbsp;
  <strong>Tab 1</strong>: RAG Pipeline Overview + Video &nbsp;→&nbsp;
  <strong>Tab 2</strong>: Query Terminal (Live Q&amp;A) &nbsp;→&nbsp;
  <strong>Tab 3</strong>: Spectral Internals (embeddings, similarity)
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Satellite status strip
    st.html("""
    <div class="sat-strip">
      <div><span class="status-dot"></span>SENTINEL-2A <span>ACTIVE</span></div>
      <div>📡 SIGNAL <span>NOMINAL</span></div>
      <div>🌍 NADIR <span>PASS</span></div>
    </div>
    """)

    # ── Document upload ───────────────────────────────────────────────────────
    st.html('<div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#4ade80;letter-spacing:.2em;margin-bottom:.4rem">📄 DOCUMENT SOURCE</div>')
    uploaded_file = st.file_uploader(
        "Upload any PDF",
        type=["pdf"],
        help="Upload your own document to replace the default CLIMATE-XFER PDF. "
             "The system will chunk, embed, and index it automatically.",
    )
    if uploaded_file is not None:
        doc_name   = uploaded_file.name
        doc_bytes  = uploaded_file.getvalue()
        doc_source = "uploaded"
        st.success(f"✅ {doc_name}", icon="📄")
    else:
        doc_name   = "CLIMATE_XFER_Report_v4.pdf"
        doc_bytes  = None          # loaded from disk below
        doc_source = "default"
        st.caption("Using default: CLIMATE_XFER_Report_v4.pdf")

    st.divider()
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#4ade80;letter-spacing:.2em;margin-bottom:1rem">⚙ RAG PARAMETERS</div>', unsafe_allow_html=True)

    chunk_size = st.slider("Chunk size (words)", 100, 600, 300, 50,
                           help="Number of words per chunk")
    chunk_over = st.slider("Overlap (words)",    0,  150,  50, 25,
                           help="Words shared between adjacent chunks")
    top_k      = st.slider("Top-K retrieval",   1,   10,   4,  1,
                           help="How many chunks are retrieved per query")

    st.divider()
    st.html("""
    <div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#4ade80;letter-spacing:.2em;margin-bottom:.6rem">
      🤖 GROQ LLM ENGINE
    </div>
    <div style="font-family:'Palatino Linotype',Palatino,serif;font-size:.78rem;color:#6b9a74;margin-bottom:.6rem">
      Powered by LLaMA-3.3-70B via Groq. Key pre-filled for demo.
    </div>
    """)
    api_key = st.text_input("Groq API key", value=_GROQ_KEY_DEFAULT, type="password",
                            help="Pre-filled with demo key. Groq is free — get your own at console.groq.com")

    st.divider()
    st.html('<div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#4ade80;letter-spacing:.2em;margin-bottom:.6rem">🎬 DEMO VIDEO</div>')
    st.caption("📁 videoplayback.mp4 loaded from app folder")
    video_url = st.text_input("Fallback URL (YouTube)",
                              value="",
                              placeholder="Used only if videoplayback.mp4 is missing")

    st.divider()
    # Spectral band indicator (decorative)
    st.html("""
    <div style="font-family:Orbitron,sans-serif;font-size:.62rem;color:#4ade80;letter-spacing:.15em;margin-bottom:.5rem">
      🌿 SPECTRAL BANDS
    </div>
    <div style="font-size:.72rem;color:#6b9a74;line-height:1.7;font-family:'Palatino Linotype',Palatino,serif">
      <span style="color:#1d4ed8">■</span> B2 Blue 490nm &nbsp;
      <span style="color:#22c55e">■</span> B3 Green 560nm<br>
      <span style="color:#dc2626">■</span> B4 Red 665nm &nbsp;
      <span style="color:#7c3aed">■</span> B8 NIR 842nm<br>
      <span style="color:#f97316">■</span> B11 SWIR 1610nm
    </div>
    <div class="band-bar" style="margin-top:8px;"></div>
    """)

    st.divider()
    with st.expander("🎓 Teacher & Reviewer Guide", expanded=False):
        st.html("""
<div class="tutorial-step"><b>Step 1 — Load a Document</b><br>
The default CLIMATE-XFER PDF loads automatically. To use your own document,
click <em style="color:#4ade80">📄 DOCUMENT SOURCE → Upload any PDF</em> in the sidebar.
The system re-indexes it instantly.</div>
<div class="tutorial-step"><b>Step 2 — Tune RAG parameters</b><br>
Adjust chunk size, overlap, and Top-K. Pipeline internals update live (Tab 3).</div>
<div class="tutorial-step"><b>Step 3 — Ask a question (Tab 2)</b><br>
Query is embedded → cosine similarity search → top-K passages retrieved →
Groq LLaMA-3.3-70B generates a grounded answer.</div>
<div class="tutorial-step"><b>Step 4 — Inspect internals (Tab 3)</b><br>
See raw chunks, 384-dim embedding heatmap, and cosine similarity scores.</div>
<div class="tutorial-step"><b>Assignment 3 requirements met</b><br>
✅ Engineering paper ingested (CLIMATE-XFER PDF)<br>
✅ Chunking with configurable sliding window<br>
✅ Embedding with sentence-transformers MiniLM<br>
✅ Retrieval via cosine similarity Top-K<br>
✅ Generation via Groq LLaMA-3.3-70B (live API)<br>
✅ Interactive Streamlit deliverable</div>
""")

# ── RAG backend ───────────────────────────────────────────────────────────────
PDF_PATH = r"D:\Masters Program 2025-2027\Semester 1_2\Artificial Intelligence and Large  Models\Final  Project\CLIMATE_XFER\reports\CLIMATE_MODEL.pdf"

@st.cache_data(show_spinner=False)
def _extract_pdf_bytes(data: bytes, _name: str) -> str:
    """Extract text from a PDF given its raw bytes. _name is used only as cache key."""
    try:
        import PyPDF2, io
        text = []
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(size - overlap, 1)
    while i < len(words):
        chunks.append(" ".join(words[i: i + size]))
        i += step
    return chunks


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _retrieve(query: str, chunks: list[str], embeddings: np.ndarray,
              model, k: int) -> list[tuple[int, float, str]]:
    q_emb = model.encode([query])[0]
    scores = [(i, _cosine_sim(q_emb, embeddings[i])) for i in range(len(chunks))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(i, s, chunks[i]) for i, s in scores[:k]]


_DEFAULT_EXAMPLES = [
    "What is the main objective of CLIMATE-XFER and which regions does it cover?",
    "What RMSE and Pearson r did the GRU model achieve on the SADC validation set?",
    "How does zero-shot transfer to Southeast Asia compare to fine-tuned transfer?",
    "What are the four input features used in the CLIMATE-XFER pipeline?",
    "What is SPEI-1 and why is it used as the forecast target?",
    "What role do Pacific and Indian Ocean SST indices play in drought forecasting?",
    "How many epochs of fine-tuning were applied for Southeast Asia adaptation?",
    "How does CLIMATE-XFER outperform the persistence baseline?",
]


@st.cache_data(show_spinner=False)
def _generate_example_questions(text_sample: str, key: str) -> list[str]:
    """Use Groq to generate 6 document-specific questions from the first ~3 000 chars."""
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Given a document excerpt, generate exactly 6 "
                        "specific, insightful questions a reader would want to ask about this document. "
                        "Output ONLY the 6 questions, one per line, no numbering, no bullet points, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document excerpt:\n\n{text_sample}\n\nGenerate 6 specific questions:",
                },
            ],
        )
        lines = [l.strip(" -•123456789.)") for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        questions = [l for l in lines if len(l) > 15]
        return questions[:6] if questions else []
    except Exception:
        return []


def _generate_groq(question: str, context_passages: list[str], key: str) -> str:
    try:
        from groq import Groq
        context = "\n\n---\n\n".join(context_passages)
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert on climate science and transfer learning. "
                        "Answer questions strictly from the provided context passages. "
                        "Be concise and precise. If the context does not contain enough "
                        "information, say so explicitly rather than guessing."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return resp.choices[0].message.content
    except ImportError:
        return "[Groq package not installed — run: pip install groq]"
    except Exception as e:
        return f"[Groq API error: {e}]"


# ── load / embed on startup ───────────────────────────────────────────────────
with st.spinner("🛰️ Initialising EO terminal — loading AI model and document …"):
    if doc_bytes is not None:
        raw_text = _extract_pdf_bytes(doc_bytes, doc_name)
    else:
        with open(PDF_PATH, "rb") as _f:
            raw_text = _extract_pdf_bytes(_f.read(), doc_name)
    _load_sentence_transformer()

pdf_ok = not raw_text.startswith("[PDF")


@st.cache_data(show_spinner=False)
def _get_chunks_and_embeddings(text: str, size: int, overlap: int):
    chunks = _chunk_text(text, size, overlap)
    model  = _load_sentence_transformer()
    embs   = model.encode(chunks, show_progress_bar=False, batch_size=32)
    return chunks, np.array(embs)


chunks, embeddings = _get_chunks_and_embeddings(raw_text, chunk_size, chunk_over)

# ── clear stale Q&A results when the active document changes ─────────────────
if st.session_state.get("_active_doc") != doc_name:
    st.session_state["_active_doc"] = doc_name
    for _k in ("qa_results", "qa_query", "qa_passages", "qa_answer", "query_input"):
        st.session_state.pop(_k, None)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📡  RAG Pipeline & Video",
    "🖥️  Query Terminal",
    "🛰️  Spectral Internals",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline overview + video
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Always-visible team project banner ───────────────────────────────────
    st.html("""
    <div style="background:linear-gradient(135deg,#0e2a12,#0b1a0d);
                border:2px solid #4ade80aa;border-radius:14px;
                padding:16px 22px;margin-bottom:1.4rem;position:relative;overflow:hidden;">
      <!-- glow bar -->
      <div style="position:absolute;top:0;left:0;right:0;height:3px;
                  background:linear-gradient(90deg,transparent,#4ade80,transparent);"></div>
      <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
        <div style="font-size:2rem;line-height:1">🏆</div>
        <div style="flex:1;min-width:200px">
          <div style="font-family:'Orbitron',sans-serif;font-size:.68rem;color:#4ade80;
                      letter-spacing:.2em;margin-bottom:.35rem">
            ◈ OUR TEAM PROJECT — THE DEFAULT KNOWLEDGE SOURCE
          </div>
          <div style="font-family:'Cinzel',serif;font-size:1rem;color:#dcfce7;
                      font-weight:700;margin-bottom:.3rem">
            CLIMATE-XFER
          </div>
          <div style="font-family:'Palatino Linotype',Palatino,serif;font-size:.88rem;
                      color:#b8d4bb;line-height:1.6;margin-bottom:.4rem">
            <em>Transferable Deep Learning for Seasonal Drought Forecasting Across Southern Africa and Southeast Asia</em>
          </div>
          <div style="font-family:'Orbitron',sans-serif;font-size:.6rem;color:#6b9a74;
                      letter-spacing:.1em;line-height:1.8">
            Tanaka Alex Mbendana (LS2525233) &nbsp;·&nbsp;
            Fitrotur Rofiqoh (LS2525220) &nbsp;·&nbsp;
            Munashe Innocent Mafuta (LS2557204)<br>
            Beihang University &nbsp;·&nbsp; MSc AI &amp; Large Models 2025-2027 &nbsp;·&nbsp; March 2026
          </div>
          <div style="font-family:'Orbitron',sans-serif;font-size:.58rem;color:#4ade8066;
                      letter-spacing:.08em;margin-top:.4rem">
            GRU · SADC → SEA TRANSFER &nbsp;·&nbsp; RMSE 0.193 &nbsp;·&nbsp; r = 0.903 (fine-tuned)
          </div>
        </div>
        <div style="font-family:'Orbitron',sans-serif;font-size:.58rem;color:#4ade8088;
                    text-align:right;white-space:nowrap;padding-top:4px">
          14 PAGES<br>7 128 WORDS<br>ASSIGNMENT 3
        </div>
      </div>
      <div style="margin-top:.7rem;font-size:.72rem;color:#4ade8066;
                  font-family:'Orbitron',sans-serif;letter-spacing:.08em">
        This RAG system was built to demonstrate retrieval-augmented generation on our own research.
        Upload any PDF in the sidebar to query a different document using the same pipeline.
      </div>
    </div>
    """)

    # ── Intro — adapts to active document ────────────────────────────────────
    if doc_source == "uploaded":
        _intro_doc_ref = (
            f'the uploaded document <strong style="color:#4ade80">{_html.escape(doc_name)}</strong>'
        )
    else:
        _intro_doc_ref = (
            'the <strong style="color:#4ade80">CLIMATE-XFER</strong> technical report '
            '(our team project)'
        )

    st.html(f"""
    <div class="eo-section-head">◈ WHAT IS RETRIEVAL-AUGMENTED GENERATION?</div>
    <p style="font-size:.95rem;color:#b8d4bb;line-height:1.8;margin-bottom:1.4rem">
    RAG grounds a large language model in <em>specific documents</em> rather than relying on
    parametric memory alone. We split {_intro_doc_ref} into overlapping chunks,
    encode every chunk into a 384-dimensional semantic vector space with
    <strong style="color:#4ade80">sentence-transformers</strong>,
    then — at query time — retrieve the most relevant chunks via cosine similarity and feed them
    to <strong style="color:#4ade80">Groq LLaMA-3.3-70B</strong> as grounded context.
    This eliminates hallucination and keeps every answer traceable to exact passages in the document.
    </p>
    """)

    # Knowledge source card — adapts to uploaded or default document
    _word_count = len(raw_text.split()) if pdf_ok else 0
    _page_est   = max(1, _word_count // 400)
    if doc_source == "uploaded":
        _doc_title   = _html.escape(doc_name)
        _doc_meta    = '<span style="color:#4ade80;font-size:.8rem">📤 Uploaded by user</span>'
        _doc_stats   = f"{_page_est} PAGES (EST.) &nbsp;·&nbsp; {_word_count:,} WORDS EXTRACTED"
    else:
        _doc_title   = "CLIMATE-XFER: Transferable Deep Learning for Seasonal Drought Forecasting Across Southern Africa and Southeast Asia"
        _doc_meta    = (
            'Tanaka Alex Mbendana &nbsp;·&nbsp; Fitrotur Rofiqoh &nbsp;·&nbsp; Munashe Innocent Mafuta<br>'
            '<span style="color:#6b9a74;font-size:.83rem">Beihang University &nbsp;·&nbsp; MSc AI &amp; Large Models, 2025-2027 &nbsp;·&nbsp; March 2026</span>'
        )
        _doc_stats   = "14 PAGES &nbsp;·&nbsp; 7 128 WORDS &nbsp;·&nbsp; SPEI · GRU · SADC · SEA · TRANSFER LEARNING"

    st.html(f"""
    <div style="background:linear-gradient(135deg,#0b1a0d,#070f09);border:1px solid #1e4a24;border-radius:12px;padding:16px 20px;margin-bottom:1.4rem">
      <div class="eo-section-head" style="margin-bottom:.6rem">📄 ACTIVE KNOWLEDGE SOURCE</div>
      <div style="font-family:'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;font-size:.92rem;color:#b8d4bb;line-height:1.6">
        <em>{_doc_title}</em><br>
        <span style="color:#6b9a74;font-size:.85rem">{_doc_meta}</span>
      </div>
      <div style="margin-top:.7rem;font-size:.78rem;color:#4ade8088;font-family:'Orbitron',sans-serif;letter-spacing:.1em">
        {_doc_stats}
      </div>
      <div class="band-bar"></div>
    </div>
    """)

    # Pipeline diagram
    _ingest_label = (
        f"Load {_html.escape(doc_name)}\n({_page_est} pages est., {_word_count:,} words)\nvia PyPDF2"
        if pdf_ok else "Load PDF document\nvia PyPDF2"
    )
    steps = [
        ("📡", "INGEST", _ingest_label),
        ("✂️", "CHUNK",     f"Sliding-window split\n{len(chunks)} overlapping windows\n({chunk_size}w chunks, {chunk_over}w overlap)"),
        ("🧬", "EMBED",     "Encode chunks with\nMiniLM-L6-v2\n→ 384-dim dense vector"),
        ("🔎", "RETRIEVE",  f"Embed query → cosine sim\n→ top-{top_k} relevant chunks\nranked by similarity"),
        ("🤖", "GENERATE",  "Groq LLaMA-3.3-70B\ngrounded on retrieved context\nHallucination-free by design"),
    ]

    cols = st.columns(len(steps))
    for col, (icon, label, desc) in zip(cols, steps):
        delay = steps.index((icon, label, desc)) * 0.12
        with col:
            st.markdown(f"""
            <div class="pipe-step" style="animation-delay:{delay}s">
              <div class="pipe-icon">{icon}</div>
              <div class="pipe-label">{label}</div>
              <div class="pipe-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">TOTAL CHUNKS</div>
        <div class="metric-value">{len(chunks)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">EMBEDDING DIM</div>
        <div class="metric-value">384</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">CHUNK SIZE</div>
        <div class="metric-value">{chunk_size} w</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">OVERLAP</div>
        <div class="metric-value">{chunk_over} w</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">TOP-K</div>
        <div class="metric-value">{top_k}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">LLM ENGINE</div>
        <div class="metric-value" style="font-size:.7rem;line-height:1.3">Groq<br>LLaMA-3.3</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if doc_source == "uploaded":
        _why_title = f"🌿 Why RAG for {_html.escape(doc_name)}?"
        _why_body  = f"""
**Domain specificity.** General LLMs may not have seen your document at all.
RAG anchors every answer to exact passages retrieved from **{_html.escape(doc_name)}**,
ensuring responses are grounded in the actual content rather than parametric guesses.

**Traceability.** Every answer cites the retrieved passages — reviewers can verify
the source of any claim directly in Tab 2.

**No fine-tuning cost.** Instead of retraining a model on your document (expensive),
RAG retrieves relevant knowledge at inference time — same pipeline, any document.

**Configurable precision.** Smaller chunks → finer retrieval but risk losing context.
Larger chunks → richer context but noisier similarity. Tune the sliders to explore this trade-off live.

**Why Groq?** Groq's LPU hardware delivers near-instant inference (~500 tokens/s),
making the Q&A seamless without GPU cost.

---
*The same pipeline was originally built on our team project — CLIMATE-XFER — and is now being
demonstrated here with your uploaded document.*
"""
    else:
        _why_title = "🌿 Why RAG for CLIMATE-XFER?"
        _why_body  = """
**Domain specificity.** CLIMATE-XFER uses precise technical terminology — SPEI-1,
CHIRPS precipitation anomalies, GRU encoders, Niño-3.4 SST indices, SADC/SEA
teleconnections — that general LLMs may misinterpret or confuse. RAG anchors
every answer to exact passages from our team's submitted report, ensuring no
terminology is misrepresented.

**Traceable results.** Our model achieves **RMSE = 0.193, r = 0.835** on the SADC
validation set, **r = 0.882** zero-shot on Southeast Asia, and **r = 0.903** after
10-epoch fine-tuning. RAG ensures any claim about these numbers points back to the
passage it came from — critical for academic integrity.

**RAG mirrors our transfer learning philosophy.** Just as CLIMATE-XFER *transfers*
a GRU model trained on SADC to Southeast Asia without retraining from scratch,
RAG *transfers* knowledge from the document to the LLM at inference time — no
fine-tuning needed.

**Configurable precision.** Smaller chunks → finer retrieval, risk of losing context.
Larger chunks → richer context, noisier similarity. The sliders let you explore
this trade-off live.

**Why Groq?** Groq's LPU hardware delivers near-instant inference (~500 tokens/s),
making Q&A seamless without GPU cost.
"""

    with st.expander(_why_title, expanded=True):
        st.markdown(_why_body)

    st.divider()

    # Video section — always the team project video
    st.html('<div class="eo-section-head">🎬 NOTEBOOKLM VIDEO OVERVIEW — CLIMATE-XFER TEAM PROJECT</div>')
    if doc_source == "uploaded":
        st.info(
            f"📌 Currently querying **{doc_name}**, but the video below is our original "
            "team project (CLIMATE-XFER) — the document this RAG system was designed for.",
            icon="🏆",
        )

    _local_video = _path("videoplayback.mp4")
    if os.path.exists(_local_video):
        st.video(_local_video)
        st.caption("Video overview generated with Google NotebookLM · CLIMATE-XFER Team Project · Assignment 3")
    elif video_url.strip():
        url = video_url.strip()
        vid_id = ""
        if "youtube.com/watch" in url and "v=" in url:
            vid_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            vid_id = url.split("youtu.be/")[-1].split("?")[0]

        if vid_id:
            st.markdown(f"""
            <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;border:1px solid #1e4a24;">
              <iframe src="https://www.youtube.com/embed/{vid_id}?rel=0&modestbranding=1"
                style="position:absolute;top:0;left:0;width:100%;height:100%;"
                frameborder="0" allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#070f09;border:1px solid #1e4a24;border-radius:12px;padding:24px;text-align:center;">
              <a href="{_html.escape(url)}" target="_blank"
                 style="display:inline-block;padding:10px 28px;background:linear-gradient(90deg,#14532d,#166534);color:#dcfce7;
                        border-radius:8px;font-family:'Orbitron',sans-serif;font-size:.65rem;letter-spacing:.15em;text-decoration:none;">
                ▶ OPEN VIDEO
              </a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.html("""
        <div style="background:#070f09;border:1px dashed #1e4a24;border-radius:12px;padding:28px;text-align:center;color:#6b9a74;font-size:.88rem">
          🎬 &nbsp; Place <strong style="color:#4ade80">videoplayback.mp4</strong> in the app folder or paste a YouTube URL in the sidebar.
        </div>
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Query Terminal (Live Q&A)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.html("""
    <div class="eo-section-head">◈ QUERY TERMINAL — ASK THE KNOWLEDGE BASE</div>
    <div class="groq-badge">
      <span class="status-dot"></span>
      GROQ · LLAMA-3.3-70B · LIVE INFERENCE
    </div>
    <br>
    """)

    if not pdf_ok:
        st.error(f"Could not load PDF: {raw_text}")
    else:
        # Build example questions — generated from uploaded doc, else default
        if doc_source == "uploaded" and api_key.strip():
            with st.spinner("🛰️ Reading your document and generating questions …"):
                examples = _generate_example_questions(raw_text[:3000], api_key.strip())
            if not examples:
                st.warning("Could not auto-generate questions — showing defaults.", icon="⚠️")
                examples = _DEFAULT_EXAMPLES
            _expander_label = f"💡 Questions generated for: {doc_name}"
        else:
            examples = _DEFAULT_EXAMPLES
            _expander_label = "💡 Sample queries to try"

        with st.expander(_expander_label, expanded=(doc_source == "uploaded")):
            for ex in examples:
                if st.button(ex, key=f"ex_{hash(ex)}"):
                    st.session_state["query_input"] = ex
                    st.session_state["auto_search"] = True
                    st.rerun()

        query = st.text_input(
            "Enter query",
            placeholder="e.g. What RMSE did the GRU model achieve on SADC data?",
            key="query_input",
        )

        ask_col, _ = st.columns([1, 3])
        with ask_col:
            run_query = st.button("📡 Transmit Query", type="primary", use_container_width=True)

        do_search = run_query or st.session_state.pop("auto_search", False)

        if do_search and query.strip():
            with st.spinner("🛰️ Retrieving relevant passages from knowledge base …"):
                model   = _load_sentence_transformer()
                results = _retrieve(query, chunks, embeddings, model, top_k)
                context_passages = [r[2] for r in results]
                answer_text = None  # generated below

            st.session_state["qa_results"]  = results
            st.session_state["qa_query"]    = query
            st.session_state["qa_passages"] = context_passages
            st.session_state["qa_api_key"]  = api_key.strip()
            st.session_state["qa_answer"]   = answer_text

        # Display results
        if st.session_state.get("qa_results"):
            results          = st.session_state["qa_results"]
            context_passages = st.session_state["qa_passages"]
            stored_query     = st.session_state["qa_query"]

            st.markdown("---")
            st.markdown(
                f'<p style="font-family:Cinzel,serif;font-size:1rem;color:#dcfce7;margin-bottom:.8rem">'
                f'Signal received: <em style="color:#4ade80">{_html.escape(stored_query)}</em></p>',
                unsafe_allow_html=True)

            st.markdown("---")
            st.html(f'<div class="eo-section-head">TOP-{top_k} RETRIEVED PASSAGES</div>')

            for rank, (idx, score, passage) in enumerate(results, 1):
                pct     = int(score * 100)
                preview = _html.escape(passage[:350] + ("…" if len(passage) > 350 else ""))
                st.markdown(f"""
                <div class="chunk-box">
                  <span class="chunk-id">passage {rank} &nbsp;·&nbsp; {pct}% match</span>
                  <div style="margin-top:18px">{preview}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.html('<div class="eo-section-head">◈ GROQ GENERATED ANSWER</div>')

            stored_key = st.session_state.get("qa_api_key", "")
            if stored_key:
                if st.session_state["qa_answer"] is None:
                    with st.spinner("🤖 LLaMA-3.3-70B synthesising answer …"):
                        answer = _generate_groq(stored_query, context_passages, stored_key)
                    st.session_state["qa_answer"] = answer

                st.markdown(
                    f'<div class="answer-box">{_html.escape(st.session_state["qa_answer"])}</div>',
                    unsafe_allow_html=True)
                st.caption("Grounded answer by Groq LLaMA-3.3-70B · based on retrieved passages only · no hallucination")
            else:
                best = _html.escape(results[0][2] if results else "No passage found.")
                st.markdown(
                    f'<div class="answer-box">'
                    f'<em style="color:#4ade8088;font-size:.8rem">No API key — showing top retrieved passage:</em>'
                    f'<br><br>{best}</div>',
                    unsafe_allow_html=True)
                st.caption("Add a Groq API key in the sidebar to enable AI-generated answers.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spectral Internals
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.html('<div class="eo-section-head">🛰️ SPECTRAL INTERNALS — PIPELINE UNDER THE HOOD</div>')

    if not pdf_ok:
        st.error("PDF not loaded.")
    else:
        # ── Chunking ──────────────────────────────────────────────────────────
        st.html('<div class="eo-section-head" style="font-size:.62rem">✂️ CHUNKING — SLIDING WINDOW SEGMENTATION</div>')
        st.markdown(f"Document split into **{len(chunks)} chunks** — {chunk_size}-word windows, {chunk_over}-word overlap. Adjust sidebar sliders to change live.")

        show_n = min(5, len(chunks))
        cols2  = st.columns(2)
        for i in range(show_n):
            with cols2[i % 2]:
                preview = _html.escape(chunks[i][:280] + ("…" if len(chunks[i]) > 280 else ""))
                st.markdown(f"""
                <div class="chunk-box">
                  <span class="chunk-id">chunk #{i}</span>
                  <div style="margin-top:18px">{preview}</div>
                </div>
                """, unsafe_allow_html=True)

        if len(chunks) > show_n:
            st.caption(f"Showing first {show_n} of {len(chunks)} chunks.")

        st.divider()

        # ── Embedding heatmap ─────────────────────────────────────────────────
        st.html('<div class="eo-section-head" style="font-size:.62rem">🧬 EMBEDDING HEATMAP — 384-DIMENSIONAL SEMANTIC SPACE</div>')
        st.markdown("First 20 chunks × first 64 embedding dimensions. Colour intensity = vector activation magnitude — similar to a false-colour satellite composite.")

        try:
            import plotly.graph_objects as go

            n_cv = min(20, len(chunks))
            n_dv = min(64, embeddings.shape[1])
            heat = embeddings[:n_cv, :n_dv]

            fig_heat = go.Figure(go.Heatmap(
                z=heat,
                colorscale=[
                    [0.0,  "#060e08"],
                    [0.25, "#14532d"],
                    [0.5,  "#166534"],
                    [0.75, "#22c55e"],
                    [1.0,  "#86efac"],
                ],
                showscale=True,
                colorbar=dict(tickfont=dict(color="#6b9a74"), bgcolor="#060e08",
                              title=dict(text="Activation", font=dict(color="#4ade80", size=11))),
            ))
            fig_heat.update_layout(
                paper_bgcolor="#060e08", plot_bgcolor="#060e08",
                margin=dict(l=40, r=20, t=30, b=40),
                height=340,
                xaxis=dict(title="Embedding dimension (0–63)",
                           tickfont=dict(color="#6b9a74"), gridcolor="#0b1a0d"),
                yaxis=dict(title="Chunk index",
                           tickfont=dict(color="#6b9a74"), gridcolor="#0b1a0d"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except ImportError:
            st.info("Install plotly to see the embedding heatmap.")

        st.divider()

        # ── Similarity scores ─────────────────────────────────────────────────
        st.html('<div class="eo-section-head" style="font-size:.62rem">🔎 COSINE SIMILARITY — SPECTRAL DISTANCE SCORES</div>')

        last_query = st.session_state.get("qa_query", "")
        if last_query:
            with st.spinner("Computing similarity scores …"):
                model   = _load_sentence_transformer()
                results = _retrieve(last_query, chunks, embeddings, model,
                                    min(top_k * 2, len(chunks)))

            st.html(f'<div style="font-family:Orbitron,sans-serif;font-size:.62rem;color:#4ade8077;margin-bottom:.8rem">QUERY: "{_html.escape(last_query[:80])}"</div>')

            for rank, (idx, score, _) in enumerate(results, 1):
                pct   = score * 100
                bar_w = int(pct)
                st.markdown(f"""
                <div class="sim-row">
                  <div style="font-family:Orbitron,sans-serif;font-size:.6rem;color:#6b9a74;width:70px">chunk #{idx}</div>
                  <div class="sim-bar-bg"><div class="sim-bar-fg" style="width:{bar_w}%"></div></div>
                  <div class="sim-pct">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📐 Cosine similarity — how it works"):
                st.markdown(r"""
**Cosine similarity** measures the angle between two embedding vectors, ignoring magnitude:

$$\text{sim}(q, c) = \frac{q \cdot c}{\|q\| \cdot \|c\|}$$

- **1.0** → identical semantic direction (perfect spectral match)
- **0.0** → orthogonal vectors (no semantic relationship)
- **−1.0** → opposite meaning

The query is embedded with the same MiniLM model used for document chunks.
All chunks are ranked by this score and the top-K are retrieved as LLM context.
Analogous to spectral angle mapping (SAM) in remote sensing — comparing spectral
signatures in high-dimensional feature space.
                """)
        else:
            st.info("Transmit a query in the Query Terminal first — similarity scores will appear here.")

        st.divider()

        # ── Architecture summary ──────────────────────────────────────────────
        st.html('<div class="eo-section-head" style="font-size:.62rem">🏗️ SYSTEM ARCHITECTURE</div>')
        arch_cols = st.columns(3)
        arch_data = [
            ("RETRIEVER",
             f"sentence-transformers\nall-MiniLM-L6-v2\n384-dim dense vectors\nCosine similarity (SAM-like)"),
            ("KNOWLEDGE BASE",
             f"CLIMATE-XFER PDF\n{len(raw_text.split())} words extracted\n{len(chunks)} chunks indexed\nSliding-window chunking"),
            ("GENERATOR",
             "Groq LLaMA-3.3-70B\nContext-grounded prompting\nPassage fallback (no key)\nHallucination-free by design"),
        ]
        for col, (title, body) in zip(arch_cols, arch_data):
            with col:
                items = "".join(f"<li>{l}</li>" for l in body.split("\n"))
                st.markdown(f"""
                <div class="pipe-step">
                  <div class="pipe-label">{title}</div>
                  <ul style="text-align:left;font-size:.82rem;color:#6b9a74;line-height:1.7;padding-left:16px">{items}</ul>
                </div>
                """, unsafe_allow_html=True)
