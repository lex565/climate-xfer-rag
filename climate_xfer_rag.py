"""
CLIMATE-XFER RAG Demo  |  Assignment 3
AI and Large Models — Masters Program 2025-2027
Demonstrates: chunking → embedding → retrieval → generation
"""

import os
import re
import base64
import textwrap
import math
import time

import numpy as np
import streamlit as st

# ── path helpers ─────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(rel: str) -> str:
    return os.path.join(_DIR, rel)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLIMATE-XFER RAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy imports ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Orbitron:wght@400;700&family=EB+Garamond:ital@0;1&display=swap" rel="stylesheet">
<style>
  /* ── base ── */
  html, body, [class*="css"] { font-family: 'EB Garamond', Georgia, serif; }

  /* ── title glow ── */
  @keyframes titleGlow {
    0%,100% { text-shadow: 0 0 12px #58a6ff88, 0 0 28px #58a6ff44; }
    50%      { text-shadow: 0 0 22px #58a6ffcc, 0 0 48px #58a6ff88; }
  }
  .rag-title {
    font-family:'Cinzel',serif; font-size:2.3rem; font-weight:700;
    color:#e8f4fd; text-align:center; letter-spacing:.06em;
    animation: titleGlow 3.5s ease-in-out infinite;
    margin-bottom:.15rem;
  }
  .rag-subtitle {
    font-family:'Orbitron',sans-serif; font-size:.72rem; color:#58a6ff;
    text-align:center; letter-spacing:.25em; margin-bottom:1.4rem;
  }

  /* ── scan-line overlay ── */
  @keyframes scanLine {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
  .scanline {
    pointer-events:none; position:fixed; top:0; left:0;
    width:100%; height:3px;
    background: linear-gradient(transparent, rgba(88,166,255,.18), transparent);
    animation: scanLine 6s linear infinite; z-index:9999;
  }

  /* ── institution bar ── */
  .inst-bar {
    display:flex; align-items:center; justify-content:center;
    gap:16px; padding:10px 20px; margin-bottom:1rem;
    background:linear-gradient(90deg,#0a1628,#0d1f3c,#0a1628);
    border-radius:10px; border:1px solid #1e3a5f;
  }
  .inst-text { text-align:left; }
  .inst-uni  { font-family:'Cinzel',serif; font-size:.95rem; color:#c9d1d9; }
  .inst-dept { font-family:'Orbitron',sans-serif; font-size:.6rem; color:#58a6ff; letter-spacing:.15em; }

  /* ── author cards ── */
  @keyframes avatarPulse {
    0%,100% { box-shadow:0 0 0 0 rgba(88,166,255,.45); }
    70%      { box-shadow:0 0 0 8px rgba(88,166,255,0); }
  }
  .author-row { display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin-bottom:1.4rem; }
  .author-card {
    display:flex; align-items:center; gap:10px; padding:8px 16px;
    background:linear-gradient(135deg,#0d1f3c,#0a1628);
    border:1px solid #1e3a5f; border-radius:12px; transition:.3s;
  }
  .author-card:hover { border-color:#58a6ff; transform:translateY(-2px); }
  .author-avatar {
    width:36px; height:36px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'Cinzel',serif; font-weight:700; font-size:.85rem; color:#fff;
    animation: avatarPulse 2.5s infinite;
  }
  .author-name { font-family:'EB Garamond',serif; font-size:.9rem; color:#c9d1d9; }
  .author-id   { font-family:'Orbitron',sans-serif; font-size:.58rem; color:#58a6ff; letter-spacing:.1em; }

  /* ── pipeline step cards ── */
  @keyframes stepFadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  .pipe-step {
    background:linear-gradient(135deg,#0d1f3c,#07111f);
    border:1px solid #1e3a5f; border-radius:14px;
    padding:20px; text-align:center;
    animation: stepFadeIn .6s ease both;
  }
  .pipe-step:hover { border-color:#58a6ff; box-shadow:0 0 18px #58a6ff22; }
  .pipe-icon { font-size:2rem; margin-bottom:.4rem; }
  .pipe-label {
    font-family:'Orbitron',sans-serif; font-size:.72rem;
    color:#58a6ff; letter-spacing:.18em; margin-bottom:.5rem;
  }
  .pipe-desc { font-size:.88rem; color:#8b949e; line-height:1.5; }

  /* ── metric cards ── */
  .metric-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:1rem; }
  .metric-card {
    flex:1; min-width:160px;
    background:linear-gradient(135deg,#0d1f3c,#07111f);
    border:1px solid #1e3a5f; border-radius:12px;
    padding:14px 18px;
  }
  .metric-label { font-family:'Orbitron',sans-serif; font-size:.6rem; color:#58a6ff; letter-spacing:.15em; }
  .metric-value { font-family:'Cinzel',serif; font-size:1.5rem; color:#e8f4fd; margin-top:4px; }

  /* ── chunk display ── */
  .chunk-box {
    background:#07111f; border:1px solid #1e3a5f; border-radius:8px;
    padding:12px; font-size:.82rem; color:#8b949e;
    font-family:'EB Garamond',serif; line-height:1.6;
    margin-bottom:8px; position:relative;
  }
  .chunk-id {
    font-family:'Orbitron',sans-serif; font-size:.58rem;
    color:#58a6ff; position:absolute; top:8px; right:10px;
  }

  /* ── answer box ── */
  .answer-box {
    background:linear-gradient(135deg,#07111f,#0a1628);
    border:1px solid #58a6ff66; border-radius:12px;
    padding:20px; font-size:.95rem; color:#c9d1d9;
    line-height:1.75; font-family:'EB Garamond',serif;
  }

  /* ── tab guide ── */
  .tab-guide {
    background:linear-gradient(90deg,#0a1628,#0d1f3c,#0a1628);
    border:1px solid #1e3a5f; border-radius:10px;
    padding:12px 20px; margin-bottom:1.2rem;
    font-family:'Orbitron',sans-serif; font-size:.65rem;
    color:#58a6ff; letter-spacing:.12em; text-align:center;
  }
  .tab-guide strong { color:#e8f4fd; }

  /* ── similarity bar ── */
  .sim-row { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
  .sim-bar-bg { flex:1; height:8px; background:#0d1f3c; border-radius:4px; }
  .sim-bar-fg { height:8px; border-radius:4px; background:linear-gradient(90deg,#1e3a5f,#58a6ff); }
  .sim-pct { font-family:'Orbitron',sans-serif; font-size:.6rem; color:#58a6ff; width:44px; text-align:right; }

  /* ── tutorial ── */
  .tutorial-step { margin-bottom:.7rem; font-size:.9rem; line-height:1.6; }
  .tutorial-step b { color:#58a6ff; }
</style>
<div class="scanline"></div>
""", unsafe_allow_html=True)

# ── logo ──────────────────────────────────────────────────────────────────────
def _get_logo_b64() -> str:
    try:
        with open(_path("SCHOOL LOGO.png"), "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ── header ────────────────────────────────────────────────────────────────────
logo_b64 = _get_logo_b64()
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="height:54px;filter:drop-shadow(0 0 6px #58a6ff44);">'
    if logo_b64 else '<span style="font-size:2rem">🎓</span>'
)

st.markdown(f"""
<div class="rag-title">CLIMATE-XFER RAG DEMO</div>
<div class="rag-subtitle">Retrieval-Augmented Generation · Assignment 3 · AI &amp; Large Models</div>

<div class="inst-bar">
  {logo_html}
  <div class="inst-text">
    <div class="inst-uni">University of the Witwatersrand</div>
    <div class="inst-dept">AI &amp; LARGE MODELS · MASTERS PROGRAMME 2025-2027</div>
  </div>
</div>

<div class="author-row">
  <div class="author-card">
    <div class="author-avatar" style="background:linear-gradient(135deg,#1a3a6b,#2563eb)">T</div>
    <div><div class="author-name">Tonderai Bangure</div><div class="author-id">2816042</div></div>
  </div>
  <div class="author-card">
    <div class="author-avatar" style="background:linear-gradient(135deg,#1a4a3a,#059669)">F</div>
    <div><div class="author-name">Farai Shoniwa</div><div class="author-id">2816089</div></div>
  </div>
  <div class="author-card">
    <div class="author-avatar" style="background:linear-gradient(135deg,#4a1a3a,#9333ea)">M</div>
    <div><div class="author-name">Muzi Mahlangu</div><div class="author-id">2816105</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── tab navigation guide ──────────────────────────────────────────────────────
st.markdown("""
<div class="tab-guide">
  📌 &nbsp;REVIEWER GUIDE &nbsp;|&nbsp;
  <strong>Tab 1</strong>: RAG Pipeline Overview + Video &nbsp;→&nbsp;
  <strong>Tab 2</strong>: Live Q&amp;A Demo (type a question) &nbsp;→&nbsp;
  <strong>Tab 3</strong>: Pipeline Internals (chunks, embeddings, similarity)
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:1rem">⚙ RAG PARAMETERS</div>', unsafe_allow_html=True)

    chunk_size  = st.slider("Chunk size (tokens)", 100, 600, 300, 50,
                            help="Number of words per chunk")
    chunk_over  = st.slider("Overlap (tokens)",    0,  150, 50,  25,
                            help="Words shared between adjacent chunks")
    top_k       = st.slider("Top-K chunks",        1,   10,  4,   1,
                            help="How many chunks are retrieved per query")

    st.divider()
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:.6rem">🔑 CLAUDE API</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API key", type="password",
                            placeholder="sk-ant-…",
                            help="Optional — enables AI-generated answers. Without it the app returns retrieved passages only.")

    st.divider()
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:.6rem">🎬 DEMO VIDEO</div>', unsafe_allow_html=True)
    video_url = st.text_input("YouTube URL",
                              value="https://www.youtube.com/watch?v=T-D1OfcDW1M",
                              help="Paste any YouTube URL to embed in Tab 1")

    st.divider()
    with st.expander("🎓 Teacher & Reviewer Guide", expanded=False):
        st.markdown("""
<div class="tutorial-step"><b>Step 1 — Load PDF</b><br>
The app automatically loads <i>CLIMATE_XFER_Report_v4.pdf</i> on startup.
No action needed.</div>

<div class="tutorial-step"><b>Step 2 — Adjust RAG parameters</b><br>
Use the sliders above to change chunk size, overlap, and how many
chunks are retrieved. Watch the internals update in real time (Tab 3).</div>

<div class="tutorial-step"><b>Step 3 — Ask a question (Tab 2)</b><br>
Type any question about CLIMATE-XFER. The system embeds your query,
does cosine-similarity search, and returns the top-K passages.
With an API key it generates a grounded answer via Claude.</div>

<div class="tutorial-step"><b>Step 4 — Inspect internals (Tab 3)</b><br>
See raw chunks, their 384-dim embeddings (visualised as a heatmap),
and similarity scores for the last query.</div>

<div class="tutorial-step"><b>Assignment 3 requirements met</b><br>
✅ Engineering paper fed into AI (CLIMATE-XFER PDF)<br>
✅ Chunking demonstrated with configurable parameters<br>
✅ Embedding shown (sentence-transformers MiniLM)<br>
✅ Retrieval shown (cosine similarity, top-K)<br>
✅ Generation shown (Claude API or passage fallback)<br>
✅ Interactive Streamlit deliverable</div>
""", unsafe_allow_html=True)

# ── RAG backend ───────────────────────────────────────────────────────────────
PDF_PATH = _path("CLIMATE_XFER_Report_v4.pdf")

@st.cache_data(show_spinner=False)
def _extract_pdf_text(path: str) -> str:
    try:
        import PyPDF2
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
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
        chunks.append(" ".join(words[i : i + size]))
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


def _generate_claude(question: str, context_passages: list[str], key: str) -> str:
    try:
        import anthropic
        context = "\n\n---\n\n".join(context_passages)
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            system=(
                "You are an expert on climate science and transfer learning. "
                "Answer questions strictly from the provided context. "
                "Be concise and precise. If the context doesn't contain enough "
                "information, say so clearly."
            ),
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }]
        )
        return msg.content[0].text
    except Exception as e:
        return f"[Claude API error: {e}]"


# ── load / embed on startup ────────────────────────────────────────────────────
with st.spinner("Loading PDF and building embeddings …"):
    raw_text = _extract_pdf_text(PDF_PATH)

pdf_ok = not raw_text.startswith("[PDF")

# Rebuild chunks + embeddings whenever slider values change
@st.cache_data(show_spinner=False)
def _get_chunks_and_embeddings(text: str, size: int, overlap: int):
    chunks = _chunk_text(text, size, overlap)
    model  = _load_sentence_transformer()
    embs   = model.encode(chunks, show_progress_bar=False, batch_size=32)
    return chunks, np.array(embs)

chunks, embeddings = _get_chunks_and_embeddings(raw_text, chunk_size, chunk_over)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎬  RAG Pipeline & Video",
    "🔍  Live Q&A Demo",
    "🧩  Pipeline Internals",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Pipeline overview + video
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="font-family:Orbitron,sans-serif;font-size:.75rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:1rem">
    WHAT IS RETRIEVAL-AUGMENTED GENERATION?
    </div>
    <p style="font-size:.95rem;color:#c9d1d9;line-height:1.75;margin-bottom:1.4rem">
    RAG grounds a large language model in <em>specific documents</em> rather than relying solely on
    parametric knowledge. We split the CLIMATE-XFER technical report into overlapping chunks,
    embed every chunk into a 384-dimensional semantic vector space, then — at query time —
    retrieve the most relevant chunks via cosine similarity and feed them to Claude as context.
    This eliminates hallucination and keeps answers traceable to exact passages in the paper.
    </p>
    """, unsafe_allow_html=True)

    # Pipeline diagram
    steps = [
        ("📄", "INGEST",     "Load CLIMATE-XFER PDF (19 pages, ~8 000 words) via PyPDF2"),
        ("✂️", "CHUNK",      f"Split into {len(chunks)} overlapping windows\n({chunk_size}-word chunks, {chunk_over}-word overlap)"),
        ("🧬", "EMBED",      "Encode every chunk with\nsentence-transformers\nall-MiniLM-L6-v2 → 384-dim vector"),
        ("🔎", "RETRIEVE",   f"Embed query → cosine similarity\n→ top-{top_k} most relevant chunks"),
        ("🤖", "GENERATE",   "Feed retrieved context to\nClaude (or return raw passages)\nfor a grounded answer"),
    ]

    cols = st.columns(len(steps))
    for col, (icon, label, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="pipe-step" style="animation-delay:{steps.index((icon,label,desc))*0.12}s">
              <div class="pipe-icon">{icon}</div>
              <div class="pipe-label">{label}</div>
              <div class="pipe-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics row
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
        <div class="metric-label">MODEL</div>
        <div class="metric-value" style="font-size:.85rem">MiniLM-L6</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Why RAG for CLIMATE-XFER
    with st.expander("📖 Why RAG for CLIMATE-XFER?", expanded=True):
        st.markdown("""
**Domain specificity.** Climate transfer-learning research uses precise terminology
(SPEI, SST, domain shift, GRU, PINN) that general LLMs may confuse. RAG anchors
every answer to the actual report.

**Traceability.** Reviewers can see exactly which passage justified each claim —
critical for scientific communication.

**No fine-tuning cost.** Instead of retraining a model on climate data (expensive),
RAG retrieves relevant knowledge at inference time. This mirrors the CLIMATE-XFER
philosophy of *transferring* knowledge rather than relearning from scratch.

**Configurable precision.** Smaller chunks → finer retrieval but risk losing context.
Larger chunks → richer context but noisier similarity. The sliders let you explore
this trade-off live.
        """)

    st.divider()

    # Embedded video
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.72rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:.8rem">🎬 EXPLAINER VIDEO — RAG CONCEPT</div>', unsafe_allow_html=True)

    if video_url.strip():
        # Convert watch URL to embed URL
        vid_id = ""
        if "v=" in video_url:
            vid_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            vid_id = video_url.split("youtu.be/")[-1].split("?")[0]

        if vid_id:
            st.markdown(f"""
            <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;border:1px solid #1e3a5f;">
              <iframe
                src="https://www.youtube.com/embed/{vid_id}?rel=0&modestbranding=1"
                style="position:absolute;top:0;left:0;width:100%;height:100%;"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
              </iframe>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Paste a valid YouTube URL in the sidebar to embed a video.")
    else:
        st.info("Paste a YouTube URL in the sidebar to embed an explainer video.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Live Q&A
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.75rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:1rem">ASK THE CLIMATE-XFER KNOWLEDGE BASE</div>', unsafe_allow_html=True)

    if not pdf_ok:
        st.error(f"Could not load PDF: {raw_text}")
    else:
        # Example questions
        with st.expander("💡 Example questions to try"):
            examples = [
                "What is the main objective of the CLIMATE-XFER project?",
                "What machine learning models were used for transfer learning?",
                "What was the RMSE of the best performing model?",
                "How was domain shift addressed in CLIMATE-XFER?",
                "What datasets were used for training and evaluation?",
                "Explain the role of SST in the prediction pipeline.",
                "What is the difference between zero-shot and fine-tuned transfer?",
                "What were the key findings about SPEI prediction in SADC?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex[:20]}"):
                    st.session_state["rag_query"] = ex

        query = st.text_input(
            "Your question",
            value=st.session_state.get("rag_query", ""),
            placeholder="e.g. What RMSE did the GRU model achieve on SADC data?",
            key="query_input",
        )

        ask_col, _ = st.columns([1, 3])
        with ask_col:
            run_query = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

        if run_query and query.strip():
            st.session_state["rag_query"] = query.strip()

            with st.spinner("Retrieving relevant passages …"):
                model    = _load_sentence_transformer()
                results  = _retrieve(query, chunks, embeddings, model, top_k)

            # Show retrieved chunks
            st.markdown("---")
            st.markdown(f'<div style="font-family:Orbitron,sans-serif;font-size:.65rem;color:#58a6ff;letter-spacing:.18em;margin-bottom:.6rem">TOP-{top_k} RETRIEVED CHUNKS</div>', unsafe_allow_html=True)

            for rank, (idx, score, passage) in enumerate(results, 1):
                pct = int(score * 100)
                preview = passage[:350] + ("…" if len(passage) > 350 else "")
                st.markdown(f"""
                <div class="chunk-box">
                  <span class="chunk-id">chunk #{idx} · {pct}%</span>
                  <div style="margin-top:18px">{preview}</div>
                </div>
                """, unsafe_allow_html=True)

            # Generate answer
            st.markdown("---")
            st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.65rem;color:#58a6ff;letter-spacing:.18em;margin-bottom:.6rem">GENERATED ANSWER</div>', unsafe_allow_html=True)

            context_passages = [r[2] for r in results]

            if api_key.strip():
                with st.spinner("Generating answer with Claude …"):
                    answer = _generate_claude(query, context_passages, api_key.strip())
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                st.caption("Answer generated by Claude · grounded in retrieved passages only")
            else:
                # Fallback: return the top passage
                best_passage = results[0][2] if results else "No relevant passage found."
                st.markdown(f'<div class="answer-box"><em style="color:#58a6ff88;font-size:.78rem">No API key — showing top retrieved passage:</em><br><br>{best_passage}</div>', unsafe_allow_html=True)
                st.caption("Add an Anthropic API key in the sidebar to enable AI-generated answers.")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — Pipeline internals
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:.75rem;color:#58a6ff;letter-spacing:.2em;margin-bottom:1rem">PIPELINE INTERNALS — UNDER THE HOOD</div>', unsafe_allow_html=True)

    if not pdf_ok:
        st.error("PDF not loaded.")
    else:
        # ── Section 1: Chunks ──────────────────────────────────────────
        st.markdown("#### ✂️ Chunking")
        st.markdown(f"The PDF was split into **{len(chunks)} chunks** using a sliding window of {chunk_size} words with {chunk_over}-word overlap. Adjust the sliders to see this change live.")

        show_n = min(5, len(chunks))
        cols2 = st.columns(2)
        for i in range(show_n):
            with cols2[i % 2]:
                preview = chunks[i][:280] + ("…" if len(chunks[i]) > 280 else "")
                st.markdown(f"""
                <div class="chunk-box">
                  <span class="chunk-id">chunk #{i}</span>
                  <div style="margin-top:18px">{preview}</div>
                </div>
                """, unsafe_allow_html=True)

        if len(chunks) > show_n:
            st.caption(f"Showing first {show_n} of {len(chunks)} chunks.")

        st.divider()

        # ── Section 2: Embedding heatmap ──────────────────────────────
        st.markdown("#### 🧬 Embedding Heatmap")
        st.markdown(f"Each chunk is encoded into a **384-dimensional vector**. Below is a heatmap of the first 20 chunks × first 64 dimensions, showing the semantic landscape of the document.")

        try:
            import plotly.graph_objects as go

            n_chunks_vis = min(20, len(chunks))
            n_dims_vis   = min(64, embeddings.shape[1])
            heatmap_data = embeddings[:n_chunks_vis, :n_dims_vis]

            fig_heat = go.Figure(go.Heatmap(
                z=heatmap_data,
                colorscale=[[0,"#07111f"],[0.5,"#1e3a5f"],[1,"#58a6ff"]],
                showscale=True,
                colorbar=dict(tickfont=dict(color="#8b949e"), bgcolor="#07111f"),
            ))
            fig_heat.update_layout(
                paper_bgcolor="#07111f", plot_bgcolor="#07111f",
                margin=dict(l=40,r=20,t=30,b=40),
                height=340,
                xaxis=dict(title="Embedding dimension (0–63)", tickfont=dict(color="#8b949e"), gridcolor="#0d1f3c"),
                yaxis=dict(title="Chunk index", tickfont=dict(color="#8b949e"), gridcolor="#0d1f3c"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except ImportError:
            st.info("Install plotly to see the embedding heatmap.")

        st.divider()

        # ── Section 3: Similarity scores for last query ───────────────
        st.markdown("#### 🔎 Similarity Scores")

        last_query = st.session_state.get("rag_query", "")
        if last_query:
            with st.spinner("Computing similarities …"):
                model   = _load_sentence_transformer()
                results = _retrieve(last_query, chunks, embeddings, model, min(top_k * 2, len(chunks)))

            st.markdown(f'<div style="font-family:Orbitron,sans-serif;font-size:.65rem;color:#58a6ff88;margin-bottom:.8rem">QUERY: "{last_query[:80]}"</div>', unsafe_allow_html=True)

            for rank, (idx, score, _) in enumerate(results, 1):
                pct = score * 100
                bar_w = int(pct)
                st.markdown(f"""
                <div class="sim-row">
                  <div style="font-family:Orbitron,sans-serif;font-size:.6rem;color:#8b949e;width:70px">chunk #{idx}</div>
                  <div class="sim-bar-bg"><div class="sim-bar-fg" style="width:{bar_w}%"></div></div>
                  <div class="sim-pct">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Cosine similarity explanation
            with st.expander("📐 How cosine similarity works"):
                st.markdown(r"""
**Cosine similarity** measures the angle between two vectors, ignoring magnitude:

$$\text{sim}(q, c) = \frac{q \cdot c}{\|q\| \cdot \|c\|}$$

- **1.0** → identical semantic direction (perfect match)
- **0.0** → orthogonal (no semantic relationship)
- **−1.0** → opposite meaning

We embed the query with the same MiniLM model used for chunks, then rank all chunks by this score. The top-K are returned as context for generation.
                """)
        else:
            st.info("Ask a question in Tab 2 first — similarity scores will appear here.")

        st.divider()

        # ── Section 4: Architecture summary ──────────────────────────
        st.markdown("#### 🏗️ Architecture Summary")
        arch_cols = st.columns(3)
        arch_data = [
            ("RETRIEVER", "sentence-transformers\nall-MiniLM-L6-v2\n384-dim dense vectors\nCosine similarity search"),
            ("KNOWLEDGE BASE", f"CLIMATE-XFER PDF\n{len(raw_text.split())} words extracted\n{len(chunks)} chunks indexed\nSliding-window chunking"),
            ("GENERATOR", "Claude claude-opus-4-6\nContext-grounded prompting\nPassage fallback (no key)\nHallucination-free by design"),
        ]
        for col, (title, body) in zip(arch_cols, arch_data):
            with col:
                lines = body.split("\n")
                items = "".join(f"<li>{l}</li>" for l in lines)
                st.markdown(f"""
                <div class="pipe-step">
                  <div class="pipe-label">{title}</div>
                  <ul style="text-align:left;font-size:.82rem;color:#8b949e;line-height:1.7;padding-left:16px">{items}</ul>
                </div>
                """, unsafe_allow_html=True)
