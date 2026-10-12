"""Streamlit UI for the document assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import NOT_FOUND, DocumentIndex
from rag.embeddings import Embedder
from rag.llm import DEFAULT_MODEL, GroqLLM, build_messages, cited_numbers
from rag.retriever import tokenize

load_dotenv()
SAMPLE = Path(__file__).parent / "sample_docs" / "northwind_handbook.pdf"
MODELS = [DEFAULT_MODEL, "llama-3.1-8b-instant", "openai/gpt-oss-120b"]

st.set_page_config(page_title="Docs Q&A", page_icon="📑", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1100px;}
  h1 {font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.5px;}
  .tagline {color: #6b5e4f; margin-top: -0.6rem; margin-bottom: 1.4rem;}
  .src-card {border: 1px solid #e3d9c8; border-left: 4px solid #b5562b; border-radius: 6px;
             padding: 0.7rem 0.9rem; margin-bottom: 0.6rem; background: #fffdf8;}
  .src-head {font-size: 0.8rem; color: #6b5e4f; margin-bottom: 0.3rem; display: flex; gap: 0.5rem; flex-wrap: wrap;}
  .src-num {font-weight: 700; color: #b5562b;}
  .badge {background: #f1e8da; border-radius: 10px; padding: 0 0.5rem; font-size: 0.72rem;}
  .src-text {font-size: 0.88rem; line-height: 1.45; color: #2b2520;}
  .src-text mark {background: #ffe1a8; padding: 0 1px;}
  .stat {font-size: 0.85rem; color: #6b5e4f;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Reading and indexing documents...")
def build_index(files: tuple[tuple[str, bytes], ...], use_dense: bool) -> DocumentIndex:
    return DocumentIndex.from_files(list(files), embedder=get_embedder(), use_dense=use_dense)


def highlight(text: str, query: str) -> str:
    safe = html.escape(text)
    terms = sorted({t for t in tokenize(query) if len(t) > 2}, key=len, reverse=True)
    for t in terms:
        safe = re.sub(rf"(?i)\b({re.escape(t)}\w*)", r"<mark>\1</mark>", safe)
    return safe


def render_sources(hits, question: str, cited: list[int] | None = None):
    for n, h in enumerate(hits, start=1):
        if cited is not None and n not in cited:
            continue
        badges = "".join(f'<span class="badge">{s}</span>' for s in h.sources)
        st.markdown(
            f'<div class="src-card"><div class="src-head"><span class="src-num">[{n}]</span>'
            f'<span>{html.escape(h.chunk.source)} · page {h.chunk.page}</span>{badges}</div>'
            f'<div class="src-text">{highlight(h.chunk.text, question)}</div></div>',
            unsafe_allow_html=True)


# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("Documents")
    uploads = st.file_uploader("PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    use_sample = st.toggle("Use the sample handbook", value=not uploads,
                           help="A made-up company handbook, handy for trying the app out.")

    st.subheader("Settings")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""),
                            help="Free at console.groq.com. Without a key you still get the matching passages.")
    model = st.selectbox("Model", MODELS)
    top_k = st.slider("Passages to retrieve", 2, 10, 5)
    use_dense = st.toggle("Semantic search (embeddings)", value=True,
                          help="Turn off to use keyword search (BM25) only.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

files = [(f.name, f.getvalue()) for f in (uploads or [])]
if use_sample and SAMPLE.exists():
    files.append((SAMPLE.name, SAMPLE.read_bytes()))

st.title("Ask your documents")
st.markdown('<div class="tagline">Answers come only from the files you upload, with a page reference for every claim.</div>',
            unsafe_allow_html=True)

if not files:
    st.info("Upload a document in the sidebar, or switch on the sample handbook to try it out.")
    st.stop()

index = build_index(tuple(files), use_dense)
retriever = index.retriever
mode = "hybrid (keyword + semantic)" if retriever.mode == "hybrid" else "keyword only (BM25)"
st.markdown(f'<div class="stat">{len(index.sources)} file(s) · {len(index.pages)} pages · '
            f'{len(index.chunks)} chunks · search: {mode}</div>', unsafe_allow_html=True)
if use_dense and retriever.dense_error:
    st.warning("The embedding model couldn't be loaded, so search is keyword-only for now. "
               "It downloads once on first use and needs an internet connection.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            with st.expander(f"Sources ({len(m['cited'] or m['hits'])})"):
                render_sources(m["hits"], m["question"], m["cited"] or None)

if not st.session_state.messages:
    st.caption("Try: *How many leave days can I carry over?* · *What's the meal allowance abroad?* · "
               "*Can I work from Dubai for a month?*")

question = st.chat_input("Ask a question about your documents")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits, relevant = index.retrieve(question, k=top_k)
    with st.chat_message("assistant"):
        cited: list[int] = []
        if not relevant:
            answer = NOT_FOUND
            st.markdown(answer)
        elif not api_key:
            answer = "Add a Groq API key in the sidebar to get a written answer. These are the most relevant passages:"
            st.markdown(answer)
            render_sources(hits, question)
        else:
            try:
                llm = GroqLLM(api_key=api_key, model=model)
                answer = st.write_stream(llm.stream(build_messages(question, hits, history)))
                cited = cited_numbers(answer, len(hits))
                with st.expander(f"Sources ({len(cited) or len(hits)})"):
                    render_sources(hits, question, cited or None)
            except Exception as e:
                answer = f"The model call failed: {e}"
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "question": question,
                                      "hits": hits if relevant and api_key else [], "cited": cited})
