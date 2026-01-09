import os
import streamlit as st
import pandas as pd

from backend.config import SETTINGS
from backend.utils import ensure_dir
from backend.loaders import load_pdf_pages
from backend.chunking import chunk_pages
from backend.embeddings import Embedder
from backend.vectorstore import FaissStore
from app._bootstrap import bootstrap
bootstrap()

st.title("Ingest & Build FAISS Index")

ensure_dir("data")
ensure_dir(SETTINGS.index_dir)

uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

col1, col2, col3 = st.columns(3)
with col1:
    chunk_tokens = st.slider("Chunk size (tokens)", 250, 700, SETTINGS.chunk_tokens, step=10)
with col2:
    overlap_tokens = st.slider("Overlap (tokens)", 0, 200, SETTINGS.chunk_overlap, step=10)
with col3:
    model_name = st.text_input("Embedding model", SETTINGS.embedding_model)

if uploaded:
    st.write("Uploaded files:")
    for f in uploaded:
        st.write("-", f.name)

if st.button("Build Index", type="primary", disabled=not uploaded):
    # Save PDFs into data/
    paths = []
    for f in uploaded:
        path = os.path.join("data", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)

    # Load pages
    pages = []
    for path in paths:
        pages.extend(load_pdf_pages(path))

    st.write(f"Loaded pages: {len(pages)}")

    # Chunk
    chunks = chunk_pages(pages, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    st.write(f"Created chunks: {len(chunks)}")

    df = pd.DataFrame([{
        "chunk_id": c["chunk_id"],
        "source": c["source"],
        "page": c["page"],
        "token_count": c["token_count"],
        "preview": c["text"][:140].replace("\n", " ") + "..."
    } for c in chunks])
    st.subheader("Chunk preview")
    st.dataframe(df, use_container_width=True)

    # Embed
    embedder = Embedder(model_name)
    texts = [c["text"] for c in chunks]
    vecs = embedder.embed_texts(texts)

    # Build FAISS
    store = FaissStore(SETTINGS.index_dir)
    store.build(vecs, chunks)

    st.success("Index built and saved to disk (index/faiss.index + index/meta.json).")
    st.info("Next: go to “Ask & Explain” and try questions.")
