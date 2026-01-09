import streamlit as st
import pandas as pd
import plotly.express as px

from backend.config import SETTINGS
from backend.vectorstore import FaissStore
from backend.embeddings import Embedder

import umap

st.title("Embedding Space Explorer (UMAP)")

store = FaissStore(SETTINGS.index_dir)
if not store.load():
    st.warning("No index found. Build one first.")
    st.stop()

items = store.meta["items"]
N_total = len(items)

st.write(f"Chunks in index: {N_total}")

max_points = st.slider("Max chunks to plot (for speed)", 50, 3000, min(800, N_total), step=50)
subset = items[:max_points]
N = len(subset)

embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)
embedder = Embedder(embed_model)

# UMAP controls
requested_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15, 1)
min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.01)

if st.button("Generate 2D Map", type="primary"):
    if N < 5:
        st.error("Not enough chunks to run UMAP. Ingest a bigger PDF or more documents (need at least ~5 chunks).")
        st.stop()

    # Embed chunk texts
    texts = [c["text"] for c in subset]
    vecs = embedder.embed_texts(texts)

    # ✅ Make UMAP safe for small N
    # n_neighbors must be < N
    n_neighbors = min(requested_neighbors, max(2, N - 1))

    # For very small N, spectral init can fail. Use random init.
    init_mode = "random" if N < 20 else "spectral"

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        init=init_mode,
        random_state=42
    )

    xy = reducer.fit_transform(vecs)

    df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "source": [c["source"] for c in subset],
        "page": [c["page"] for c in subset],
        "chunk_id": [c["chunk_id"] for c in subset],
        "preview": [c["text"][:160].replace("\n", " ") for c in subset]
    })

    fig = px.scatter(
        df, x="x", y="y",
        hover_data=["chunk_id", "source", "page", "preview"],
        color="source",
        title=f"Chunk Embeddings in 2D (UMAP) | N={N}, n_neighbors={n_neighbors}, init={init_mode}"
    )
    st.plotly_chart(fig, use_container_width=True)

st.info("If UMAP fails, it’s usually because there are too few chunks. Upload more PDFs or lower max_points / neighbors.")
