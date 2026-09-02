import streamlit as st
import pandas as pd
import plotly.express as px

from backend.config import SETTINGS
from backend.vectorstore import FaissStore
from backend.embeddings import Embedder
from app.ui import configure_page, footer, page_header, section, status_pills
from app.security_ui import security_context, tenant_store

configure_page("Embedding explorer", "⠿")
security = security_context("documents:read")
page_header("Semantic map", "Explore how your knowledge clusters.", "Project high-dimensional chunk embeddings into an interactive 2D map to spot themes, outliers, and document overlap.")

store = tenant_store(security)
if store.index is None:
    st.warning("No index found. Build one first.")
    st.stop()

items = store.meta["items"]
N_total = len(items)
status_pills([("Index online", True), (f"{N_total:,} chunks", True), (f"{len({x.get('source') for x in items})} sources", True)])

if N_total < 5:
    st.info("This map needs at least five chunks. Add a larger document set in Ingest & Index, then return here.")
    st.stop()

try:
    import umap
except ImportError:
    st.error("UMAP is not installed in this runtime. Install requirements.txt and restart the app.")
    st.stop()

section("Projection controls", "UMAP preserves local semantic neighborhoods; tune these settings to inspect different structure.")
max_limit = min(3000, N_total)
max_points = st.slider("Max chunks to plot (for speed)", 5, max_limit, min(800, max_limit), step=1 if max_limit < 50 else 50)
subset = items[:max_points]
N = len(subset)

embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)
embedder = Embedder(embed_model)

# UMAP controls
requested_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15, 1)
min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.01)

if st.button("Generate semantic map", type="primary", use_container_width=True):
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

st.caption("For a stable map, use at least 20 chunks. Small corpora automatically switch to random initialization.")
footer()
