import streamlit as st
import pandas as pd
import plotly.express as px

from backend.telemetry import fetch_runs
from app._bootstrap import bootstrap
bootstrap()

st.title("Latency & Observability")

rows = fetch_runs(limit=200)
if not rows:
    st.info("No runs logged yet. Ask a few questions first.")
    st.stop()

df = pd.DataFrame(rows, columns=[
    "ts_ms", "query", "top_k", "use_mmr", "retrieval_ms", "generation_ms", "total_ms", "citations"
])
df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms")

st.subheader("Recent runs")
st.dataframe(df[["ts", "query", "top_k", "use_mmr", "retrieval_ms", "generation_ms", "total_ms"]], use_container_width=True)

st.subheader("Latency distribution (total_ms)")
fig1 = px.histogram(df, x="total_ms", nbins=25, title="Total latency distribution")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Latency over time")
fig2 = px.line(df.sort_values("ts"), x="ts", y="total_ms", title="Total latency over time")
st.plotly_chart(fig2, use_container_width=True)

st.caption("In production, you’d also log token usage, cache hits, and model response times.")
