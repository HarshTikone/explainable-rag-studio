import streamlit as st
import pandas as pd
import plotly.express as px

from backend.telemetry import fetch_runs
from app._bootstrap import bootstrap
bootstrap()
from app.ui import configure_page, footer, page_header, section, status_pills
from app.security_ui import security_context

configure_page("Observability", "↗")
security = security_context("security:read")
page_header("Performance telemetry", "Measure the path from question to answer.", "Track retrieval, generation, and end-to-end latency across recent queries to expose bottlenecks and regressions.")

rows = fetch_runs(limit=200)
if not rows:
    st.info("No runs logged yet. Ask a few questions first.")
    st.stop()

df = pd.DataFrame(rows, columns=[
    "ts_ms", "query", "top_k", "use_mmr", "retrieval_ms", "generation_ms", "total_ms", "citations",
    "verification_ms", "grounding_status", "accepted_claims", "rejected_claims", "conflict_count",
])
df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms")
status_pills([("Telemetry online", True), (f"{len(df)} recent runs", True), ("SQLite history", True)])

section("Service level snapshot", "A quick view of typical and tail latency across recorded runs.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Median", f"{df['total_ms'].median():.0f} ms")
c2.metric("P95", f"{df['total_ms'].quantile(.95):.0f} ms")
c3.metric("Avg retrieval", f"{df['retrieval_ms'].mean():.0f} ms")
c4.metric("Avg verification", f"{df['verification_ms'].mean():.0f} ms")

section("Recent traces", "Query-level timings and retrieval configuration.")
st.dataframe(df[["ts", "query", "top_k", "use_mmr", "retrieval_ms", "generation_ms", "verification_ms", "total_ms", "grounding_status", "accepted_claims", "rejected_claims", "conflict_count"]], use_container_width=True)

section("Latency distribution")
fig1 = px.histogram(df, x="total_ms", nbins=25, title="Total latency distribution")
st.plotly_chart(fig1, use_container_width=True)

section("Latency trend")
fig2 = px.line(df.sort_values("ts"), x="ts", y="total_ms", title="Total latency over time")
st.plotly_chart(fig2, use_container_width=True)

st.caption("Grounding telemetry records verification latency, strict-filter decisions, and conflicts without changing reviewer labels automatically.")
footer()
