import json
from collections import Counter
from pathlib import Path

import streamlit as st

from app._bootstrap import bootstrap

bootstrap()

from app.ui import configure_page, footer, metric_card, page_header, section, status_pills


configure_page("Results", "◫")
page_header(
    "Retained evidence",
    "What the project can prove — and what it cannot yet claim.",
    "These numbers come from committed benchmark and release artifacts rather than a live evaluation job on the free hosted service.",
)

root = Path(__file__).resolve().parents[2]
quality_path = root / "docs" / "benchmarks" / "quality-gate-reference.json"
benchmark_path = root / "data" / "public_demo_benchmark.json"
quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else []
gate = quality.get("quality_gate", quality)
decisions = {name: gate.get(name, {}) for name in ("retrieval", "grounding", "runtime", "security")}
overall = bool(gate.get("overall", {}).get("promoted", False))
status_pills([
    ("Evidence artifact loaded", bool(quality)),
    ("Overall promoted" if overall else "Overall honestly rejected", overall),
    (f"{len(benchmark)} benchmark questions", len(benchmark) >= 100),
])

categories = Counter(item.get("category", "unknown") for item in benchmark)
unanswerable = categories.get("unanswerable", 0)
columns = st.columns(4)
cards = [
    ("Benchmark", str(len(benchmark)), "Versioned questions"),
    ("Categories", str(len(categories)), "Failure slices"),
    ("Unanswerable", f"{(100 * unanswerable / max(1, len(benchmark))):.1f}%", "Adversarial coverage"),
    ("Hosted retriever", "BM25", "512 MB public profile"),
]
for column, card in zip(columns, cards):
    with column:
        metric_card(*card)

section("Release decisions", "Rejected gates remain visible; the UI never converts them into success claims.")
for name, decision in decisions.items():
    if not decision:
        continue
    promoted = bool(decision.get("promoted", decision.get("decision") == "promoted"))
    with st.container(border=True):
        st.markdown(f"### {name.title()} — {'Promoted' if promoted else 'Rejected'}")
        retained = decision.get("retained_strategy") or decision.get("retained_policy")
        if retained:
            st.caption(f"Retained: {retained}")
        measured = decision.get("gates", {}).get("measured", {})
        if measured:
            st.json(measured, expanded=False)

section("Two deliberate profiles", "The portfolio demo and the full reliability platform solve different operating constraints.")
st.markdown("""
| Capability | Free hosted demo | Full platform |
|---|---|---|
| Retrieval | BM25 | Dense + BM25 + reranking |
| Answer selection | Groq-assisted verbatim evidence with local fallback | Structured provider or extractive drafts |
| Verification | Exact evidence + deterministic guards | Pinned local NLI + deterministic guards |
| Purpose | Stable public walkthrough on 512 MB | Evaluation, security, and production evidence |
""")

section("Known limitations", "These are release constraints, not hidden footnotes.")
limitations = [
    "Render's free instance can take roughly a minute to wake after inactivity.",
    "Free-tier Groq is allowance-limited; quota, invalid selections, and provider failures use the local exact-evidence fallback.",
    "The public profile intentionally hides ingestion and administrative workspaces.",
]
if overall:
    limitations.append(
        "The promoted decision applies to the committed release evidence; the hosted 512 MB profile intentionally uses BM25 and deterministic verification instead of the full hybrid/NLI runtime."
    )
else:
    limitations.append(
        "The overall production-quality claim remains rejected until every required gate is promoted for the same release SHA."
    )
st.markdown("\n".join(f"- {item}" for item in limitations))
footer()
