import pandas as pd
import streamlit as st

from app._bootstrap import bootstrap
bootstrap()
from app.ui import configure_page, footer, page_header, section, status_pills
from backend.config import SETTINGS
from backend.review_registry import ReviewRegistry
from app.security_ui import security_context, tenant_dir


configure_page("Grounding review", "⚖")
security = security_context("reviews:write")
page_header(
    "Human review queue",
    "Resolve the claims the verifier is least certain about.",
    "Compare cited and conflicting passages, record a durable label, and export review data without changing live thresholds.",
)
if SETTINGS.platform_mode == "postgres":
    from app.production_ui import render_reviews
    render_reviews()
    st.stop()
registry = ReviewRegistry(str(tenant_dir(security) / "reviews.db"))
status = st.selectbox("Case status", ["open", "resolved", "all"])
cases = registry.list_cases(status=status, limit=200)
status_pills([("SQLite queue", True), (f"{len(cases)} {status} cases", bool(cases)), ("Local reviewer", True)])

if not cases:
    st.info("No review cases match this filter. Low-confidence and disputed claims appear here automatically.")
    footer()
    st.stop()

section("Review inventory", "Cases are deduplicated by claim, evidence IDs, and verifier configuration.")
st.dataframe(pd.DataFrame([{
    "case_id": case["case_id"], "status": case["status"], "reason": case["reason"],
    "verdict": case["verdict"], "claim": case["claim_text"], "updated_at": case["updated_at"],
} for case in cases]), use_container_width=True, hide_index=True)

selected_id = st.selectbox("Review case", [case["case_id"] for case in cases])
selected = registry.get_case(selected_id)
payload = selected["payload"]
section("Claim and evidence", "The evidence excerpts are the bounded premises used by the verifier.")
with st.container(border=True):
    st.markdown(f"**{payload['verdict']}** · {payload['text']}")
    st.caption(f"Entailment {payload['entailment_score']:.3f} · Contradiction {payload['contradiction_score']:.3f} · Reasons: {', '.join(payload['reason_codes'])}")
evidence_rows = [{
    "chunk_id": item["chunk_id"], "cited": item["cited"], "source": item["source"],
    "entailment": round(item["entailment_score"], 4),
    "contradiction": round(item["contradiction_score"], 4), "excerpt": item["excerpt"],
} for item in payload.get("evidence", [])]
st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

if selected["status"] == "open":
    with st.form("review-decision"):
        decision = st.radio("Human verdict", ["supported", "unsupported", "contradicted"], horizontal=True)
        reviewer = st.text_input("Reviewer", "local")
        notes = st.text_area("Notes", max_chars=2000)
        if st.form_submit_button("Save decision", type="primary", use_container_width=True):
            registry.decide(selected_id, decision, reviewer, notes)
            st.success("Decision saved. Reviewer labels do not automatically change production behavior.")
            st.rerun()
else:
    st.dataframe(pd.DataFrame(selected.get("decisions", [])), use_container_width=True, hide_index=True)

st.download_button(
    "Export labeled review JSONL", registry.export_jsonl(status="all"),
    file_name="grounding_reviews.jsonl", mime="application/x-ndjson", use_container_width=True,
)
footer()
