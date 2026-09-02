"""Build the reviewed 96-case grounding benchmark from deterministic public chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.contextual_chunking import build_contextual_chunks
from backend.document_parsers import parse_document
from backend.grounding_eval import validate_grounding_benchmark


BASES = [
    {
        "exact": "Its tenant-safe document service is named Atlas.",
        "paraphrase": "Atlas is Northstar's tenant-safe document service.",
        "detail": "The Aurora 3.2 platform release was published on 14 March 2026.",
        "neutral": "Northstar's document service is hosted only in Europe.",
        "contradiction": "Northstar's tenant-safe document service is named Beacon.",
        "temporal": "Aurora 3.2 was not published on 14 March 2026.",
    },
    {
        "exact": "Dense and lexical rankings are combined with Reciprocal Rank Fusion using a constant of 60.",
        "paraphrase": "Northstar fuses dense and BM25 rankings with RRF at k=60.",
        "detail": "The hybrid pipeline retrieves at least 50 candidates from each available ranking.",
        "neutral": "Northstar uses Elasticsearch as its only vector database.",
        "contradiction": "Northstar's RRF constant is 10.",
        "temporal": "MMR is applied after hybrid fusion.",
    },
    {
        "exact": "Permission filtering must happen during retrieval, never after generation.",
        "paraphrase": "Northstar enforces access permissions inside the retrieval stage.",
        "detail": "The Sentinel canary phrase is ORCHID-VAULT-29.",
        "neutral": "Sentinel encrypts every uploaded document with a customer-managed key.",
        "contradiction": "Permission filtering is performed only after answer generation.",
        "temporal": "A cross-tenant retrieval result does not block release promotion.",
    },
    {
        "exact": "Citation validity must equal 1.00.",
        "paraphrase": "Northstar requires perfect citation validity for an experiment.",
        "detail": "Retrieval p95 latency may not exceed 1.25 times the baseline.",
        "neutral": "The evaluation policy requires BLEU to exceed 0.90.",
        "contradiction": "A citation validity score of 0.50 passes the policy.",
        "temporal": "Recall and MRR may each regress by 0.20 during promotion.",
    },
    {
        "exact": "The production service exposes a health endpoint at /health.",
        "paraphrase": "Northstar reports service health through the /health endpoint.",
        "detail": "The local container port defaults to 8501.",
        "neutral": "Production telemetry is retained for exactly 30 days.",
        "contradiction": "The service health endpoint is /status-only.",
        "temporal": "The application cannot operate when Gemini is unavailable.",
    },
    {
        "exact": "No customer data was exposed, and the recovery time was 37 minutes.",
        "paraphrase": "Recovery took 37 minutes and did not expose customer data.",
        "detail": "BM25 correctly favored the exact error identifier NX-417.",
        "neutral": "The incident was caused by a compromised administrator account.",
        "contradiction": "Incident IR-2026-08 exposed customer data.",
        "temporal": "The recovery from IR-2026-08 took 90 minutes.",
    },
    {
        "exact": "Meridian issues access tokens with a 45-minute lifetime and rotates signing keys every 14 days.",
        "paraphrase": "Current Meridian tokens expire after 45 minutes.",
        "detail": "AU-4012 requires checking tenant clock skew before credentials are rotated.",
        "neutral": "Meridian access tokens use biometric authentication.",
        "contradiction": "Current Meridian access tokens last eight hours.",
        "temporal": "The current Meridian runbook does not supersede the eight-hour token procedure.",
    },
    {
        "exact": "Status: obsolete since 12 January 2025.",
        "paraphrase": "The eight-hour Meridian token procedure has been obsolete since 12 January 2025.",
        "detail": "The former Meridian procedure rotated signing keys every 90 days.",
        "neutral": "The legacy runbook requires hardware security keys.",
        "contradiction": "The legacy Meridian runbook is the current procedure.",
        "temporal": "Operators should continue using the obsolete eight-hour token procedure.",
    },
    {
        "exact": "Incident ID ID-2026-014.",
        "paraphrase": "Meridian's clock-skew incident was identified as ID-2026-014.",
        "detail": "The customer gateway drifted seven minutes from network time.",
        "neutral": "The incident originated from a stolen signing key.",
        "contradiction": "The Meridian clock-skew incident was ID-2025-999.",
        "temporal": "Rotating signing keys resolved the AU-4012 incident.",
    },
    {
        "exact": "Ledger generates invoices on the third calendar day of each month.",
        "paraphrase": "Current Ledger invoices are created on day three of the calendar month.",
        "detail": "Invoice evidence has an official retention period of seven years.",
        "neutral": "Ledger invoices are paid automatically with cryptocurrency.",
        "contradiction": "Ledger currently generates invoices on the fifth business day.",
        "temporal": "Invoice evidence is retained for only two years under the current policy.",
    },
    {
        "exact": "Status: obsolete since July 2024.",
        "paraphrase": "Ledger's fifth-business-day invoice policy was retired in July 2024.",
        "detail": "The retired policy used BI-2207 for any manual change.",
        "neutral": "The retired policy required payment in euros.",
        "contradiction": "The fifth-business-day Ledger policy is still current.",
        "temporal": "The legacy two-year evidence retention rule remains valid.",
    },
    {
        "exact": "Incident ID FI-2026-031.",
        "paraphrase": "FI-2026-031 was the Ledger duplicated-metering incident.",
        "detail": "The duplicated batch affected 84 test accounts.",
        "neutral": "The billing incident affected ten thousand production customers.",
        "contradiction": "Customers were charged because of FI-2026-031.",
        "temporal": "Finance Operations released the incorrect invoices before reversing the batch.",
    },
    {
        "exact": "Beacon combines dense retrieval and BM25 using Reciprocal Rank Fusion with constant 60.",
        "paraphrase": "Current Beacon search fuses dense and lexical rankings with RRF k=60.",
        "detail": "Production refreshes the lexical index every 20 minutes.",
        "neutral": "Beacon refreshes its index only when a user requests it.",
        "contradiction": "Beacon's current RRF constant is 100.",
        "temporal": "SE-3104 indicates an embedding timeout in the current guide.",
    },
    {
        "exact": "Status: obsolete since November 2025.",
        "paraphrase": "Beacon's dense-only ranking guide became obsolete in November 2025.",
        "detail": "The legacy Beacon system refreshed vectors every six hours.",
        "neutral": "The legacy system used a graph database for ranking.",
        "contradiction": "The dense-only Beacon guide is the current production procedure.",
        "temporal": "Operators should map SE-3104 to an embedding timeout today.",
    },
    {
        "exact": "Incident ID SR-2026-044.",
        "paraphrase": "Beacon's stale BM25 scheduler incident was SR-2026-044.",
        "detail": "The BM25 index remained stale for 71 minutes.",
        "neutral": "A database disk failure caused the Beacon incident.",
        "contradiction": "Dense retrieval also failed during SR-2026-044.",
        "temporal": "Replaying the lexical refresh did not fix SE-3104.",
    },
    {
        "exact": "Harbor targets a 15-minute recovery point objective and a 60-minute recovery time objective for tier-one data.",
        "paraphrase": "Tier-one Harbor data has a 15-minute RPO and a 60-minute RTO.",
        "detail": "Harbor copies backups to us-east-2 and verifies them every Sunday.",
        "neutral": "Harbor stores its only backup in us-west-1.",
        "contradiction": "Tier-one Harbor data has a 24-hour recovery point objective.",
        "temporal": "BK-1500 starts an unaudited restore workflow under the current standard.",
    },
]

VARIANTS = [
    ("exact_support", "exact", "supported"),
    ("paraphrase_support", "paraphrase", "supported"),
    ("identifier_numeric", "detail", "supported"),
    ("unsupported", "neutral", "neutral"),
    ("hard_contradiction", "contradiction", "contradiction"),
    ("temporal_negation", "temporal", "contradiction"),
]


def build_chunks():
    chunks = []
    for path in sorted((ROOT / "data" / "public_demo").glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    return chunks


def main() -> None:
    chunks = build_chunks()
    if len(chunks) < len(BASES):
        raise RuntimeError("The public corpus no longer contains the required deterministic chunks.")
    cases = []
    for base_index, (base, chunk) in enumerate(zip(BASES, chunks)):
        for variant_index, (category, field, label) in enumerate(VARIANTS):
            cases.append({
                "case_id": f"gnd_{base_index + 1:02d}_{variant_index + 1:02d}",
                "claim": base[field], "evidence_chunk_ids": [chunk.chunk_id],
                "label": label, "category": category,
                "split": "calibration" if (base_index + variant_index) % 2 == 0 else "heldout",
                "expected_source_versions": [chunk.source_version],
            })
    validate_grounding_benchmark(cases, [chunk.chunk_id for chunk in chunks])
    target = ROOT / "data" / "grounding_benchmark.json"
    target.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} verified grounding cases to {target}.")


if __name__ == "__main__":
    main()
