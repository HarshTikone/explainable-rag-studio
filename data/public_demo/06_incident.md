# Northstar Incident Review

Incident IR-2026-08 occurred when an obsolete support article ranked above the current runbook for query NX-417. Dense-only retrieval matched the broad topic, while BM25 correctly favored the exact error identifier. The remediation combined dense and lexical rankings with Reciprocal Rank Fusion and added a freshness field for future filtering. The incident owner was the Search Reliability team. No customer data was exposed, and the recovery time was 37 minutes.
