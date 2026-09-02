# Beacon Incident SE-3104

Incident ID SR-2026-044. A failed scheduler left the BM25 index stale for 71 minutes while dense retrieval remained healthy. Exact product identifiers were missed. Replaying the lexical refresh fixed SE-3104, and monitoring now compares dense and lexical index timestamps.
