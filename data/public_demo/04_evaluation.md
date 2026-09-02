# Northstar Evaluation Policy

Every experiment records a corpus fingerprint, benchmark fingerprint, Git revision, retrieval strategy, embedding model, and chunk settings. The benchmark includes exact-term, identifier, paraphrase, multi-hop, and unanswerable questions. The primary retrieval measures are Recall at k and Mean Reciprocal Rank. Citation validity must equal 1.00. A candidate must improve Recall or MRR, neither metric may regress by more than 0.01, and retrieval p95 latency may not exceed 1.25 times the baseline.
