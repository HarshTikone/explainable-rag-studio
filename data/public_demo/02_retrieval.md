# Northstar Retrieval Architecture

Northstar's dense retriever uses the embedding model all-MiniLM-L6-v2. The lexical retriever uses BM25 and is especially useful for exact identifiers such as error code NX-417. Dense and lexical rankings are combined with Reciprocal Rank Fusion using a constant of 60. The hybrid pipeline retrieves at least 50 candidates from each available ranking and returns the configured top-k results. MMR is available only for the dense_mmr strategy and is not applied after hybrid fusion.
