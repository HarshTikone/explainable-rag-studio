# Vault Storage — Current Index Policy

Status: current. Vault writes FAISS vectors and aligned metadata atomically under one index generation. Storage code ST-7305 means vector and metadata counts differ. A mismatched generation is never activated and must be rebuilt.
