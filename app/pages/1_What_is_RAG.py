import streamlit as st
from app._bootstrap import bootstrap
bootstrap()

st.title("What is RAG? (Client-Friendly Explanation)")

st.subheader("What is an LLM?")
st.write("""
A Large Language Model (LLM) generates text by predicting the next token.
It can be very fluent, but it does not automatically “know” your private documents.
""")

st.subheader("Why do LLMs hallucinate?")
st.write("""
If an LLM is asked a question without the right information, it may guess.
This can produce confident but incorrect answers.
""")

st.subheader("What is RAG (Retrieval-Augmented Generation)?")
st.write("""
RAG reduces hallucinations by:
1) Retrieving relevant document chunks for the question
2) Feeding only that context into the LLM
3) Generating an answer **grounded in sources** with citations
""")

st.subheader("RAG pipeline overview")
st.markdown("""
**Ingest**
- PDF → text
- text → token-based chunks (300–500 tokens)

**Index**
- chunks → embeddings (vectors)
- vectors → FAISS (fast similarity search)

**Ask**
- question → embedding
- retrieve top-k chunks
- LLM answers using retrieved context + citations
""")

st.success("Next: Go to “Ingest & Index” to build your document index.")
