# Explainable RAG Studio

> **An end-to-end, recruiter-ready Retrieval-Augmented Generation (RAG) system with explainability, evaluation, and latency observability.**

This project demonstrates how to build a **production-style RAG Document Q&A system** using modern NLP techniques. It is designed not only to *work*, but to clearly **explain every step of the RAG pipeline** to recruiters, clients, or non-technical stakeholders through an interactive UI.

---

## 🎥 Demo Video

▶️ **Project Walkthrough (3–5 min):**
[https://youtu.be/REPLACE_WITH_DEMO_LINK]([https://youtu.be/REPLACE_WITH_DEMO_LINK](https://app.govideolink.com/videos/0DSg0V06vaOuG9vvYxpv/?utm_source=direct&utm_medium=invite_link))

This short demo walks through:

* What problem RAG solves
* PDF ingestion and FAISS indexing
* Ask & Explain (retrieval + citations)
* Embedding visualization (UMAP)
* Evaluation and latency dashboard

> 📌 *Tip for reviewers:* Watch this video first to understand the system end-to-end in minutes.

---

## 🚀 What This Project Does

* Upload PDF documents
* Split them into **token-based chunks (300–500 tokens)**
* Convert chunks into **vector embeddings**
* Store and search them efficiently using **FAISS**
* Answer user questions using **Gemini (LLM)** grounded strictly in retrieved context
* Provide **2–3 citations per answer** for traceability
* Visualize retrieval, embeddings, and similarity scores
* Evaluate system accuracy using a reproducible **JSON-based benchmark**
* Track **latency and performance metrics** for each query

This mirrors how real-world RAG systems are built and evaluated in industry.

---

## 🧠 Why RAG?

Large Language Models (LLMs) are powerful, but they **hallucinate** when asked about private or unseen data. RAG solves this by:

1. Retrieving relevant document chunks
2. Injecting only those chunks into the LLM prompt
3. Generating answers **grounded in sources**

This system enforces grounding and provides citations so answers are **verifiable and trustworthy**.

---

## 🏗️ System Architecture

```
PDF Documents
      │
      ▼
Document Loader (PDF → Text)
      │
      ▼
Token-based Chunking (300–500 tokens, overlap)
      │
      ▼
Embedding Model (SentenceTransformers)
      │
      ▼
FAISS Vector Index (Cosine Similarity)
      │
      ▼
Retriever (Top-K / MMR)
      │
      ▼
Prompt Construction (Context + Rules)
      │
      ▼
Gemini LLM
      │
      ▼
Answer + Citations + Metrics
```

---

## 🖥️ User Interface (Streamlit)

The project includes a **multi-page interactive Streamlit app**:

### 1️⃣ What is RAG?

* Client-friendly explanation of LLMs and hallucinations
* Step-by-step overview of the RAG pipeline

### 2️⃣ Ingest & Index

* Upload PDFs
* Configure chunk size and overlap
* Build FAISS index
* Preview chunks and token counts

### 3️⃣ Ask & Explain

* Ask natural language questions
* View retrieved chunks and similarity scores
* See the exact context sent to the LLM
* Answers returned with **2–3 citations**

### 4️⃣ Embedding Explorer

* 2D visualization of chunk embeddings using **UMAP**
* Shows semantic clustering of document content

### 5️⃣ Evaluation

* Upload a JSON evaluation set
* Measure accuracy automatically
* Inspect failure cases

### 6️⃣ Latency Dashboard

* Track retrieval time, generation time, total latency
* View performance trends across queries

---

## 📊 Evaluation Methodology

The system supports **reproducible evaluation** using a JSON file:

```json
[
  {"question": "What is the purpose of the document?", "expected": "purpose"},
  {"question": "What technology is used?", "expected": "faiss"}
]
```

* Each question is asked automatically
* An answer is considered correct if it contains the expected phrase
* Accuracy is computed as:

```
accuracy = correct_answers / total_questions
```

Evaluation results are saved to disk and displayed in the UI.

---

## ⚡ Performance & Latency

For every query, the system logs:

* Retrieval latency (FAISS)
* Generation latency (Gemini)
* Total end-to-end latency

This allows comparison between:

* Baseline vs tuned retrieval
* Different Top-K values
* MMR vs standard similarity search

---

## 🛠️ Tech Stack

**Backend / ML**

* Python
* FAISS (vector database)
* SentenceTransformers (embeddings)
* Gemini API (LLM)

**Frontend**

* Streamlit
* Plotly (visualizations)

**Evaluation & Ops**

* JSON-based benchmarks
* SQLite logging
* Latency tracking

---

## 📁 Project Structure

```
rag-studio/
│
├── app/                # Streamlit UI
│   ├── Home.py
│   └── pages/
│       ├── 1_What_is_RAG.py
│       ├── 2_Ingest_and_Index.py
│       ├── 3_Ask_and_Explain.py
│       ├── 4_Embedding_Explorer.py
│       ├── 5_Evaluation.py
│       └── 6_Latency_Dashboard.py
│
├── backend/            # Core RAG logic
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── qa.py
│   └── eval.py
│
├── data/               # Input PDFs
├── index/              # FAISS index (gitignored)
├── outputs/            # Logs & evaluation reports
├── requirements.txt
├── README.md
└── .env.example
```

---

## ▶️ How to Run Locally

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Add API key
cp .env.example .env
# Add GEMINI_API_KEY=your_key_here

# Run app
streamlit run app/Home.py
```

---

## 🔐 Security & Best Practices

* `.env` is gitignored (API keys never committed)
* FAISS index is built locally (not stored in repo)
* System gracefully falls back to extractive mode if LLM key is missing

---

## 💼 Why This Project Matters

This project demonstrates:

* Deep understanding of **RAG architectures**
* Strong **ML engineering discipline** (evaluation, latency, explainability)
* Ability to **explain complex systems clearly**
* Production-minded design with graceful fallbacks

It is intentionally built to be **interview-demo ready**.

---

## 📌 Future Improvements

* Hybrid retrieval (BM25 + vectors)
* Cross-encoder reranking
* Citation correctness scoring
* LLM-as-judge evaluation
* Cloud deployment (Docker + API)

---

**Author:** Harsh Mahesh Tikone
**Focus:** AI / ML Engineering, RAG Systems, Applied LLMs
