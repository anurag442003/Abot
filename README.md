# 🤖 Portfolio RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for a personal portfolio website. Visitors can ask natural language questions about skills, projects, work experience, and education — and get accurate, grounded answers pulled directly from a structured knowledge base.

Built with two complete pipelines: a **local pipeline** (Ollama + qwen3:4b, fully offline) and a **Gemini pipeline** (Google Gemini 2.5 Flash, cloud-based with higher quality). Both share the same evaluation framework.

---

## 📁 Project Structure

```
week5/
├── knowledge-base/              # Source documents (Markdown)
│   ├── AboutMe/
│   ├── Projects/
│   └── Contact/
│
├── implementation/              # RAG answer pipelines
│   ├── answer_1.py              # Ollama + LangChain Chroma (basic)
│   ├── answer_gemini.py         # Gemini + LangChain Chroma (basic)
│   └── answer_2.py              # Ollama + raw ChromaDB + rerank + rewrite (advanced)
│
├── pro_implementation/          # Advanced Gemini pipeline
│   └── answer_gemini.py         # Gemini + raw ChromaDB + rerank + rewrite
│
├── ingest/                      # Ingestion pipelines
│   ├── ingest.py                # LangChain RecursiveTextSplitter → raw ChromaDB
│   ├── ingest_gemini.py         # LangChain splitter → LangChain Chroma (basic)
│   └── ingest_gemini_llm.py     # LLM-based chunking (headline + summary + text)
│
├── evaluation/                  # Evaluation framework
│   ├── test.py                  # TestQuestion model + loader
│   ├── tests.jsonl              # 10 test questions with reference answers
│   ├── eval_qwen.py             # Eval using Ollama judge
│   ├── eval_gemini.py           # Eval using Gemini judge
│   └── eval_gemini_unranked.py  # Eval using Ollama judge + fetch_context_unranked (fast)     
│
├── vector_db/                   # ChromaDB vector store
│
├── app.py                       # Gradio chatbot UI
└── evaluator.py                 # Gradio evaluation dashboard
```

---

## How It Works

### The RAG Pipeline

Every user question goes through up to four stages depending on which pipeline is used:

```
User Question
      │
      ▼
1. Query Rewriting        (LLM rewrites question to be more KB-friendly)
      │
      ▼
2. Dual Retrieval         (embed both original + rewritten → ChromaDB cosine search)
      │
      ▼
3. Reranking              (LLM reorders chunks by relevance)
      │
      ▼
4. Answer Generation      (LLM answers using top-K chunks as context)
      │
      ▼
Final Answer
```


### Retrieval

ChromaDB stores document chunks as 1024-dimensional vectors using `Qwen/Qwen3-Embedding-0.6B`. When a question comes in, it is embedded with the same model and compared against all stored chunks using **cosine similarity** — chunks whose meaning most closely matches the question are returned.

### Chunking Strategies

Two strategies are available:

**RecursiveCharacterTextSplitter** (`ingest.py`) — splits documents mechanically by character count (chunk_size=500, chunk_overlap=200). Fast, no API calls.

**LLM-based chunking** (`ingest_gemini_llm.py`) — sends each document to Gemini, which returns structured chunks each with a `headline`, `summary`, and `original_text`. Results in semantically coherent chunks that retrieve better, at the cost of API calls during ingestion.

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.com) (for local pipeline only)
- Google Gemini API key (for Gemini pipeline only)

### Install dependencies

```bash
uv sync
# or
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Pull Ollama model (local pipeline only)

```bash
ollama pull qwen3:4b
```

---

## Quickstart

### Step 1 — Build the vector database

**Option A — Fast mechanical chunking (recommended for local pipeline):**
```bash
uv run implementation/ingest.py
```
This creates `vector_db/` using `RecursiveCharacterTextSplitter` and raw ChromaDB.

**Option B — LLM-based chunking (recommended for Gemini pipeline, higher quality):**
```bash
uv run ingest_gemini_llm.py
```
This creates `vector_db/` with semantically structured chunks (requires Gemini API key).

### Step 2 — Run the chatbot

```bash
uv run app.py
```

Opens a Gradio UI at `http://localhost:7860`. The right panel shows the retrieved context chunks for each answer.

### Step 3 — Run the evaluation dashboard

```bash
uv run evaluator.py
```

Opens a Gradio evaluation dashboard at `http://localhost:7860` with two tabs: Retrieval Evaluation and Answer Evaluation.

---

## Evaluation

### Test Suite

10 hand-crafted test questions in `evaluation/tests.jsonl` covering:

| Category | Description | Example |
|---|---|---|
| `direct_fact` | Single-chunk factual retrieval | GPA, CLI commands |
| `temporal` | Date reasoning across chunks | Gap between jobs |
| `comparative` | Comparing two projects | CrewAI vs AutoGen |
| `numerical` | Aggregation across documents | Total APIs used |
| `relationship` | Logical inference | IL vs RL relationship |
| `holistic` | Multi-document reasoning | Best project for a use case |

### Metrics

**Retrieval metrics** (no LLM needed — pure keyword matching):
- **MRR** (Mean Reciprocal Rank) — how early the first relevant chunk appears. Green ≥ 0.9
- **nDCG** (Normalized Discounted Cumulative Gain) — quality of full ranking. Green ≥ 0.9
- **Keyword Coverage** — % of expected keywords found in top-K chunks. Green ≥ 90%

**Answer metrics** (LLM-as-a-judge, scored 1–5):
- **Accuracy** — factual correctness vs reference answer. Green ≥ 4.5
- **Completeness** — all key information included. Green ≥ 4.5
- **Relevance** — no off-topic content. Green ≥ 4.5


### Which eval file to use?

| File | Answer pipeline | Judge model | 
|---|---|---|
| `eval_qwen.py` | Ollama full pipeline (rerank+rewrite) | qwen3:4b local | 
| `eval_gemini_unranked.py` | Ollama fast (unranked only) | Gemini 2.5 Flash | 
| `eval_gemini.py` | Gemini full pipeline | Gemini 2.5 Flash |

> **Note:** Retrieval eval scores are not affected by reranking or query rewriting — those only affect answer quality. Retrieval scores reflect knowledge base and embedding model quality.

---

## Pipeline Variants

### Local Pipeline (Offline)

| File | Model | ChromaDB client | Chunking | Rerank/Rewrite |
|---|---|---|---|---|
| `answer_qwen.py` | qwen3:4b | LangChain Chroma | Recursive | ❌ |
| `answer_qwen_adv.py` | qwen3:4b | Chroma | Recursive | ✅ |

### Gemini Pipeline (Cloud)

| File | Model | ChromaDB client | Chunking | Rerank/Rewrite |
|---|---|---|---|---|
| `answer_gemini.py` | gemini-2.5-flash | LangChain Chroma | Recursive | ❌ |
| `answer_gemini_adv.py` | gemini-2.5-flash | Chroma | Recursive | ✅ |

---


##  Architecture Decisions

**Why raw ChromaDB over LangChain Chroma?**
LangChain's Chroma wrapper uses a default collection name (`langchain`) and stores data in a format that's harder to inspect directly. The raw `PersistentClient` gives full control over collection names, IDs, and metadata — which matters when `ingest.py` and `answer.py` need to share the same DB reliably.

**Why `Qwen3-Embedding-0.6B` for embeddings?**
It produces 1024-dimensional vectors with strong multilingual and technical text understanding, runs fully locally with no API key required, and is small enough (~600MB) to load on a CPU.

**Why `fast=True` in evaluation but not production?**
Evaluation measures knowledge base and LLM baseline quality. Reranking and query rewriting improve results for real users (ambiguous phrasing, multi-turn context) but add 2–3 extra LLM calls per question. Keeping eval fast means 10 tests run in ~10 minutes instead of 75+.

---

## Knowledge Base

Documents live in `knowledge-base/` as Markdown files, organized by type:

```
knowledge-base/
├── AboutMe/      # Education, skills, work experience
├── Projects/     # Individual project descriptions
└── Contact/      # Contact information
```

To add new content, drop a `.md` file in the appropriate folder and re-run `ingest.py`.
