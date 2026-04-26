# Abot — Portfolio RAG Chatbot 🤖💬

Abot is a conversational AI assistant embedded in Anurag's portfolio website. Visitors can ask any question about Anurag's background, skills, projects, work experience, and education — and get instant, accurate answers powered by the actual content of the portfolio. It is built using RAG (Retrieval-Augmented Generation), meaning the chatbot retrieves relevant information from a knowledge base before answering, rather than relying on general LLM knowledge.

## 🎯 Why RAG?

A plain LLM doesn't know anything about this specific person. RAG solves this:

* Pre-process and store portfolio content in a searchable vector database
* When a question arrives, retrieve only the relevant pieces
* Pass those pieces to the LLM as context
* The LLM answers using real facts — not hallucinations

## 🏗️ System Architecture

The system has two phases:

### Phase 1: Ingestion (run once)
1. Load all `.md` files from the `knowledge-base/` folder
2. Split each document into overlapping 500-character chunks (200-character overlap)
3. Convert each chunk to a 1024-dimension vector using **Qwen3-Embedding-0.6B**
4. Store all vectors + text in **ChromaDB** on disk at `vector_db/`

### Phase 2: Runtime (every user message)
1. User types a question in the Gradio chatbot UI
2. Question (combined with prior conversation history) is embedded into a vector
3. ChromaDB finds the top-10 most similar chunks via cosine similarity
4. Chunks are formatted into a context string
5. System prompt + context + question are sent to the LLM
6. Answer is displayed in the chat

### Phase 3: Evaluation (developer only)
* **Retrieval eval:** Checks if the right chunks were retrieved using MRR, nDCG, and keyword coverage — no LLM required
* **Answer eval:** Generates a full RAG answer, then uses qwen3:4b as an LLM judge to score accuracy, completeness, and relevance


## 💡 Key Implementation Details

### Context-Aware Retrieval
Prior user messages are combined with the current question before retrieval. This ensures follow-up questions like "What about his internship?" still find the right chunks by carrying full context into the vector search.

### Collection Name Alignment
The ingest script writes into a ChromaDB collection named `"docs"`. The answer script must specify the same collection name when reading — failing to do so causes the retriever to query an empty default collection and hallucinate.

### LLM Preprocessing with Overlap
Chunks use 200-character overlap — if an answer spans the boundary of two chunks, overlap ensures the relevant text appears in at least one complete chunk without losing meaning.

### Chroma DB Reset Pattern
On re-ingestion, the entire `vector_db/` folder is deleted with `shutil.rmtree()` before rebuilding. This prevents dimension mismatch errors when switching embedding models. Using `delete_collection()` alone is insufficient as it leaves files on disk.

## 📊 Evaluation Metrics

### Retrieval Metrics

* MRR :How high up is the relevant chunk in results? ; ≥ 0.9 
* nDCG : Overall ranking quality ; ≥ 0.9 
* Keyword Coverage: % of expected keywords in top-10 chunks; ≥ 90% 

### Answer Quality (1–5 scale, scored by qwen3:4b judge)

* Accuracy : Answer is perfectly factually correct — any wrong fact = score 1 
* Completeness : Nothing from the reference answer is missing 
* Relevance : Answer stays on-topic with no irrelevant noise
  
### Test Question Categories

* direct_fact | Can RAG find a simple stated fact? (Easy) |
* temporal | Can it reason over dates and time gaps? (Hard) |
* comparative | Can it differentiate between two similar things? (Medium) |
* numerical | Can it aggregate numbers across multiple documents? (Hard) |
* holistic | Can it synthesize across the entire knowledge base? (Hard) |
* relationship | Can it understand sequential/causal dependencies? (Medium) |
## 🚀 Deployment
- Deployed on **HuggingFace Spaces** as a Docker container
- The vector database is built at Docker image build time by running `ingest.py` inside the Dockerfile — the `vector_db/` folder does not need to be committed to the repository
- Exposed as a **FastAPI REST API** with a `POST /chat` endpoint
- Frontend portfolio website calls this API and displays answers in the chat UI
- CORS configured to allow the portfolio domain and localhost for development

## ⚙️ Tech Stack

* **Embedding Model:** Qwen/Qwen3-Embedding-0.6B (HuggingFace, runs locally)
* **Vector Database:** ChromaDB (persistent, disk-based)
* **LLM:** Gemini 2.5 Flash (via LangChain Google GenAI integration)
* **API:** FastAPI + uvicorn
* **RAG Framework:** LangChain (Chroma retriever, document loaders, text splitters)
* **Evaluation LLM:** qwen3:4b via Ollama (local, zero cost)
* **Deployment:** HuggingFace Spaces (Docker)