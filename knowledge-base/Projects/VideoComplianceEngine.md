# Video Compliance AI 🎬⚖️

Video Compliance AI is an end-to-end agentic system that automatically audits YouTube videos for brand and regulatory compliance. Give it a YouTube URL and it returns a detailed legal audit report — flagging FTC violations, misleading claims, missing disclosures (#ad, #sponsored), and more. No human needs to watch the video.

Built on Azure with LangGraph agent orchestration, Azure Video Indexer for multimodal ingestion, Azure AI Search as the RAG knowledge base, and deployed as a production FastAPI service with full telemetry.

## 🎯 Problem It Solves

* Brand teams manually review hundreds of influencer videos for regulatory compliance — slow, inconsistent, and expensive
* FTC regulations require specific disclosures in exact locations — missing them is a legal risk
* This system automates the full pipeline: ingest a video URL → extract speech + on-screen text → retrieve relevant rules → LLM reasons over evidence → structured audit report

## 🏗️ System Architecture — 5 Layers

### Layer 1: Data Ingestion (`video_indexer.py`)
* **yt-dlp** downloads the YouTube video to a local `.mp4` file
* **Azure Video Indexer (AVI)** receives the file and performs **Speech-to-Text** (transcript) and **OCR** (on-screen text recognition) simultaneously — this is multimodal ingestion
* AVI returns structured JSON with timestamped transcript lines and OCR frames
* The service polls AVI every 30 seconds until processing completes — the **polling pattern for async jobs**
* Authentication uses **Azure DefaultAzureCredential** — two-step: ARM token → VI account token
### Layer 2: Knowledge Base (`index_documents.py` + Azure AI Search)
* **PyPDFLoader** reads the FTC Disclosures guide and YouTube Ad Specifications PDFs
* **RecursiveCharacterTextSplitter** cuts them into 1000-character chunks with 200-character overlap
* **AzureOpenAIEmbeddings** (text-embedding-3-small) converts each chunk to a 1536-dimensional vector
* All vectors stored in **Azure AI Search** — supports both keyword and semantic (vector) search
* This is a one-time offline step — run `index_documents.py` once to populate the knowledge base

### Layer 3: Orchestration (`workflow.py` + LangGraph)
* **LangGraph** builds a Directed Acyclic Graph (DAG) — a fixed ordered pipeline
* Two nodes: `indexer` (video ingestion) and `auditor` (RAG + LLM analysis)
* All nodes share a single **`VideoAuditState` TypedDict** — like a shared whiteboard
* `operator.add` on list fields allows multiple nodes to append violations without overwriting each other

### Layer 4: Reasoning (`nodes.py` — `audit_content_node`)
* **RAG Retrieval:** Transcript + OCR combined into a query → `similarity_search(k=3)` retrieves top 3 rule chunks from Azure AI Search
* **Prompt Construction:** Retrieved rules injected into the system prompt as context — grounds the LLM in the actual rulebook
* **LLM Call:** AzureChatOpenAI at temperature=0 (deterministic, critical for compliance) reasons over evidence + rules
* **Structured Output:** LLM returns strict JSON — `compliance_results`, `status`, `final_report`. Regex strips markdown fences before parsing

### Layer 5: Deployment & Observability (`server.py` + `telemetry.py`)
* **FastAPI** exposes `POST /audit` with Pydantic models validating both request and response schemas
* **Azure Monitor + OpenTelemetry** auto-instruments every API call — captures latency, error rates, dependency traces with zero manual logging
* Telemetry setup is fail-safe: if the connection string is missing, the app still runs normally


## 🔑 Key Technical Decisions

### LangGraph State with Reducers
```python
compliance_results: Annotated[List[ComplianceIssue], operator.add]
```
`operator.add` tells LangGraph to append new items rather than overwrite — safe for concurrent node writes.

### Two-Step Azure Authentication
1. `DefaultAzureCredential` → ARM (Azure Resource Manager) token → proves Azure identity
2. Exchange ARM token → Video Indexer account token with Contributor permissions
In development: uses Azure CLI auth. In production: switches automatically to Managed Identity — zero code change.

### RAG Query Strategy
```python
query_text = f"{transcript} {' '.join(ocr_text)}"
docs = vector_store.similarity_search(query_text, k=3)
```
Transcript and OCR are combined because a violation might be spoken, shown on screen, or both.

### Deterministic Compliance via Temperature=0
For compliance auditing, `temperature=0` ensures the same video always produces the same findings — non-negotiable for legal reproducibility.

### Defensive JSON Parsing
```python
if "```" in content:
    content = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL).group(1)
```
Even with explicit JSON instructions, LLMs sometimes wrap responses in markdown fences. This regex strips them before `json.loads()`.
## 📊 Evaluation Metrics


* Precision : Of all violations flagged, what % are real? (TP / (TP + FP)) 
* Recall : Of all real violations, what % were caught? (TP / (TP + FN)) 
* F1 Score: Harmonic mean of precision and recall 
* RAG Retrieval Accuracy : Did top-3 chunks contain the relevant compliance rule? 
* Schema Conformance Rate : % of LLM responses that parse successfully 
* Latency P95 : 95th percentile end-to-end response time (target < 60s) 

## ⚙️ Tech Stack

* **Agent Orchestration:** LangGraph (DAG-style stateful workflow)
* **LLM:** Azure OpenAI (GPT-4) via AzureChatOpenAI
* **Embeddings:** AzureOpenAIEmbeddings — text-embedding-3-small (1536 dimensions)
* **Vector Store:** Azure AI Search (semantic + keyword search)
* **Video Processing:** Azure Video Indexer (Speech-to-Text + OCR)
* **Video Download:** yt-dlp
* **API Framework:** FastAPI + uvicorn + Pydantic
* **Observability:** Azure Monitor + OpenTelemetry (auto-instrumented)
* **Authentication:** Azure DefaultAzureCredential
* **PDF Loading:** LangChain PyPDFLoader
* **Knowledge Base:** FTC Disclosures 101 PDF + YouTube Ad Specifications PDF