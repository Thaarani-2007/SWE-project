# 🧠 Codebase-Aware RAG

A local-first, privacy-focused Retrieval-Augmented Generation (RAG) system engineered specifically for software codebases. By unifying **Abstract Syntax Tree (AST) parsing**, **NetworkX call & data-flow graph traversal**, **hybrid lexical + dense vector search**, **FlashRank cross-encoder reranking**, and **strict citation validation**, Code-RAG delivers pinpoint, grounded answers with zero external cloud dependencies or API keys.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Environment Variables Setup](#-environment-variables-setup)
- [Step-by-Step Installation](#-step-by-step-installation)
- [How to Run the Project Locally](#-how-to-run-the-project-locally)
  - [1. (Optional) Start Local LLM via Ollama](#1-optional-start-local-llm-via-ollama)
  - [2. Start the FastAPI Backend](#2-start-the-fastapi-backend)
  - [3. Start the React Frontend Dashboard](#3-start-the-react-frontend-dashboard)
- [Testing & Verification](#-testing--verification)
- [Retrieval Ablation Benchmarks](#-retrieval-ablation-benchmarks)
- [API Endpoints Reference](#-api-endpoints-reference)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🚀 Key Features

- **Semantic AST Code Analysis**: Extracts Python functions, methods, classes, signatures, docstrings, caller-callee hierarchies, and variable data-flow edges using Python's native `ast` module.
- **NetworkX Multidirectional Code Graph**: Constructs and persists an in-memory directed dependency graph (`CallEdge`, `DataFlowEdge`) capable of 1-hop and multi-hop neighborhood expansion to retrieve upstream callers and downstream callees.
- **Hybrid Retrieval (RRF)**: Fuses SQLite exact token/symbol matches with ChromaDB dense vector similarity using Reciprocal Rank Fusion ($k=60$).
- **Zero-Cloud Cross-Encoder Reranking**: Re-scores candidate code snippets with an embedded FlashRank (`ms-marco-TinyBERT-L-2-v2`) cross-encoder for superior precision over standalone bi-encoders.
- **Strict Citation & Grounding Engine**: Enforces exact line-level citations (`[file.py:start_line-end_line]`) and validates that cited files and line ranges actually exist in the repository to eliminate hallucinations.
- **Real-Time Streaming Interface**: Delivers token-by-token LLM responses over Server-Sent Events (SSE) alongside live caller/callee relation chips and an in-app source file viewer.
- **100% Local & Privacy-Preserving**: No proprietary third-party LLM APIs, tracking, or cloud uploads required.

---

## 🏛️ Architecture Overview

```
                        ┌────────────────────────────────────────┐
                        │      Uploaded Codebase (.zip)          │
                        └───────────────────┬────────────────────┘
                                            │
                        ┌───────────────────▼────────────────────┐
                        │  Stage 1: Safe Extraction & AST Parser │
                        │  (Symbols, Signatures, Calls, Flow)    │
                        └─────────────┬──────────────────────────┘
                                      │
              ┌───────────────────────┴────────────────────────┐
              │                                                │
   ┌──────────▼────────────┐                        ┌──────────▼──────────┐
   │ Stage 2: Code Graph   │                        │ Stage 3: Indexing   │
   │ (NetworkX Relations & │                        │ (ChromaDB Vectors & │
   │ SQLite Metadata)      │                        │ SQLite Full-Text)   │
   └──────────┬────────────┘                        └──────────┬──────────┘
              │                                                │
              └───────────────────────┬────────────────────────┘
                                      │
                        ┌─────────────▼──────────────────────────┐
                        │ User Query -> Hybrid Search (RRF)      │
                        └─────────────┬──────────────────────────┘
                                      │
                        ┌─────────────▼──────────────────────────┐
                        │ Stage 4: Graph Subgraph Expansion      │
                        │ (Enrich with Callers, Callees, Flow)   │
                        └─────────────┬──────────────────────────┘
                                      │
                        ┌─────────────▼──────────────────────────┐
                        │ FlashRank Cross-Encoder Reranking      │
                        └─────────────┬──────────────────────────┘
                                      │
                        ┌─────────────▼──────────────────────────┐
                        │ Stage 5: Context Assembly & Ollama LLM │
                        │ (SSE Token Stream + Citation Validator)│
                        └────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
code-rag/
├── api/
│   ├── main.py                  # FastAPI application entrypoint with CORS & lifespan
│   ├── routes.py                # REST & SSE streaming endpoints (/upload, /query, /repos, etc.)
│   └── services.py              # Ingestion pipeline background worker & query orchestrator
├── core/
│   ├── ast_parser.py            # AST visitor for functions, classes, and call detection
│   ├── dataflow.py              # Variable assignment & argument passing flow analyzer
│   ├── extractor.py             # Secure zip extractor with path traversal defenses
│   ├── models.py                # Pydantic data models (CodeSymbol, CallEdge, DataFlowEdge)
│   └── resolver.py              # Global symbol resolution & caller-callee linking
├── data/
│   ├── chromadb/                # Local persistent vector store
│   ├── graphs/                  # Serialized NetworkX JSON graphs per repository
│   ├── metadata.db              # SQLite repository & symbol database
│   └── repos/                   # Extracted codebase working trees
├── frontend/                    # Modern React 19 + TypeScript + Vite web dashboard
│   ├── src/
│   │   ├── App.tsx              # Main UI (upload, progress, chat, citation viewer, graph)
│   │   ├── App.css              # Custom responsive stylesheet & glassmorphic aesthetics
│   │   └── main.tsx             # React root mount
│   ├── package.json             # Node dependencies and scripts
│   └── vite.config.ts           # Vite development server configuration
├── graph/
│   ├── builder.py               # NetworkX graph construction & serialization
│   └── traversal.py             # Subgraph expansion (callers, callees, data flows)
├── llm/
│   ├── embeddings.py            # Local embedding client (sentence-transformers / Ollama)
│   └── summarizer.py            # Function summarization worker pool with fallback
├── rag/
│   ├── context_builder.py       # Formats code evidence with symbol & line bounds
│   └── generator.py             # Prompt builder and regex citation validator
├── retrieval/
│   ├── graph_enricher.py        # Expands candidate passages using graph neighbors
│   ├── hybrid_search.py         # Reciprocal Rank Fusion of dense and lexical scores
│   └── reranker.py              # FlashRank TinyBERT cross-encoder reranker
├── storage/
│   ├── db.py                    # SQLite database schema, transactions, and search
│   └── vector_store.py          # ChromaDB collection management and query wrapper
├── tests/                       # Comprehensive Pytest regression suite
├── benchmark_retrieval.py       # Hit-Rate@K and MRR ablation benchmark script
├── test_stage1.py ... stage5.py # Quick individual stage verification scripts
├── requirements.txt             # Python backend dependencies
├── .env.example                 # Environment configuration template
└── README.md                    # Project documentation
```

---

## 📋 Prerequisites

Ensure your development machine has the following tools installed:

1. **Python**: Version `3.10` or higher (`python --version`)
2. **Node.js & npm**: Node `18.0` or higher (`node -v`, `npm -v`)
3. **Ollama (Recommended)**: For local LLM answers. Download from [ollama.com](https://ollama.com).  
   *(Note: The system gracefully falls back to deterministic embeddings and structural fallbacks if Ollama is not running).*

---

## ⚙️ Environment Variables Setup

Code-RAG is **100% self-hosted and offline-capable**. **No sensitive API keys (e.g., OpenAI, Anthropic, Cohere, Pinecone) are required.**

All environment variables have sensible out-of-the-box defaults. You can customize them by creating a `.env` file in the project root:

### 1. Copy the Example Template

```bash
# On Linux / macOS
cp .env.example .env

# On Windows (PowerShell / CMD)
copy .env.example .env
```

### 2. Available Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of your local Ollama daemon service. |
| `OLLAMA_MODEL` | `qwen2.5-coder:1.5b` | Model name for code reasoning and token streaming. (Alternatives: `qwen2.5-coder:7b`, `deepseek-coder:6.7b`). |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding engine: `sentence-transformers` (runs locally via HuggingFace) or `ollama`. If unset, auto-detects. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model identifier (`all-MiniLM-L6-v2` for sentence-transformers, or `nomic-embed-text` / `bge-m3` for Ollama). |
| `BACKEND_HOST` | `0.0.0.0` | Host interface for the FastAPI backend server. |
| `BACKEND_PORT` | `8000` | Port for the FastAPI backend server. |
| `FRONTEND_PORT` | `5173` | Development port for the Vite frontend server. |

---

## 🛠️ Step-by-Step Installation

### Step 1: Clone or Navigate to the Workspace

```bash
cd /path/to/code-rag
```

### Step 2: Set Up Python Virtual Environment

Create and activate a virtual environment:

```bash
# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt)
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### Step 3: Install Backend Dependencies

Upgrade `pip` and install all necessary Python libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies

In a separate terminal or subshell, install the React / TypeScript packages:

```bash
cd frontend
npm install
cd ..
```

---

## 💻 How to Run the Project Locally

Running Code-RAG locally involves starting Ollama (optional but recommended), launching the FastAPI backend server, and launching the React development server.

### 1. (Optional) Start Local LLM via Ollama

To enable local code comprehension and real-time answer generation:

1. Start the Ollama server:
   ```bash
   ollama serve
   ```
2. Pull the recommended lightweight code model:
   ```bash
   ollama pull qwen2.5-coder:1.5b
   ```
   *(Optional: If you want Ollama to handle embeddings too, run `ollama pull nomic-embed-text`).*

### 2. Start the FastAPI Backend

With your virtual environment activated, run the Uvicorn server from the repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Backend API**: `http://localhost:8000`
- **Interactive OpenAPI Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative ReDoc Docs**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### 3. Start the React Frontend Dashboard

In a new terminal window:

```bash
cd frontend
npm run dev
```

- Open **`http://localhost:5173`** in your browser.

---

## 🎯 Usage Walkthrough

1. **Upload a Codebase**:
   - Drag & drop or select any zipped Python project (a sample `django_ecommerce_sample.zip` is included in the project root for immediate testing).
   - Watch the real-time stage progress:
     1. `EXTRACTING` (Zip extraction & safety checks)
     2. `AST_PARSING` (Syntax tree extraction & data-flow graph)
     3. `GRAPH_BUILDING` (NetworkX dependency linking & SQLite persistence)
     4. `INDEXING` (ChromaDB vector embedding)
2. **Query the Codebase**:
   - Ask questions such as:
     - *"How does order checkout reserve inventory and handle payment failure?"*
     - *"Where are user credentials verified against password hashes?"*
     - *"Explain the IP rate limiting middleware implementation."*
3. **Inspect Citations & Code**:
   - Every generated answer includes clickable citations such as `[checkout/services.py:45-80]`.
   - Click any citation to open the source code drawer with exact line highlighting.
   - Explore the caller/callee relation tags directly under the response.

---

## 🧪 Testing & Verification

Code-RAG comes with both quick checkpoint scripts for each ingestion stage and a comprehensive Pytest suite.

### Run Individual Stage Checkpoints

```bash
# Stage 1: AST parsing, symbol extraction, call & dataflow linking
python test_stage1.py

# Stage 2: NetworkX graph construction, serialization, & subgraph expansion
python test_stage2.py

# Stage 3: Sentence-Transformers / Ollama embeddings & ChromaDB indexing
python test_stage3.py

# Stage 4: FlashRank cross-encoder reranking & citation validation
python test_stage4.py

# Stage 5: FastAPI file upload, background tasks, & status endpoints
python test_stage5.py
```

### Run Comprehensive Pytest Suite

```bash
pytest tests/ -v
```

---

## 📊 Retrieval Ablation Benchmarks

To evaluate retrieval efficacy across different stages of the pipeline on a standardized Django codebase benchmark:

```bash
python benchmark_retrieval.py
```

The benchmark evaluates:
1. **Dense Vector Search** (Pure ChromaDB embeddings)
2. **Lexical Matching** (Pure SQLite token frequency)
3. **Hybrid Search** (Reciprocal Rank Fusion)
4. **Full Code-RAG Pipeline** (Hybrid + NetworkX Graph Enrichment + FlashRank Cross-Encoder)

Metrics calculated: **Hit Rate / Recall @ 1, 3, 5**, **Mean Reciprocal Rank (MRR)**, and **Latency (ms)**.

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status check. |
| `GET` | `/api/repos` | Returns all indexed repositories with status and metadata. |
| `POST` | `/api/upload` | Uploads a `.zip` archive and triggers asynchronous background ingestion. |
| `GET` | `/api/status/{repo_id}` | Polls ingestion stage (1-4), percentage (0-100%), and status message. |
| `POST` | `/api/query` | Streams an answer via Server-Sent Events (`event: sources`, `event: token`, `event: done`). |
| `GET` | `/api/files/{repo_id}` | Retrieves all parsed file paths and line counts for a repository. |
| `GET` | `/api/file_content` | Returns raw source code for a specific file (`?repo_id=...&path=...`). |
| `GET` | `/api/symbol/{symbol_id}` | Retrieves symbol signature, docstring, callers, and body. |
| `DELETE` | `/api/repos/{repo_id}` | Deletes a repository, its ChromaDB vectors, graph JSON, and SQLite records. |

---

## ❓ Troubleshooting & FAQ

### 1. Ollama Connection Issues
- **Symptom**: Responses show `LLM generation failed: Connection refused` or take longer to load.
- **Solution**: Ensure the Ollama daemon is running (`ollama serve`) and the model is downloaded (`ollama pull qwen2.5-coder:1.5b`). If Ollama is unavailable, the pipeline still retrieves source passages, computes graph relations, and displays citations.

### 2. Port Already in Use
- If port `8000` is busy, launch Uvicorn on a different port:
  ```bash
  uvicorn api.main:app --port 8080 --reload
  ```
  *(Update `API_BASE` in `frontend/src/App.tsx` if changed).*
- If port `5173` is busy, Vite will automatically select the next available port (e.g. `5174`).

### 3. Sentence-Transformers First-Run Model Download
- On the very first run, `sentence-transformers` automatically downloads `all-MiniLM-L6-v2` (~90MB) from Hugging Face and caches it locally. Subsequent runs operate entirely offline.

### 4. Windows Long Paths & Execution Policy
- If you encounter a script execution error in PowerShell (`Activate.ps1 cannot be loaded`), run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
