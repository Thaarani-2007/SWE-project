import asyncio
import json
import logging
import os
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from core.ast_parser import parse_repository
from core.dataflow import analyze_repo_dataflow
from core.extractor import safe_extract_zip
from core.resolver import resolve_symbols_and_calls
from graph.builder import build_code_graph, load_graph, save_graph
from graph.traversal import expand_subgraph
from llm.embeddings import LocalEmbeddingClient
from rag.context_builder import build_rag_context
from rag.generator import validate_citations
from retrieval.graph_enricher import enrich_candidates_with_graph
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import CodeReranker
from storage.db import DatabaseManager
from storage.vector_store import CodeVectorStore

logger = logging.getLogger(__name__)

# Shared backend service singletons
db = DatabaseManager(db_path="data/metadata.db")
vstore = CodeVectorStore(persist_directory="data/chromadb")
embedder = LocalEmbeddingClient()
reranker = CodeReranker()

# Ingestion state storage: repo_id -> state dict
INGESTION_STATUS: Dict[str, Dict[str, Any]] = {}
_status_lock = threading.Lock()


def get_status(repo_id: str) -> Dict[str, Any]:
    """Retrieves current ingestion progress for a repository."""
    with _status_lock:
        if repo_id in INGESTION_STATUS:
            return INGESTION_STATUS[repo_id]

    # Fallback to SQLite
    repo = db.get_repo(repo_id)
    if repo:
        return {
            "repo_id": repo_id,
            "status": repo["status"],
            "stage": 4 if repo["status"] == "READY" else 0,
            "percentage": 100 if repo["status"] == "READY" else 0,
            "message": "Repository ready." if repo["status"] == "READY" else f"Status: {repo['status']}",
        }

    return {
        "repo_id": repo_id,
        "status": "NOT_FOUND",
        "stage": 0,
        "percentage": 0,
        "message": "Repository not found.",
    }


def update_status(repo_id: str, status: str, stage: int, percentage: int, message: str, **kwargs):
    """Updates in-memory and database status for a repository."""
    state = {
        "repo_id": repo_id,
        "status": status,
        "stage": stage,
        "percentage": percentage,
        "message": message,
        **kwargs,
    }
    with _status_lock:
        INGESTION_STATUS[repo_id] = state

    try:
        db.update_repo_status(repo_id, status)
    except Exception:
        pass


def run_ingestion_pipeline(repo_id: str, zip_path: str, repo_name: Optional[str] = None) -> None:
    """Executes the streamlined 4-stage codebase ingestion pipeline in a background thread."""
    extract_target = os.path.join("data", "repos", repo_id)
    graph_path = os.path.join("data", "graphs", f"{repo_id}.json")

    try:
        # [1] Extracting
        update_status(repo_id, "EXTRACTING", 1, 20, "Extracting Python files securely...")
        extracted_files = safe_extract_zip(zip_path, extract_target=extract_target)

        # [2] AST & Data-Flow
        update_status(repo_id, "AST_PARSING", 2, 45, "Parsing Abstract Syntax Trees and Data Flow...")
        repo_analysis = parse_repository(extract_target, repo_id=repo_id)
        resolve_symbols_and_calls(repo_analysis)
        analyze_repo_dataflow(repo_analysis)

        # [3] NetworkX Graph & SQLite
        update_status(repo_id, "GRAPH_BUILDING", 3, 70, "Constructing code graph & saving metadata...")
        code_graph = build_code_graph(repo_analysis=repo_analysis)
        save_graph(code_graph, graph_path)

        # Save files and symbols to SQLite
        file_paths = [m.file_path for m in repo_analysis.modules]
        db.save_repository_analysis(
            repo_id=repo_id,
            symbols=repo_analysis.symbols,
            files=file_paths,
            name=repo_name or repo_id,
            status="GRAPH_BUILDING",
        )

        # [4] ChromaDB Index (Local LLM summarization removed for instant, deterministic ingestion)
        update_status(repo_id, "INDEXING", 4, 90, "Indexing code chunks into ChromaDB...")
        vstore.index_symbols(repo_id, repo_analysis.symbols, embedder=embedder)

        # Complete
        update_status(
            repo_id,
            "READY",
            4,
            100,
            "Codebase successfully analyzed and indexed!",
            symbol_count=len(repo_analysis.symbols),
            file_count=len(file_paths),
        )

    except Exception as e:
        logger.error(f"Ingestion pipeline failed for repo {repo_id}: {e}", exc_info=True)
        update_status(repo_id, "FAILED", 0, 0, f"Analysis failed: {str(e)}")
    finally:
        # Remove uploaded zip file to preserve space
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass


async def stream_query_pipeline(repo_id: str, question: str) -> AsyncGenerator[str, None]:
    """Orchestrates hybrid retrieval, graph expansion, reranking, and streamed LLM generation."""
    # 1. Initial Retrieval: Hybrid Search
    candidates = hybrid_search(repo_id=repo_id, query=question, vstore=vstore, db=db, embedder=embedder, top_n=8)

    # 2. Graph Context Enrichment
    graph_path = os.path.join("data", "graphs", f"{repo_id}.json")
    graph_relations = {"callers": [], "callees": [], "data_flows": []}
    if os.path.exists(graph_path):
        try:
            code_graph = load_graph(graph_path)
            seed_sids = [c["id"] for c in candidates if "id" in c and code_graph.has_node(c["id"])]
            expansion = expand_subgraph(code_graph, seed_symbol_ids=seed_sids, hops=1)
            graph_relations["callers"] = expansion.get("callers", [])
            graph_relations["callees"] = expansion.get("callees", [])
            graph_relations["data_flows"] = expansion.get("data_flows", [])
            candidates = enrich_candidates_with_graph(candidates, code_graph, db, hops=1)
        except Exception as ge:
            logger.warning(f"Graph enrichment failed: {ge}")

    # 3. FlashRank Cross-Encoder Reranking
    reranked = reranker.rerank(query=question, documents=candidates, top_k=5)

    # Event 1: Sources and Graph Relations
    initial_event = {
        "sources": [
            {
                "id": r.get("id"),
                "file": r.get("meta", {}).get("file", "unknown"),
                "lines": r.get("meta", {}).get("lines", ""),
                "name": r.get("meta", {}).get("name", ""),
                "relationship": r.get("meta", {}).get("relationship", "Direct Match"),
                "score": float(r.get("score", 0.0)),
            }
            for r in reranked
        ],
        "graph_relations": graph_relations,
    }
    yield f"event: sources\ndata: {json.dumps(initial_event)}\n\n"

    # 4. Context Assembly
    context_str = build_rag_context(reranked, max_characters=12000)

    # 5. Stream LLM Answer Tokens
    system_prompt = (
        "You are an expert software architect answering questions about a codebase.\n"
        "Answer the user's question clearly and accurately, strictly based on the provided source code evidence.\n"
        "FORMATTING RULES:\n"
        "- Structure your answer with clean Markdown: use section headers (###), bold keywords, and bullet points.\n"
        "- Always place each numbered item or bullet point on a NEW line with proper blank line separation.\n"
        "- Always cite your source code evidence with exact brackets format: [file_path:start_line-end_line] immediately following the function or concept mentioned.\n"
        "- If showing code examples, wrap them in ```python ... ``` blocks.\n"
        "- Do not extrapolate or speculate beyond what is explicitly shown in the context."
    )
    prompt = (
        f"{system_prompt}\n\n"
        f"SOURCE CODE EVIDENCE:\n{context_str}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSWER:"
    )

    full_answer_parts: List[str] = []
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = os.environ.get("OLLAMA_MODEL")
    if not model_name:
        try:
            with httpx.Client(timeout=2.0) as check_client:
                tag_res = check_client.get(f"{base_url}/api/tags")
                if tag_res.status_code == 200:
                    installed = [m.get("name", "") for m in tag_res.json().get("models", [])]
                    preferred = [m for m in installed if "1.5b" in m] or [m for m in installed if "qwen" in m]
                    model_name = preferred[0] if preferred else (installed[0] if installed else None)
        except Exception:
            pass
    if not model_name:
        model_name = "qwen2.5-coder:1.5b"

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    full_answer_parts.append(token)
                                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                            except Exception:
                                pass
                else:
                    err_body = ""
                    try:
                        raw_body = await response.aread()
                        err_body = f" ({raw_body.decode('utf-8', errors='ignore').strip()})"
                    except Exception:
                        pass
                    err_msg = f"LLM error: HTTP {response.status_code}{err_body}"
                    full_answer_parts.append(err_msg)
                    yield f"event: token\ndata: {json.dumps({'token': err_msg})}\n\n"
    except Exception as e:
        err_msg = f"\n[LLM stream error: {e}]"
        full_answer_parts.append(err_msg)
        yield f"event: token\ndata: {json.dumps({'token': err_msg})}\n\n"

    # 6. Post-Generation Citation Validation
    full_answer = "".join(full_answer_parts)
    valid_files_map: Dict[str, tuple] = {}
    repo_files = db.get_files_by_repo(repo_id)
    for f in repo_files:
        valid_files_map[f["path"]] = (1, f.get("total_lines") or 100000)

    validated_citations, is_grounded = validate_citations(full_answer, valid_files_map)

    done_event = {
        "citations": validated_citations,
        "is_grounded": is_grounded,
        "relations": graph_relations,
    }
    yield f"event: done\ndata: {json.dumps(done_event)}\n\n"
