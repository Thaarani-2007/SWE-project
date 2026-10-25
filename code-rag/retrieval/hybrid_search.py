import re
from typing import Any, Dict, List, Optional, Tuple

from core.models import CodeSymbol
from storage.db import DatabaseManager
from storage.vector_store import CodeVectorStore
from llm.embeddings import LocalEmbeddingClient


def hybrid_search(
    repo_id: str,
    query: str,
    vstore: CodeVectorStore,
    db: DatabaseManager,
    embedder: Optional[LocalEmbeddingClient] = None,
    top_n: int = 10,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Performs hybrid retrieval combining semantic vector search (ChromaDB)

    and exact keyword/symbol search (SQLite) using Reciprocal Rank Fusion (RRF).
    """
    if embedder is None:
        embedder = LocalEmbeddingClient()

    # Step A: Semantic Vector Search
    vector_results = vstore.search(
        repo_id=repo_id,
        query=query,
        n_results=top_n * 2,
        embedder=embedder,
    )
    vector_rank: Dict[str, int] = {}
    for rank, item in enumerate(vector_results, start=1):
        sid = item.get("symbol_id")
        if sid and sid not in vector_rank:
            vector_rank[sid] = rank

    # Step B: Exact Keyword / Symbol Search in SQLite
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", query) if len(t) > 1]
    keyword_matches: List[Tuple[CodeSymbol, float]] = []

    if tokens:
        repo_symbols = db.get_symbols_by_repo(repo_id)
        for sym in repo_symbols:
            score = 0.0
            s_name = sym.name.lower()
            s_file = sym.file_path.lower()

            for token in tokens:
                if token == s_name:
                    score += 10.0
                elif token in s_name:
                    score += 4.0
                if token in s_file:
                    score += 2.0

            if score > 0:
                keyword_matches.append((sym, score))

        keyword_matches.sort(key=lambda x: x[1], reverse=True)

    keyword_rank: Dict[str, int] = {}
    for rank, (sym, _) in enumerate(keyword_matches[: top_n * 2], start=1):
        if sym.symbol_id not in keyword_rank:
            keyword_rank[sym.symbol_id] = rank

    # Step C: Merge candidates using Reciprocal Rank Fusion (RRF)
    all_symbol_ids = set(vector_rank.keys()) | set(keyword_rank.keys())
    rrf_scores: Dict[str, float] = {}

    for sid in all_symbol_ids:
        score = 0.0
        if sid in vector_rank:
            score += 1.0 / (rrf_k + vector_rank[sid])
        if sid in keyword_rank:
            score += 1.0 / (rrf_k + keyword_rank[sid])
        rrf_scores[sid] = score

    sorted_sids = sorted(all_symbol_ids, key=lambda s: rrf_scores[s], reverse=True)

    # Format result items
    candidates: List[Dict[str, Any]] = []
    for sid in sorted_sids[:top_n]:
        sym = db.get_symbol(sid)
        if sym:
            text = sym.code_body or sym.signature or sym.name
            candidates.append({
                "id": sym.symbol_id,
                "text": text,
                "meta": {
                    "file": sym.file_path,
                    "file_path": sym.file_path,
                    "lines": f"{sym.start_line}-{sym.end_line}",
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "name": sym.name,
                    "symbol_id": sym.symbol_id,
                    "relationship": "Direct Match",
                    "rrf_score": rrf_scores[sid],
                },
                "score": rrf_scores[sid],
            })

    return candidates
