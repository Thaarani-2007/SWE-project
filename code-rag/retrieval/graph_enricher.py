from typing import Any, Dict, List

import networkx as nx

from graph.traversal import expand_subgraph
from storage.db import DatabaseManager


def enrich_candidates_with_graph(
    candidates: List[Dict[str, Any]],
    graph: nx.MultiDiGraph,
    db: DatabaseManager,
    hops: int = 1,
) -> List[Dict[str, Any]]:
    """Enriches candidate retrieval passages with connected callers and callees

    from the NetworkX code graph and fetches their code bodies from SQLite.
    """
    if not candidates or graph is None:
        return candidates

    seed_symbol_ids = [
        c.get("id") or c.get("meta", {}).get("symbol_id")
        for c in candidates
        if c.get("id") or c.get("meta", {}).get("symbol_id")
    ]
    seed_symbol_ids = [s for s in seed_symbol_ids if s and graph.has_node(s)]

    if not seed_symbol_ids:
        return candidates

    expansion = expand_subgraph(graph, seed_symbol_ids=seed_symbol_ids, hops=hops)
    existing_ids = {
        c.get("id") or c.get("meta", {}).get("symbol_id") for c in candidates
    }
    enriched = list(candidates)

    # 1. Add Callers
    for caller_id in expansion.get("callers", []):
        if caller_id not in existing_ids:
            sym = db.get_symbol(caller_id)
            if sym:
                existing_ids.add(caller_id)
                enriched.append({
                    "id": sym.symbol_id,
                    "text": sym.code_body or sym.signature or sym.name,
                    "meta": {
                        "file": sym.file_path,
                        "file_path": sym.file_path,
                        "lines": f"{sym.start_line}-{sym.end_line}",
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                        "name": sym.name,
                        "symbol_id": sym.symbol_id,
                        "relationship": "Caller",
                    },
                })

    # 2. Add Callees
    for callee_id in expansion.get("callees", []):
        if callee_id not in existing_ids:
            sym = db.get_symbol(callee_id)
            if sym:
                existing_ids.add(callee_id)
                enriched.append({
                    "id": sym.symbol_id,
                    "text": sym.code_body or sym.signature or sym.name,
                    "meta": {
                        "file": sym.file_path,
                        "file_path": sym.file_path,
                        "lines": f"{sym.start_line}-{sym.end_line}",
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                        "name": sym.name,
                        "symbol_id": sym.symbol_id,
                        "relationship": "Callee",
                    },
                })

    return enriched
