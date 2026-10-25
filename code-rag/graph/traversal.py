from typing import Any, Dict, List, Optional, Set

import networkx as nx


def expand_subgraph(
    graph: nx.MultiDiGraph,
    seed_symbol_ids: List[str],
    hops: int = 1,
    k: Optional[int] = None,
    db: Optional[Any] = None,
) -> Dict[str, Any]:
    """Expands a code subgraph around seed symbols using NetworkX graph traversal.

    Traverses:
    - Incoming edges with type CALLS (callers who invoke seed functions).
    - Outgoing edges with type CALLS (callees invoked by seed functions).
    - DATA_FLOW_TO connections (parameter, variable assignment, and return flows).

    Args:
        graph: The repository NetworkX code graph.
        seed_symbol_ids: The initial matched symbol IDs.
        hops: Number of traversal hops (default 1).
        k: Alias for hops.
        db: Optional DatabaseManager instance to enrich symbols with source code from SQLite.

    Returns:
        Dict containing:
        - seed_symbols: Detailed metadata for seed symbols.
        - callers: List of caller symbol IDs.
        - callees: List of callee symbol IDs.
        - data_flows: List of connected data-flow edges.
    """
    hop_count = k if k is not None else hops

    # 1. Collect seed symbol metadata
    seed_symbols: List[Dict[str, Any]] = []
    for sid in seed_symbol_ids:
        if graph.has_node(sid):
            attrs = dict(graph.nodes[sid])
            attrs["symbol_id"] = sid
            seed_symbols.append(attrs)

    # 2. Traverse incoming CALLS edges (Callers)
    callers: List[str] = []
    visited_callers: Set[str] = set()
    frontier = set(seed_symbol_ids)

    for _ in range(hop_count):
        next_frontier: Set[str] = set()
        for node in frontier:
            if not graph.has_node(node):
                continue
            for u, _, data in graph.in_edges(node, data=True):
                if data.get("type") == "CALLS" and u not in seed_symbol_ids:
                    if u not in visited_callers:
                        visited_callers.add(u)
                        callers.append(u)
                        next_frontier.add(u)
        frontier = next_frontier

    # 3. Traverse outgoing CALLS edges (Callees)
    callees: List[str] = []
    visited_callees: Set[str] = set()
    frontier = set(seed_symbol_ids)

    for _ in range(hop_count):
        next_frontier: Set[str] = set()
        for node in frontier:
            if not graph.has_node(node):
                continue
            for _, v, data in graph.out_edges(node, data=True):
                if data.get("type") == "CALLS" and v not in seed_symbol_ids:
                    if v not in visited_callees:
                        visited_callees.add(v)
                        callees.append(v)
                        next_frontier.add(v)
        frontier = next_frontier

    # 4. Traverse DATA_FLOW_TO edges for all relevant nodes
    relevant_nodes = set(seed_symbol_ids) | visited_callers | visited_callees
    data_flows: List[Dict[str, Any]] = []
    seen_df: Set[tuple] = set()

    for node in relevant_nodes:
        if not graph.has_node(node):
            continue
        # Outgoing data flows
        for u, v, data in graph.out_edges(node, data=True):
            if data.get("type") == "DATA_FLOW_TO":
                key = (u, v, data.get("flow_type"), data.get("line_no"))
                if key not in seen_df:
                    seen_df.add(key)
                    data_flows.append({
                        "source": u,
                        "target": v,
                        "flow_type": data.get("flow_type", "DATA_FLOW"),
                        "line_no": data.get("line_no"),
                        "symbol_id": data.get("symbol_id"),
                    })
        # Incoming data flows
        for u, v, data in graph.in_edges(node, data=True):
            if data.get("type") == "DATA_FLOW_TO":
                key = (u, v, data.get("flow_type"), data.get("line_no"))
                if key not in seen_df:
                    seen_df.add(key)
                    data_flows.append({
                        "source": u,
                        "target": v,
                        "flow_type": data.get("flow_type", "DATA_FLOW"),
                        "line_no": data.get("line_no"),
                        "symbol_id": data.get("symbol_id"),
                    })

    result: Dict[str, Any] = {
        "seed_symbols": seed_symbols,
        "callers": callers,
        "callees": callees,
        "data_flows": data_flows,
    }

    # Optional SQLite enrichment
    if db is not None:
        enriched_callers = []
        for c_id in callers:
            sym = db.get_symbol(c_id)
            if sym:
                enriched_callers.append(sym)
        result["caller_symbols"] = enriched_callers

        enriched_callees = []
        for c_id in callees:
            sym = db.get_symbol(c_id)
            if sym:
                enriched_callees.append(sym)
        result["callee_symbols"] = enriched_callees

    return result
