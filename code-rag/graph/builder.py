import json
import os
from typing import List, Optional

import networkx as nx

from core.models import CallEdge, CodeSymbol, DataFlowEdge, RepoAnalysisResult


def build_code_graph(
    symbols: Optional[List[CodeSymbol]] = None,
    call_edges: Optional[List[CallEdge]] = None,
    dataflow_edges: Optional[List[DataFlowEdge]] = None,
    repo_analysis: Optional[RepoAnalysisResult] = None,
) -> nx.MultiDiGraph:
    """Constructs a directed multigraph representing code symbols, call hierarchies,

    containment, imports, and data-flow relationships.

    Edge types:
    - CONTAINS: module -> class, class -> method, module -> function
    - CALLS: caller_func -> callee_func
    - DATA_FLOW_TO: arg/var -> downstream assignment or return
    """
    if repo_analysis is not None:
        if symbols is None:
            symbols = repo_analysis.symbols
        if call_edges is None:
            call_edges = repo_analysis.call_edges
        if dataflow_edges is None:
            dataflow_edges = repo_analysis.dataflow_edges

    symbols = symbols or []
    call_edges = call_edges or []
    dataflow_edges = dataflow_edges or []

    graph = nx.MultiDiGraph()

    # 1. Add symbol nodes
    symbol_id_set = set()
    for sym in symbols:
        symbol_id_set.add(sym.symbol_id)
        graph.add_node(
            sym.symbol_id,
            name=sym.name,
            type=sym.symbol_type,
            file_path=sym.file_path,
            lines=[sym.start_line, sym.end_line],
            signature=sym.signature,
            docstring=sym.docstring,
        )

    # 2. Add CONTAINS edges
    for sym in symbols:
        # Class -> Method
        if sym.symbol_type == "method" and sym.parent_symbol:
            if graph.has_node(sym.parent_symbol):
                graph.add_edge(sym.parent_symbol, sym.symbol_id, type="CONTAINS")
        # Module -> Class / Top-level Function
        elif sym.symbol_type in ("class", "function"):
            module_symbol_id = f"{sym.file_path}::::"
            if module_symbol_id in symbol_id_set and sym.symbol_id != module_symbol_id:
                graph.add_edge(module_symbol_id, sym.symbol_id, type="CONTAINS")

    # 3. Add CALLS edges
    for call in call_edges:
        caller = call.caller
        callee = call.callee
        if not graph.has_node(caller):
            graph.add_node(caller, name=caller, type="unknown", file_path="", lines=[0, 0])
        if not graph.has_node(callee):
            graph.add_node(
                callee,
                name=callee,
                type="external" if call.is_external else "unknown",
                file_path="",
                lines=[0, 0],
            )
        graph.add_edge(
            caller,
            callee,
            type="CALLS",
            line_no=call.line_no,
            is_external=call.is_external,
        )

    # 4. Add DATA_FLOW_TO edges
    for df in dataflow_edges:
        source = df.source or df.source_expr
        target = df.target or df.sink_expr
        flow_type = df.flow_type or df.relation_type
        if source and target:
            if not graph.has_node(source):
                graph.add_node(source, name=source, type="data_node", file_path="", lines=[0, 0])
            if not graph.has_node(target):
                graph.add_node(target, name=target, type="data_node", file_path="", lines=[0, 0])
            graph.add_edge(
                source,
                target,
                type="DATA_FLOW_TO",
                flow_type=flow_type,
                line_no=df.line_no,
                symbol_id=df.symbol_id,
            )

    return graph


def save_graph(graph: nx.Graph, path: str) -> None:
    """Serializes the NetworkX graph to a JSON file using node_link_data."""
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    data = nx.node_link_data(graph)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str) -> nx.MultiDiGraph:
    """Loads a serialized NetworkX graph from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)
