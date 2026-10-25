import os
import sys
import tempfile
import networkx as nx
from storage.db import DatabaseManager
from graph.builder import build_code_graph, save_graph, load_graph
from graph.traversal import expand_subgraph
from core.models import CodeSymbol, CallEdge, DataFlowEdge, RepoAnalysisResult

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_stage2_checkpoint():
    print("--- [STAGE 2 CHECKPOINT] Testing NetworkX Graph & SQLite Storage ---")
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    graph_path = os.path.join(temp_dir, "test_graph.json")

    # 1. Initialize DB and save mock symbols
    db = DatabaseManager(db_path)
    db.init_schema()

    s1 = CodeSymbol(
        symbol_id="api.py::router::login",
        file_path="api.py",
        name="login",
        symbol_type="function",
        start_line=1,
        end_line=10,
        signature="def login(user, pwd):",
        docstring="",
        code_body="def login(user, pwd): verify(user, pwd)"
    )
    s2 = CodeSymbol(
        symbol_id="auth.py::::verify",
        file_path="auth.py",
        name="verify",
        symbol_type="function",
        start_line=1,
        end_line=5,
        signature="def verify(u, p):",
        docstring="",
        code_body="def verify(u, p): return True"
    )

    db.save_repository_analysis(repo_id="test_repo", symbols=[s1, s2], files=["api.py", "auth.py"])
    loaded_s1 = db.get_symbol("api.py::router::login")
    assert loaded_s1 is not None and loaded_s1.name == "login", "SQLite symbol retrieval failed"

    # 2. Build NetworkX Graph
    calls = [CallEdge(caller=s1.symbol_id, callee=s2.symbol_id, line_no=5)]
    dataflow = [DataFlowEdge(source=s1.symbol_id, target=s2.symbol_id, flow_type="ARG_PASS")]
    graph = build_code_graph(symbols=[s1, s2], call_edges=calls, dataflow_edges=dataflow)

    assert graph.has_node(s1.symbol_id) and graph.has_node(s2.symbol_id)
    assert graph.has_edge(s1.symbol_id, s2.symbol_id)

    # 3. Test Graph Persistence
    save_graph(graph, graph_path)
    reloaded_graph = load_graph(graph_path)
    assert reloaded_graph.number_of_nodes() == 2

    # 4. Test Subgraph Expansion
    expansion = expand_subgraph(reloaded_graph, seed_symbol_ids=[s2.symbol_id], hops=1)
    assert s1.symbol_id in expansion["callers"], f"Failed to identify {s1.symbol_id} as caller of {s2.symbol_id}"
    print("✅ Graph nodes, edges, serialization, and caller/callee expansion verified.")
    print("✅ STAGE 2 VERIFICATION PASSED.\n")

if __name__ == "__main__":
    run_stage2_checkpoint()
