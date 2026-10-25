import os
import shutil
import tempfile
import pytest

from core.models import CodeSymbol, CallEdge, DataFlowEdge, RepoAnalysisResult
from storage.db import DatabaseManager
from graph.builder import build_code_graph, save_graph, load_graph
from graph.traversal import expand_subgraph


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_sqlite_schema_and_crud(temp_workspace):
    db_path = os.path.join(temp_workspace, "test.db")
    db = DatabaseManager(db_path)
    db.init_schema()

    # Test repo upsert
    s1 = CodeSymbol(
        symbol_id="main.py::::entry",
        file_path="main.py",
        name="entry",
        symbol_type="function",
        start_line=1,
        end_line=5,
        signature="def entry():",
        docstring="Entrypoint function",
        code_body="def entry():\n    pass",
    )
    s2 = CodeSymbol(
        symbol_id="utils.py::Helper::run",
        file_path="utils.py",
        name="run",
        symbol_type="method",
        start_line=10,
        end_line=20,
        signature="def run(self):",
        docstring="Helper run method",
        code_body="def run(self):\n    return True",
    )

    db.save_repository_analysis(
        repo_id="test_project",
        symbols=[s1, s2],
        files=["main.py", "utils.py"],
        name="Test Project",
    )

    # Verify repo retrieval
    repo = db.get_repo("test_project")
    assert repo is not None
    assert repo["name"] == "Test Project"
    assert repo["status"] == "READY"

    # Verify symbols retrieval
    retrieved_s1 = db.get_symbol("main.py::::entry")
    assert retrieved_s1 is not None
    assert retrieved_s1.name == "entry"
    assert retrieved_s1.docstring == "Entrypoint function"

    repo_symbols = db.get_symbols_by_repo("test_project")
    assert len(repo_symbols) == 2

    # Verify files
    repo_files = db.get_files_by_repo("test_project")
    assert len(repo_files) == 2

    # Test function summary save & fetch
    db.save_function_summary(
        symbol_id="main.py::::entry",
        summary_text="Initializes and runs the main loop.",
        model_name="qwen2.5-coder:7b",
    )
    summary = db.get_function_summary("main.py::::entry")
    assert summary is not None
    assert summary["summary_text"] == "Initializes and runs the main loop."
    assert summary["model_name"] == "qwen2.5-coder:7b"

    # Test cascade delete
    db.delete_repository("test_project")
    assert db.get_repo("test_project") is None
    assert db.get_symbol("main.py::::entry") is None
    assert db.get_function_summary("main.py::::entry") is None


def test_graph_construction_and_multiedges(temp_workspace):
    s_mod = CodeSymbol(
        symbol_id="service.py::::",
        file_path="service.py",
        name="service",
        symbol_type="module",
        start_line=1,
        end_line=50,
    )
    s_class = CodeSymbol(
        symbol_id="service.py::Service::",
        file_path="service.py",
        name="Service",
        symbol_type="class",
        start_line=5,
        end_line=45,
    )
    s_method = CodeSymbol(
        symbol_id="service.py::Service::execute",
        file_path="service.py",
        name="execute",
        symbol_type="method",
        start_line=10,
        end_line=25,
        parent_symbol="service.py::Service::",
    )
    s_helper = CodeSymbol(
        symbol_id="service.py::::helper",
        file_path="service.py",
        name="helper",
        symbol_type="function",
        start_line=30,
        end_line=40,
    )

    symbols = [s_mod, s_class, s_method, s_helper]
    calls = [
        CallEdge(caller=s_method.symbol_id, callee=s_helper.symbol_id, line_no=15)
    ]
    dataflow = [
        DataFlowEdge(source=s_method.symbol_id, target=s_helper.symbol_id, flow_type="ARG_PASS", line_no=15)
    ]

    graph = build_code_graph(symbols=symbols, call_edges=calls, dataflow_edges=dataflow)

    assert graph.number_of_nodes() == 4
    # Check CONTAINS edges
    assert graph.has_edge(s_mod.symbol_id, s_class.symbol_id)
    assert graph.has_edge(s_class.symbol_id, s_method.symbol_id)
    assert graph.has_edge(s_mod.symbol_id, s_helper.symbol_id)

    # Check both CALLS and DATA_FLOW_TO edges exist between method and helper
    edge_dict = graph.get_edge_data(s_method.symbol_id, s_helper.symbol_id)
    types = [data["type"] for data in edge_dict.values()]
    assert "CALLS" in types
    assert "DATA_FLOW_TO" in types


def test_graph_persistence_roundtrip(temp_workspace):
    graph_path = os.path.join(temp_workspace, "graph.json")
    s1 = CodeSymbol(
        symbol_id="a.py::::foo", file_path="a.py", name="foo", symbol_type="function", start_line=1, end_line=5
    )
    s2 = CodeSymbol(
        symbol_id="b.py::::bar", file_path="b.py", name="bar", symbol_type="function", start_line=1, end_line=5
    )
    calls = [CallEdge(caller=s1.symbol_id, callee=s2.symbol_id, line_no=3)]

    graph = build_code_graph(symbols=[s1, s2], call_edges=calls)
    save_graph(graph, graph_path)

    reloaded = load_graph(graph_path)
    assert reloaded.number_of_nodes() == 2
    assert reloaded.has_edge(s1.symbol_id, s2.symbol_id)
    assert reloaded.nodes[s1.symbol_id]["name"] == "foo"


def test_subgraph_multi_hop_expansion():
    # Chain: A calls B calls C
    s_a = CodeSymbol(symbol_id="a.py::::a", file_path="a.py", name="a", symbol_type="function", start_line=1, end_line=5)
    s_b = CodeSymbol(symbol_id="b.py::::b", file_path="b.py", name="b", symbol_type="function", start_line=1, end_line=5)
    s_c = CodeSymbol(symbol_id="c.py::::c", file_path="c.py", name="c", symbol_type="function", start_line=1, end_line=5)

    calls = [
        CallEdge(caller=s_a.symbol_id, callee=s_b.symbol_id, line_no=2),
        CallEdge(caller=s_b.symbol_id, callee=s_c.symbol_id, line_no=3),
    ]
    dataflow = [
        DataFlowEdge(source=s_b.symbol_id, target=s_c.symbol_id, flow_type="ARG_PASS", line_no=3)
    ]

    graph = build_code_graph(symbols=[s_a, s_b, s_c], call_edges=calls, dataflow_edges=dataflow)

    # 1 hop from C: only B is caller
    exp_1hop = expand_subgraph(graph, seed_symbol_ids=[s_c.symbol_id], hops=1)
    assert s_b.symbol_id in exp_1hop["callers"]
    assert s_a.symbol_id not in exp_1hop["callers"]

    # 2 hops from C: both B and A are callers
    exp_2hop = expand_subgraph(graph, seed_symbol_ids=[s_c.symbol_id], hops=2)
    assert s_b.symbol_id in exp_2hop["callers"]
    assert s_a.symbol_id in exp_2hop["callers"]

    # Callees from A: 2 hops includes B and C
    exp_callees = expand_subgraph(graph, seed_symbol_ids=[s_a.symbol_id], hops=2)
    assert s_b.symbol_id in exp_callees["callees"]
    assert s_c.symbol_id in exp_callees["callees"]


def test_subgraph_enrichment_with_db(temp_workspace):
    db_path = os.path.join(temp_workspace, "enrich.db")
    db = DatabaseManager(db_path)
    db.init_schema()

    s1 = CodeSymbol(
        symbol_id="main.py::::caller",
        file_path="main.py",
        name="caller",
        symbol_type="function",
        start_line=1,
        end_line=10,
        signature="def caller():",
        docstring="Calls callee",
        code_body="def caller():\n    callee()",
    )
    s2 = CodeSymbol(
        symbol_id="main.py::::callee",
        file_path="main.py",
        name="callee",
        symbol_type="function",
        start_line=11,
        end_line=20,
        signature="def callee():",
        docstring="Called function",
        code_body="def callee():\n    return 1",
    )

    db.save_repository_analysis(repo_id="test_repo", symbols=[s1, s2], files=["main.py"])
    graph = build_code_graph(symbols=[s1, s2], call_edges=[CallEdge(caller=s1.symbol_id, callee=s2.symbol_id, line_no=5)])

    expansion = expand_subgraph(graph, seed_symbol_ids=[s2.symbol_id], hops=1, db=db)
    assert s1.symbol_id in expansion["callers"]
    assert "caller_symbols" in expansion
    assert len(expansion["caller_symbols"]) == 1
    assert expansion["caller_symbols"][0].name == "caller"
    assert "callee()" in expansion["caller_symbols"][0].code_body
