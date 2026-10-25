import os
import shutil
import tempfile
import pytest

from core.models import CodeSymbol, CallEdge
from storage.db import DatabaseManager
from storage.vector_store import CodeVectorStore
from graph.builder import build_code_graph
from llm.embeddings import LocalEmbeddingClient
from retrieval.reranker import CodeReranker
from retrieval.hybrid_search import hybrid_search
from retrieval.graph_enricher import enrich_candidates_with_graph
from rag.context_builder import build_rag_context
from rag.generator import validate_citations


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_reranker_edge_cases():
    reranker = CodeReranker()
    # Empty documents
    assert reranker.rerank("test query", []) == []

    # Single document
    single = [{"id": "1", "text": "def calculate(): pass", "meta": {"file": "calc.py"}}]
    res = reranker.rerank("calculate", single, top_k=5)
    assert len(res) == 1
    assert res[0]["id"] == "1"


def test_context_builder_character_budget():
    docs = [
        {
            "id": f"doc_{i}",
            "text": f"def function_{i}():\n" + "    print('line')\n" * 50,
            "meta": {"file": f"file_{i}.py", "lines": "1-51", "name": f"function_{i}"},
        }
        for i in range(10)
    ]
    # Restrict budget to 500 characters
    context = build_rag_context(docs, max_characters=500)
    assert len(context) < 1000
    assert "### File: file_0.py" in context
    # file_9 should be cut off by budget
    assert "file_9.py" not in context


def test_citation_validator_complex():
    valid_files = {
        "auth.py": (1, 100),
        "db/models.py": (1, 200),
    }

    # All valid citations
    text1 = "See [auth.py:10-25] and [db/models.py:50-60]."
    c1, g1 = validate_citations(text1, valid_files)
    assert g1 is True
    assert len(c1) == 2

    # Out of line bounds citation
    text2 = "See [auth.py:150-160]."
    c2, g2 = validate_citations(text2, valid_files)
    assert g2 is False

    # Missing file citation
    text3 = "See [utils.py:10-20]."
    c3, g3 = validate_citations(text3, valid_files)
    assert g3 is False


def test_hybrid_search_and_graph_enrichment(temp_workspace):
    db_path = os.path.join(temp_workspace, "test.db")
    chroma_dir = os.path.join(temp_workspace, "chroma")

    db = DatabaseManager(db_path)
    db.init_schema()

    vstore = CodeVectorStore(persist_directory=chroma_dir)
    embedder = LocalEmbeddingClient()

    s_caller = CodeSymbol(
        symbol_id="api.py::::login_endpoint",
        file_path="api.py",
        name="login_endpoint",
        symbol_type="function",
        start_line=1,
        end_line=10,
        signature="def login_endpoint():",
        docstring="Handles login HTTP endpoint.",
        code_body="def login_endpoint():\n    return authenticate_user()",
    )
    s_callee = CodeSymbol(
        symbol_id="auth.py::::authenticate_user",
        file_path="auth.py",
        name="authenticate_user",
        symbol_type="function",
        start_line=1,
        end_line=10,
        signature="def authenticate_user():",
        docstring="Authenticates user credentials.",
        code_body="def authenticate_user():\n    return True",
    )

    db.save_repository_analysis(
        repo_id="test_repo",
        symbols=[s_caller, s_callee],
        files=["api.py", "auth.py"],
    )

    # Index into vector store
    vstore.index_symbols(repo_id="test_repo", symbols=[s_caller, s_callee], embedder=embedder)

    # Build NetworkX graph
    graph = build_code_graph(
        symbols=[s_caller, s_callee],
        call_edges=[CallEdge(caller=s_caller.symbol_id, callee=s_callee.symbol_id, line_no=5)],
    )

    # Perform hybrid search for "authenticate"
    candidates = hybrid_search(
        repo_id="test_repo",
        query="authenticate user credentials",
        vstore=vstore,
        db=db,
        embedder=embedder,
        top_n=2,
    )
    assert len(candidates) > 0
    assert any("authenticate_user" in c["id"] for c in candidates)

    # Enrich with graph: caller (login_endpoint) should be pulled in
    enriched = enrich_candidates_with_graph(candidates, graph, db, hops=1)
    enriched_ids = [c["id"] for c in enriched]
    assert s_caller.symbol_id in enriched_ids
    assert s_callee.symbol_id in enriched_ids
