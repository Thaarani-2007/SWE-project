import os
import shutil
import tempfile
import pytest

from core.models import CodeSymbol
from storage.db import DatabaseManager
from storage.vector_store import CodeVectorStore
from llm.embeddings import LocalEmbeddingClient
from llm.summarizer import FunctionSummarizer


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_embedding_client_basic_and_fallback():
    embedder = LocalEmbeddingClient()
    texts = ["def calculate(): pass", "def process(): pass"]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[1]) == len(vectors[0])

    # Test deterministic fallback directly
    fallback_vecs = embedder._fallback_deterministic_embeddings(["test a", "test b"], dim=128)
    assert len(fallback_vecs) == 2
    assert len(fallback_vecs[0]) == 128


def test_vector_store_indexing_and_batching(temp_workspace):
    chroma_dir = os.path.join(temp_workspace, "chroma")
    vstore = CodeVectorStore(persist_directory=chroma_dir)
    embedder = LocalEmbeddingClient()

    symbols = [
        CodeSymbol(
            symbol_id=f"mod.py::::func_{i}",
            file_path="mod.py",
            name=f"func_{i}",
            symbol_type="function",
            start_line=i * 10,
            end_line=i * 10 + 5,
            signature=f"def func_{i}():",
            docstring=f"Documentation for function {i}",
            code_body=f"def func_{i}():\n    return {i}",
        )
        for i in range(5)
    ]

    # Test indexing with small batch size to test batch iteration
    vstore.index_symbols(repo_id="batch_repo", symbols=symbols, embedder=embedder, batch_size=2)
    assert vstore.code_chunks.count() == 5

    # Test indexing summaries
    summaries = [
        {"symbol_id": f"mod.py::::func_{i}", "summary": f"Computes value {i}."}
        for i in range(5)
    ]
    vstore.index_summaries(repo_id="batch_repo", summaries=summaries, embedder=embedder, batch_size=2)
    assert vstore.code_summaries.count() == 5

    # Search query
    res = vstore.search(repo_id="batch_repo", query="function 2 documentation", n_results=2, embedder=embedder)
    assert len(res) > 0
    assert any("func_2" in r["symbol_id"] for r in res)

    # Test delete repo
    vstore.delete_repo("batch_repo")
    assert vstore.code_chunks.count() == 0
    assert vstore.code_summaries.count() == 0


def test_summarizer_fallback_and_db_persistence(temp_workspace):
    db_path = os.path.join(temp_workspace, "test.db")
    db = DatabaseManager(db_path)
    db.init_schema()

    mock_sym = CodeSymbol(
        symbol_id="service.py::::auth",
        file_path="service.py",
        name="auth",
        symbol_type="function",
        start_line=1,
        end_line=10,
        signature="def auth(token: str) -> bool:",
        docstring="Authenticates user token.",
        code_body="def auth(token: str) -> bool:\n    return len(token) > 0",
        parameters=["token: str"],
    )

    db.save_repository_analysis(repo_id="repo1", symbols=[mock_sym], files=["service.py"])

    # Offline/fallback test
    summarizer = FunctionSummarizer(base_url="http://localhost:11434", model="offline_model_xyz", timeout=1.0)
    summary = summarizer.summarize_symbol_sync(mock_sym)

    assert "Authenticates user token" in summary
    assert "auth" in summary

    # Batch test with DB persistence
    batch_results = summarizer.summarize_symbols_batch_sync(symbols=[mock_sym], db=db, max_concurrency=2)
    assert len(batch_results) == 1
    assert batch_results[0]["symbol_id"] == "service.py::::auth"

    saved_summary = db.get_function_summary("service.py::::auth")
    assert saved_summary is not None
    assert "Authenticates user token" in saved_summary["summary_text"]
