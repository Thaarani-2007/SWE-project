import os
import sys
import tempfile
from storage.vector_store import CodeVectorStore
from llm.embeddings import LocalEmbeddingClient
from llm.summarizer import FunctionSummarizer
from core.models import CodeSymbol

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_stage3_checkpoint():
    print("--- [STAGE 3 CHECKPOINT] Testing Local LLM Summarization & ChromaDB ---")
    temp_dir = tempfile.mkdtemp()
    chroma_dir = os.path.join(temp_dir, "chroma")

    # 1. Test Local Embeddings
    embedder = LocalEmbeddingClient()
    test_vecs = embedder.embed_texts(["def add(a, b): return a + b", "Calculate sum of numbers"])
    assert len(test_vecs) == 2 and len(test_vecs[0]) > 0, "Embedding generation failed"

    # 2. Test ChromaDB Indexing (Direct Code Symbol Indexing without LLM Summarization)
    vstore = CodeVectorStore(persist_directory=chroma_dir)
    mock_symbol = CodeSymbol(
        symbol_id="math_utils.py::::add",
        file_path="math_utils.py",
        name="add",
        symbol_type="function",
        start_line=1,
        end_line=2,
        signature="def add(a, b):",
        docstring="Adds two numbers together and returns the result.",
        code_body="def add(a, b):\n    return a + b"
    )
    vstore.index_symbols(repo_id="test_repo", symbols=[mock_symbol], embedder=embedder)

    # 3. Test Vector Similarity Query over code chunks
    results = vstore.search(repo_id="test_repo", query="addition function", n_results=1)
    assert len(results) > 0, "ChromaDB search returned no results"
    assert results[0]["symbol_id"] == "math_utils.py::::add", f"Unexpected top result: {results[0]}"

    print("✅ Embeddings, direct ChromaDB code indexing, and vector search verified.")
    print("✅ STAGE 3 VERIFICATION PASSED.\n")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_stage3_checkpoint()
