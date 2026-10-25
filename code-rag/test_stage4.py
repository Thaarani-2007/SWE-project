import os
import sys
import tempfile
from retrieval.reranker import CodeReranker
from rag.context_builder import build_rag_context
from rag.generator import validate_citations

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_stage4_checkpoint():
    print("--- [STAGE 4 CHECKPOINT] Testing Hybrid Retrieval, FlashRank & Citations ---")

    # 1. Test FlashRank Cross-Encoder Reranker
    reranker = CodeReranker()
    passages = [
        {"id": "1", "text": "def calculate_taxes(income): return income * 0.2", "meta": {"file": "tax.py", "lines": "1-2"}},
        {"id": "2", "text": "def handle_user_login(user, pwd): verify_credentials(user, pwd)", "meta": {"file": "auth.py", "lines": "10-20"}},
        {"id": "3", "text": "def format_output_json(payload): return json.dumps(payload)", "meta": {"file": "json_util.py", "lines": "5-8"}}
    ]
    query = "How does user authentication work?"
    reranked = reranker.rerank(query=query, documents=passages, top_k=2)

    assert len(reranked) == 2, "FlashRank did not return requested top_k items"
    assert reranked[0]["id"] == "2", f"Expected login passage to rank #1 for auth query, got {reranked[0]['id']}"

    # 2. Test Context Builder formatting
    context_str = build_rag_context(reranked)
    assert "File: auth.py" in context_str, "Context builder omitted file metadata"
    assert "handle_user_login" in context_str, "Context builder omitted code snippet"

    # 3. Test Citation & Output Validator
    mock_llm_response = "Authentication is handled in `handle_user_login` [auth.py:10-20]. It checks credentials."
    valid_files = {"auth.py": (1, 50)}
    validated_citations, is_grounded = validate_citations(mock_llm_response, valid_files)
    assert is_grounded is True, "Valid citation was marked ungrounded"
    assert len(validated_citations) == 1 and validated_citations[0]["file"] == "auth.py"

    # Test hallucinated citation detection
    hallucinated_response = "Password hashes are salted in [security/crypto.py:99-105]."
    _, is_grounded_bad = validate_citations(hallucinated_response, valid_files)
    assert is_grounded_bad is False, "Hallucinated citation was not caught"

    print("✅ FlashRank reranking, context assembly, and citation validation verified.")
    print("✅ STAGE 4 VERIFICATION PASSED.\n")

if __name__ == "__main__":
    run_stage4_checkpoint()
