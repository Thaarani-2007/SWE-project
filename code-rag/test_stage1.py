import os
import sys
import shutil
import zipfile
import tempfile

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from core.extractor import safe_extract_zip
from core.ast_parser import parse_repository
from core.resolver import resolve_symbols_and_calls
from core.dataflow import analyze_repo_dataflow

def run_stage1_checkpoint():
    print("--- [STAGE 1 CHECKPOINT] Testing Static Analysis Pipeline ---")
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "sample_repo.zip")

    # 1. Create a dummy multi-file repo with imports, calls, and data-flow
    file_a = """
def process_data(raw_input: str) -> str:
    cleaned = raw_input.strip()
    return cleaned
"""
    file_b = """
from file_a import process_data

def handle_request(payload: str) -> str:
    validated = process_data(payload)
    result = validated.upper()
    return result
"""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file_a.py", file_a)
        zf.writestr("file_b.py", file_b)

    # 2. Test Extraction
    extract_target = os.path.join(temp_dir, "extracted")
    extracted_files = safe_extract_zip(zip_path, extract_target)
    assert len(extracted_files) == 2, f"Expected 2 extracted files, got {len(extracted_files)}"

    # 3. Test AST Parsing & Symbol Extraction
    repo_analysis = parse_repository(extract_target)
    symbols = {s.symbol_id: s for s in repo_analysis.symbols}
    assert "file_a.py::::process_data" in symbols or any("process_data" in k for k in symbols)
    assert any("handle_request" in k for k in symbols)

    # 4. Test Call Resolution
    resolve_symbols_and_calls(repo_analysis)
    call_edges = repo_analysis.call_edges
    assert any(c.caller.endswith("handle_request") and c.callee.endswith("process_data") for c in call_edges), \
        "Call resolver failed to link handle_request -> process_data"

    # 5. Test Data-Flow
    dataflow_edges = analyze_repo_dataflow(repo_analysis)
    assert len(dataflow_edges) > 0, "No dataflow edges detected"
    print(f"✅ Extracted {len(symbols)} symbols, {len(call_edges)} call edges, {len(dataflow_edges)} data-flow edges.")
    print("✅ STAGE 1 VERIFICATION PASSED.\n")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_stage1_checkpoint()
