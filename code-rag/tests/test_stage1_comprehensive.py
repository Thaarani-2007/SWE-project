import os
import shutil
import tempfile
import zipfile
import pytest

from core.extractor import safe_extract_zip, MAX_FILE_COUNT, MAX_UNCOMPRESSED_SIZE
from core.ast_parser import parse_repository, parse_source_code
from core.resolver import resolve_symbols_and_calls
from core.dataflow import analyze_repo_dataflow
from core.models import CodeSymbol, CallEdge, DataFlowEdge, RepoAnalysisResult


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_extractor_security_traversal_double_dots(temp_workspace):
    zip_path = os.path.join(temp_workspace, "traversal.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.py", "print('hacked')")

    target = os.path.join(temp_workspace, "extracted")
    with pytest.raises(ValueError, match="Path traversal"):
        safe_extract_zip(zip_path, target)


def test_extractor_security_leading_slash(temp_workspace):
    zip_path = os.path.join(temp_workspace, "leading_slash.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("/evil.py", "print('hacked')")

    target = os.path.join(temp_workspace, "extracted")
    with pytest.raises(ValueError, match="Path traversal"):
        safe_extract_zip(zip_path, target)


def test_extractor_filters_non_py_and_excluded_dirs(temp_workspace):
    zip_path = os.path.join(temp_workspace, "sample.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("app/main.py", "x = 1")
        zf.writestr("app/__pycache__/main.cpython-312.pyc", "binary")
        zf.writestr(".venv/lib/dummy.py", "print(1)")
        zf.writestr("readme.txt", "documentation")
        zf.writestr("image.png", b"\x89PNG\r\n\x1a\n")

    target = os.path.join(temp_workspace, "extracted")
    extracted = safe_extract_zip(zip_path, target)

    assert len(extracted) == 1
    assert extracted[0].replace("\\", "/").endswith("app/main.py")
    assert not os.path.exists(os.path.join(target, "readme.txt"))
    assert not os.path.exists(os.path.join(target, ".venv"))


def test_ast_parser_syntax_error_resilience(temp_workspace):
    target = os.path.join(temp_workspace, "repo_with_syntax_err")
    os.makedirs(target, exist_ok=True)

    with open(os.path.join(target, "valid.py"), "w", encoding="utf-8") as f:
        f.write("def good():\n    return 42\n")

    with open(os.path.join(target, "broken.py"), "w", encoding="utf-8") as f:
        f.write("def broken(:\n    syntax error here\n")

    repo_analysis = parse_repository(target)
    symbol_names = [s.name for s in repo_analysis.symbols]

    assert "good" in symbol_names
    assert "broken" not in symbol_names


def test_ast_classes_and_methods():
    source = '''
"""Module docstring."""
class BaseService:
    pass

class UserService(BaseService):
    """User service docstring."""
    def __init__(self, db_conn: str):
        self.conn = db_conn

    def get_user(self, user_id: int) -> dict:
        """Fetches user."""
        return {"id": user_id}
'''
    mod_result = parse_source_code(source, "services/user.py")
    assert mod_result.docstring == "Module docstring."
    
    symbols_by_id = {s.symbol_id: s for s in mod_result.symbols}
    assert "services/user.py::::" in symbols_by_id
    assert "services/user.py::UserService::" in symbols_by_id
    assert "services/user.py::UserService::get_user" in symbols_by_id

    get_user_sym = symbols_by_id["services/user.py::UserService::get_user"]
    assert get_user_sym.symbol_type == "method"
    assert get_user_sym.name == "get_user"
    assert get_user_sym.docstring == "Fetches user."
    assert "user_id: int" in get_user_sym.signature
    assert "-> dict" in get_user_sym.signature


def test_resolver_and_dataflow_multi_file(temp_workspace):
    repo_dir = os.path.join(temp_workspace, "calc_repo")
    os.makedirs(os.path.join(repo_dir, "pkg"), exist_ok=True)

    math_ops = '''
def add(a: int, b: int) -> int:
    return a + b
'''
    calc = '''
from pkg.math_ops import add

class Calculator:
    def compute(self, x: int, y: int) -> int:
        total = add(x, y)
        return total
'''
    with open(os.path.join(repo_dir, "pkg", "math_ops.py"), "w", encoding="utf-8") as f:
        f.write(math_ops)

    with open(os.path.join(repo_dir, "calc.py"), "w", encoding="utf-8") as f:
        f.write(calc)

    repo_analysis = parse_repository(repo_dir)
    resolve_symbols_and_calls(repo_analysis)

    # Check call edge from compute -> add
    matching_calls = [
        c for c in repo_analysis.call_edges
        if c.caller.endswith("compute") and c.callee.endswith("add")
    ]
    assert len(matching_calls) >= 1
    assert not matching_calls[0].is_external

    # Check dataflow edges
    dataflow_edges = analyze_repo_dataflow(repo_analysis)
    assert len(dataflow_edges) > 0
    # Check that x, y pass to add or total
    assert any(e.source == "total" and e.target == "return" for e in dataflow_edges)
    assert any(e.target == "total" and e.flow_type == "ASSIGN" for e in dataflow_edges)


def test_dataflow_model_alias_compatibility():
    edge = DataFlowEdge(source_expr="x", sink_expr="y", relation_type="ARG_PASS")
    assert edge.source == "x"
    assert edge.target == "y"
    assert edge.flow_type == "ARG_PASS"
    assert edge.source_expr == "x"
    assert edge.sink_expr == "y"
    assert edge.relation_type == "ARG_PASS"
