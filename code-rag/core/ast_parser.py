import ast
import logging
import os
from typing import List, Optional, Tuple, Union

from core.models import (
    CodeSymbol,
    ImportStatement,
    ModuleAnalysisResult,
    RepoAnalysisResult,
)

logger = logging.getLogger(__name__)

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".tox",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "site-packages",
}


def _format_function_signature(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
) -> str:
    """Reconstructs the function signature string."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args_str = ast.unparse(node.args)
    ret_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({args_str}){ret_str}:"


def _extract_parameters(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
) -> List[str]:
    """Extracts parameter names and optional type annotations."""
    params: List[str] = []
    # Positional only
    for arg in getattr(node.args, "posonlyargs", []):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        params.append(f"{arg.arg}{ann}")
    # Regular args
    for arg in node.args.args:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        params.append(f"{arg.arg}{ann}")
    # *args
    if node.args.vararg:
        ann = f": {ast.unparse(node.args.vararg.annotation)}" if node.args.vararg.annotation else ""
        params.append(f"*{node.args.vararg.arg}{ann}")
    # Keyword-only
    for arg in getattr(node.args, "kwonlyargs", []):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        params.append(f"{arg.arg}{ann}")
    # **kwargs
    if node.args.kwarg:
        ann = f": {ast.unparse(node.args.kwarg.annotation)}" if node.args.kwarg.annotation else ""
        params.append(f"**{node.args.kwarg.arg}{ann}")
    return params


def parse_source_code(
    source_code: str,
    rel_file_path: str
) -> ModuleAnalysisResult:
    """Parses a single Python file content into a ModuleAnalysisResult.

    Extracts:
    - Module docstring and imports.
    - Classes with inheritance, docstrings, line numbers, and methods.
    - Functions and methods with deterministic symbol IDs (<file>::<class>::<func>).
    """
    rel_file_path = rel_file_path.replace("\\", "/")
    lines = source_code.splitlines()
    tree = ast.parse(source_code, filename=rel_file_path)

    module_doc = ast.get_docstring(tree) or ""
    imports: List[ImportStatement] = []
    symbols: List[CodeSymbol] = []

    # Module symbol
    mod_name = os.path.splitext(os.path.basename(rel_file_path))[0]
    module_symbol_id = f"{rel_file_path}::::"
    symbols.append(
        CodeSymbol(
            symbol_id=module_symbol_id,
            file_path=rel_file_path,
            name=mod_name,
            symbol_type="module",
            start_line=1,
            end_line=len(lines) if lines else 1,
            signature=f"module {rel_file_path}",
            docstring=module_doc,
            code_body=source_code,
        )
    )

    # Helper to extract code body lines (1-indexed)
    def get_code_body(start_l: int, end_l: int) -> str:
        if not lines:
            return ""
        s = max(0, start_l - 1)
        e = min(len(lines), end_l)
        return "\n".join(lines[s:e])

    # Extract imports and top-level definitions
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportStatement(
                        module=alias.name,
                        imported_name=alias.name,
                        alias=alias.asname,
                        is_from_import=False,
                        level=0,
                        line_no=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(
                    ImportStatement(
                        module=mod,
                        imported_name=alias.name,
                        alias=alias.asname,
                        is_from_import=True,
                        level=node.level,
                        line_no=node.lineno,
                    )
                )
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            class_doc = ast.get_docstring(node) or ""
            base_classes = [ast.unparse(b) for b in node.bases]
            start_l = node.lineno
            end_l = getattr(node, "end_lineno", start_l)
            class_sig = f"class {class_name}({', '.join(base_classes)}):" if base_classes else f"class {class_name}:"
            class_symbol_id = f"{rel_file_path}::{class_name}::"

            symbols.append(
                CodeSymbol(
                    symbol_id=class_symbol_id,
                    file_path=rel_file_path,
                    name=class_name,
                    symbol_type="class",
                    start_line=start_l,
                    end_line=end_l,
                    signature=class_sig,
                    docstring=class_doc,
                    code_body=get_code_body(start_l, end_l),
                    metadata={"base_classes": base_classes},
                )
            )

            # Traverse methods within class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_start = item.lineno
                    method_end = getattr(item, "end_lineno", method_start)
                    method_sig = _format_function_signature(item)
                    method_doc = ast.get_docstring(item) or ""
                    method_params = _extract_parameters(item)
                    ret_ann = ast.unparse(item.returns) if item.returns else None
                    method_symbol_id = f"{rel_file_path}::{class_name}::{item.name}"

                    symbols.append(
                        CodeSymbol(
                            symbol_id=method_symbol_id,
                            file_path=rel_file_path,
                            name=item.name,
                            symbol_type="method",
                            start_line=method_start,
                            end_line=method_end,
                            signature=method_sig,
                            docstring=method_doc,
                            code_body=get_code_body(method_start, method_end),
                            parent_symbol=class_symbol_id,
                            parameters=method_params,
                            return_annotation=ret_ann,
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            func_doc = ast.get_docstring(node) or ""
            func_start = node.lineno
            func_end = getattr(node, "end_lineno", func_start)
            func_sig = _format_function_signature(node)
            func_params = _extract_parameters(node)
            ret_ann = ast.unparse(node.returns) if node.returns else None
            # Top-level function symbol_id: <rel_file_path>::::<func_name>
            func_symbol_id = f"{rel_file_path}::::{func_name}"

            symbols.append(
                CodeSymbol(
                    symbol_id=func_symbol_id,
                    file_path=rel_file_path,
                    name=func_name,
                    symbol_type="function",
                    start_line=func_start,
                    end_line=func_end,
                    signature=func_sig,
                    docstring=func_doc,
                    code_body=get_code_body(func_start, func_end),
                    parameters=func_params,
                    return_annotation=ret_ann,
                )
            )

    return ModuleAnalysisResult(
        file_path=rel_file_path,
        docstring=module_doc,
        imports=imports,
        symbols=symbols,
    )


def parse_repository(repo_path: str, repo_id: Optional[str] = None) -> RepoAnalysisResult:
    """Traverses repository root and parses all .py files into AST and symbol tables.

    Gracefully handles syntax errors in files by skipping them with a warning.
    """
    repo_root = os.path.abspath(repo_path)
    modules: List[ModuleAnalysisResult] = []
    all_symbols: List[CodeSymbol] = []

    for root, dirs, files in os.walk(repo_root):
        # Exclude directories
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIR_NAMES]

        for file in sorted(files):
            if not file.lower().endswith(".py"):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_root).replace("\\", "/")

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()

                mod_result = parse_source_code(source, rel_path)
                modules.append(mod_result)
                all_symbols.extend(mod_result.symbols)
            except SyntaxError as e:
                logger.warning(f"Syntax error parsing {rel_path} (line {e.lineno}): {e.msg}. Skipping file.")
            except Exception as e:
                logger.warning(f"Unexpected error parsing {rel_path}: {e}. Skipping file.")

    return RepoAnalysisResult(
        repo_id=repo_id,
        root_path=repo_root,
        modules=modules,
        symbols=all_symbols,
    )
