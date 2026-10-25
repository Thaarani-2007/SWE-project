import ast
import os
from typing import Dict, List, Optional, Set, Tuple, Union

from core.models import CallEdge, CodeSymbol, ImportStatement, RepoAnalysisResult


def _build_module_maps(repo_analysis: RepoAnalysisResult) -> Tuple[Dict[str, str], Dict[Tuple[str, str], CodeSymbol]]:
    """Builds lookup structures:

    1. module_name_to_file: dotted module name -> rel_file_path (e.g. 'file_a' -> 'file_a.py', 'pkg.sub' -> 'pkg/sub.py')
    2. symbol_by_file_and_name: (rel_file_path, symbol_name) -> CodeSymbol
    """
    module_name_to_file: Dict[str, str] = {}
    symbol_by_file_and_name: Dict[Tuple[str, str], CodeSymbol] = {}

    for sym in repo_analysis.symbols:
        # Index top-level functions and classes by (file_path, name)
        if sym.symbol_type in ("function", "class"):
            symbol_by_file_and_name[(sym.file_path, sym.name)] = sym
        elif sym.symbol_type == "method" and sym.parent_symbol:
            # Also allow indexing method as (file_path, "Class::method")
            symbol_by_file_and_name[(sym.file_path, sym.symbol_id.split("::")[-1])] = sym

    for mod in repo_analysis.modules:
        rel_path = mod.file_path.replace("\\", "/")
        # Construct module name without .py
        no_ext, _ = os.path.splitext(rel_path)
        parts = [p for p in no_ext.split("/") if p]

        if parts and parts[-1] == "__init__":
            dotted = ".".join(parts[:-1])
        else:
            dotted = ".".join(parts)

        if dotted:
            module_name_to_file[dotted] = rel_path
        # Also map filename alone for root-level modules
        filename = os.path.basename(rel_path)
        base_no_ext, _ = os.path.splitext(filename)
        if base_no_ext not in module_name_to_file:
            module_name_to_file[base_no_ext] = rel_path

    return module_name_to_file, symbol_by_file_and_name


def _resolve_import_target(
    imp: ImportStatement,
    current_file_path: str,
    module_name_to_file: Dict[str, str]
) -> Optional[str]:
    """Finds the target relative file path for an import statement."""
    current_file_path = current_file_path.replace("\\", "/")
    current_dir = os.path.dirname(current_file_path)

    # 1. Relative import (level > 0)
    if imp.level > 0:
        dir_parts = [p for p in current_dir.split("/") if p]
        # level 1 is current dir, level 2 is parent dir, etc.
        up_count = imp.level - 1
        if up_count <= len(dir_parts):
            base_dir = "/".join(dir_parts[:len(dir_parts) - up_count]) if up_count < len(dir_parts) else ""
            if imp.module:
                mod_subpath = imp.module.replace(".", "/")
                rel_candidate = f"{base_dir}/{mod_subpath}.py" if base_dir else f"{mod_subpath}.py"
            else:
                rel_candidate = f"{base_dir}.py" if base_dir else ""

            if rel_candidate in module_name_to_file.values():
                return rel_candidate

            # Check __init__.py package
            pkg_candidate = f"{base_dir}/{imp.module.replace('.', '/')}/__init__.py" if (base_dir and imp.module) else (f"{imp.module.replace('.', '/')}/__init__.py" if imp.module else "")
            if pkg_candidate in module_name_to_file.values():
                return pkg_candidate

    # 2. Dotted or root-relative module lookup
    if imp.module in module_name_to_file:
        return module_name_to_file[imp.module]

    # 3. Check relative to current file directory
    if current_dir and imp.module:
        sub_candidate = f"{current_dir}/{imp.module}".replace(".", "/") + ".py"
        if sub_candidate in module_name_to_file.values():
            return sub_candidate

    return None


class CallVisitor(ast.NodeVisitor):
    """Visits function/method ASTs and collects CallEdge relationships."""

    def __init__(
        self,
        rel_file_path: str,
        file_scope: Dict[str, str],
        symbol_by_id: Dict[str, CodeSymbol],
        symbol_by_file_and_name: Dict[Tuple[str, str], CodeSymbol],
        all_call_edges: List[CallEdge],
    ):
        self.rel_file_path = rel_file_path
        self.file_scope = file_scope
        self.symbol_by_id = symbol_by_id
        self.symbol_by_file_and_name = symbol_by_file_and_name
        self.all_call_edges = all_call_edges
        self.current_func_id: Optional[str] = None
        self.current_class_name: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        prev_class = self.current_class_name
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_func(node)

    def _handle_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        prev_func = self.current_func_id
        class_part = self.current_class_name if self.current_class_name else ""
        self.current_func_id = f"{self.rel_file_path}::{class_part}::{node.name}"
        self.generic_visit(node)
        self.current_func_id = prev_func

    def visit_Call(self, node: ast.Call):
        if self.current_func_id:
            callee, is_external = self._resolve_call(node)
            edge = CallEdge(
                caller=self.current_func_id,
                callee=callee,
                line_no=node.lineno,
                is_external=is_external,
            )
            self.all_call_edges.append(edge)

            # Record in caller's callees list if symbol exists
            caller_sym = self.symbol_by_id.get(self.current_func_id)
            if caller_sym and callee not in caller_sym.callees:
                caller_sym.callees.append(callee)

        self.generic_visit(node)

    def _resolve_call(self, node: ast.Call) -> Tuple[str, bool]:
        # Case 1: Direct function call func_name(...)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.file_scope:
                target = self.file_scope[name]
                if not target.startswith("module::"):
                    return target, False
            # Check local file functions
            if (self.rel_file_path, name) in self.symbol_by_file_and_name:
                return self.symbol_by_file_and_name[(self.rel_file_path, name)].symbol_id, False
            return name, True

        # Case 2: Attribute call obj.method(...)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                # self.method(...) or cls.method(...)
                if obj_name in ("self", "cls") and self.current_class_name:
                    method_id = f"{self.rel_file_path}::{self.current_class_name}::{attr_name}"
                    if method_id in self.symbol_by_id:
                        return method_id, False
                    return attr_name, True

                # module_alias.func(...)
                if obj_name in self.file_scope:
                    target_ref = self.file_scope[obj_name]
                    if target_ref.startswith("module::"):
                        target_file = target_ref[len("module::"):]
                        if (target_file, attr_name) in self.symbol_by_file_and_name:
                            return self.symbol_by_file_and_name[(target_file, attr_name)].symbol_id, False
                        return f"{obj_name}.{attr_name}", True

                return attr_name, True

            return attr_name, True

        return "<dynamic_call>", True


def resolve_symbols_and_calls(repo_analysis: RepoAnalysisResult) -> None:
    """Resolves imports and function calls across all files in the repository.

    Updates repo_analysis.call_edges and individual module.call_edges.
    """
    module_name_to_file, symbol_by_file_and_name = _build_module_maps(repo_analysis)
    symbol_by_id: Dict[str, CodeSymbol] = {s.symbol_id: s for s in repo_analysis.symbols}
    all_call_edges: List[CallEdge] = []

    for mod in repo_analysis.modules:
        rel_path = mod.file_path.replace("\\", "/")
        file_scope: Dict[str, str] = {}

        # 1. Populate local definitions in file scope
        for sym in mod.symbols:
            if sym.symbol_type in ("function", "class"):
                file_scope[sym.name] = sym.symbol_id

        # 2. Resolve imports into file scope
        for imp in mod.imports:
            target_file = _resolve_import_target(imp, rel_path, module_name_to_file)
            local_name = imp.alias or imp.imported_name

            if target_file:
                if imp.is_from_import:
                    # Look up symbol in target file
                    if (target_file, imp.imported_name) in symbol_by_file_and_name:
                        target_sym = symbol_by_file_and_name[(target_file, imp.imported_name)]
                        file_scope[local_name] = target_sym.symbol_id
                    else:
                        # Could be a submodule import: from pkg import submod
                        submod_name = f"{imp.module}.{imp.imported_name}" if imp.module else imp.imported_name
                        if submod_name in module_name_to_file:
                            file_scope[local_name] = f"module::{module_name_to_file[submod_name]}"
                else:
                    # import module as alias
                    file_scope[local_name] = f"module::{target_file}"

        # 3. Parse AST of file to visit Calls with exact context
        full_path = os.path.join(repo_analysis.root_path, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=rel_path)
            mod_calls: List[CallEdge] = []
            visitor = CallVisitor(
                rel_file_path=rel_path,
                file_scope=file_scope,
                symbol_by_id=symbol_by_id,
                symbol_by_file_and_name=symbol_by_file_and_name,
                all_call_edges=mod_calls,
            )
            visitor.visit(tree)
            mod.call_edges = mod_calls
            all_call_edges.extend(mod_calls)
        except Exception as e:
            # Skip if error reading/parsing during call resolution
            pass

    repo_analysis.call_edges = all_call_edges
