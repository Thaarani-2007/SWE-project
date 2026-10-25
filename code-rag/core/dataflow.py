import ast
import os
from typing import List, Optional, Set, Tuple, Union

from core.models import DataFlowEdge, RepoAnalysisResult


class DataFlowVisitor(ast.NodeVisitor):
    """Intra-procedural data-flow visitor that traces variable assignments,

    arguments passed into downstream calls, parameter usage, and returns.
    """

    def __init__(self, rel_file_path: str):
        self.rel_file_path = rel_file_path.replace("\\", "/")
        self.current_class_name: Optional[str] = None
        self.current_func_id: Optional[str] = None
        self.edges: List[DataFlowEdge] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        prev_class = self.current_class_name
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._analyze_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._analyze_function(node)

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        prev_func = self.current_func_id
        class_part = self.current_class_name if self.current_class_name else ""
        self.current_func_id = f"{self.rel_file_path}::{class_part}::{node.name}"

        # Extract function parameter names
        params = [
            arg.arg for arg in getattr(node.args, "posonlyargs", [])
        ] + [
            arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")
        ] + [
            arg.arg for arg in getattr(node.args, "kwonlyargs", [])
        ]
        if node.args.vararg:
            params.append(node.args.vararg.arg)
        if node.args.kwarg:
            params.append(node.args.kwarg.arg)

        # Traverse only the statements inside the function body
        for stmt in node.body:
            self._analyze_statement(stmt, params)

        self.current_func_id = prev_func

    def _analyze_statement(self, stmt: ast.AST, params: List[str]):
        if isinstance(stmt, ast.Assign):
            self._handle_assign(stmt.targets, stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            self._handle_assign([stmt.target], stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._handle_call_expr(stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            self._handle_return(stmt.value, stmt.lineno)
        else:
            # Recursively inspect statements in blocks (if, for, while, try, with)
            for child in ast.iter_child_nodes(stmt):
                # Do not recurse into nested function or class definitions
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self._analyze_statement(child, params)

    def _extract_target_names(self, target_node: ast.AST) -> List[str]:
        names: List[str] = []
        for n in ast.walk(target_node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.append(n.id)
        if not names:
            names.append(ast.unparse(target_node))
        return names

    def _handle_assign(self, targets: List[ast.AST], value: ast.AST, line_no: int):
        target_names: List[str] = []
        for t in targets:
            target_names.extend(self._extract_target_names(t))

        val_unparsed = ast.unparse(value)

        # Case 1: Target = func_call(...)
        if isinstance(value, ast.Call):
            callee_name = ast.unparse(value.func)
            for t_name in target_names:
                self.edges.append(
                    DataFlowEdge(
                        source=val_unparsed,
                        target=t_name,
                        flow_type="ASSIGN",
                        line_no=line_no,
                        symbol_id=self.current_func_id,
                    )
                )

            # Trace arguments passed into call
            for arg in value.args:
                arg_expr = ast.unparse(arg)
                self.edges.append(
                    DataFlowEdge(
                        source=arg_expr,
                        target=callee_name,
                        flow_type="ARG_PASS",
                        line_no=line_no,
                        symbol_id=self.current_func_id,
                    )
                )
                # Argument flowing into target variable
                for t_name in target_names:
                    self.edges.append(
                        DataFlowEdge(
                            source=arg_expr,
                            target=t_name,
                            flow_type="DATA_FLOW",
                            line_no=line_no,
                            symbol_id=self.current_func_id,
                        )
                    )

            for kw in value.keywords:
                kw_expr = ast.unparse(kw.value)
                self.edges.append(
                    DataFlowEdge(
                        source=kw_expr,
                        target=callee_name,
                        flow_type="ARG_PASS",
                        line_no=line_no,
                        symbol_id=self.current_func_id,
                    )
                )

        # Case 2: Target = source_var or general expression
        else:
            for t_name in target_names:
                self.edges.append(
                    DataFlowEdge(
                        source=val_unparsed,
                        target=t_name,
                        flow_type="ASSIGN",
                        line_no=line_no,
                        symbol_id=self.current_func_id,
                    )
                )

            # Trace individual read variables into target
            for sub in ast.walk(value):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id != val_unparsed:
                        for t_name in target_names:
                            self.edges.append(
                                DataFlowEdge(
                                    source=sub.id,
                                    target=t_name,
                                    flow_type="DATA_FLOW",
                                    line_no=line_no,
                                    symbol_id=self.current_func_id,
                                )
                            )

    def _handle_call_expr(self, call_node: ast.Call, line_no: int):
        callee_name = ast.unparse(call_node.func)
        for arg in call_node.args:
            arg_expr = ast.unparse(arg)
            self.edges.append(
                DataFlowEdge(
                    source=arg_expr,
                    target=callee_name,
                    flow_type="ARG_PASS",
                    line_no=line_no,
                    symbol_id=self.current_func_id,
                )
            )
        for kw in call_node.keywords:
            kw_expr = ast.unparse(kw.value)
            self.edges.append(
                DataFlowEdge(
                    source=kw_expr,
                    target=callee_name,
                    flow_type="ARG_PASS",
                    line_no=line_no,
                    symbol_id=self.current_func_id,
                )
            )

    def _handle_return(self, value: ast.AST, line_no: int):
        ret_expr = ast.unparse(value)
        self.edges.append(
            DataFlowEdge(
                source=ret_expr,
                target="return",
                flow_type="RETURN",
                line_no=line_no,
                symbol_id=self.current_func_id,
            )
        )
        # Trace variables in return value to return
        for sub in ast.walk(value):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id != ret_expr:
                    self.edges.append(
                        DataFlowEdge(
                            source=sub.id,
                            target="return",
                            flow_type="RETURN",
                            line_no=line_no,
                            symbol_id=self.current_func_id,
                        )
                    )


def analyze_repo_dataflow(repo_analysis: RepoAnalysisResult) -> List[DataFlowEdge]:
    """Performs intra-procedural data-flow analysis across all modules in the repository.

    Traces variable assignments, parameters passed to downstream calls, and returns.
    Updates repo_analysis.dataflow_edges and returns the list of DataFlowEdge.
    """
    all_dataflow_edges: List[DataFlowEdge] = []

    for mod in repo_analysis.modules:
        rel_path = mod.file_path.replace("\\", "/")
        full_path = os.path.join(repo_analysis.root_path, rel_path)

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()

            tree = ast.parse(source, filename=rel_path)
            visitor = DataFlowVisitor(rel_file_path=rel_path)
            visitor.visit(tree)

            # Deduplicate edges while preserving order
            seen: Set[Tuple[Optional[str], str, str, str, Optional[int]]] = set()
            unique_edges: List[DataFlowEdge] = []
            for edge in visitor.edges:
                key = (edge.symbol_id, edge.source, edge.target, edge.flow_type, edge.line_no)
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(edge)

            mod.dataflow_edges = unique_edges
            all_dataflow_edges.extend(unique_edges)
        except Exception:
            # Skip if error reading/parsing file
            pass

    repo_analysis.dataflow_edges = all_dataflow_edges
    return all_dataflow_edges
