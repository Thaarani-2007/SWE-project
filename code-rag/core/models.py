from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class CodeSymbol(BaseModel):
    """Represents an extracted code symbol (module, class, function, method)."""
    symbol_id: str
    file_path: str
    name: str
    symbol_type: str  # "module", "class", "function", "method"
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    code_body: str = ""
    parent_symbol: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    return_annotation: Optional[str] = None
    callees: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CallEdge(BaseModel):
    """Represents a call relationship between caller and callee symbols."""
    caller: str
    callee: str
    line_no: Optional[int] = None
    is_external: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataFlowEdge(BaseModel):
    """Represents a data-flow link between source and target expressions/symbols."""
    source: str = ""
    target: str = ""
    flow_type: str = "DATA_FLOW"
    line_no: Optional[int] = None
    symbol_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def handle_compat(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Allow source_expr / sink_expr / relation_type aliases
            if "source_expr" in data and not data.get("source"):
                data["source"] = data["source_expr"]
            if "sink_expr" in data and not data.get("target"):
                data["target"] = data["sink_expr"]
            if "relation_type" in data and not data.get("flow_type"):
                data["flow_type"] = data["relation_type"]
        return data

    @property
    def source_expr(self) -> str:
        return self.source

    @property
    def sink_expr(self) -> str:
        return self.target

    @property
    def relation_type(self) -> str:
        return self.flow_type


class ImportStatement(BaseModel):
    """Represents an imported module or symbol."""
    module: str
    imported_name: str
    alias: Optional[str] = None
    is_from_import: bool = False
    level: int = 0
    line_no: Optional[int] = None


class ModuleAnalysisResult(BaseModel):
    """Represents the static analysis results of a single Python source file."""
    file_path: str
    docstring: str = ""
    imports: List[ImportStatement] = Field(default_factory=list)
    symbols: List[CodeSymbol] = Field(default_factory=list)
    call_edges: List[CallEdge] = Field(default_factory=list)
    dataflow_edges: List[DataFlowEdge] = Field(default_factory=list)


class RepoAnalysisResult(BaseModel):
    """Represents the aggregated analysis results of a repository."""
    repo_id: Optional[str] = None
    root_path: str = ""
    modules: List[ModuleAnalysisResult] = Field(default_factory=list)
    symbols: List[CodeSymbol] = Field(default_factory=list)
    call_edges: List[CallEdge] = Field(default_factory=list)
    dataflow_edges: List[DataFlowEdge] = Field(default_factory=list)
