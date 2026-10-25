"""RAG package for context construction, answer generation, and citation validation."""
from rag.context_builder import build_rag_context
from rag.generator import validate_citations, generate_answer

__all__ = ["build_rag_context", "validate_citations", "generate_answer"]
