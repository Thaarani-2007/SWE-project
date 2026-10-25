"""LLM package for embeddings and function summarization."""
from llm.embeddings import LocalEmbeddingClient
from llm.summarizer import FunctionSummarizer

__all__ = ["LocalEmbeddingClient", "FunctionSummarizer"]
