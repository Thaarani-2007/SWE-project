import logging
from typing import Any, Dict, List, Optional

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)


class CodeReranker:
    """Zero-cloud cross-encoder reranker powered by FlashRank (ms-marco-TinyBERT-L-2-v2)."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate code passages against the natural language query.

        Args:
            query: The user's natural language question or keyword query.
            documents: List of candidate dicts with {id, text, meta/metadata}.
            top_k: Number of top reranked items to return.

        Returns:
            List of top-K reranked documents with cross-encoder relevance scores.
        """
        if not documents:
            return []

        # Prepare passages for FlashRank (requires id, text, meta)
        passages: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            meta = doc.get("meta") or doc.get("metadata") or {}
            passages.append({
                "id": str(doc.get("id", i)),
                "text": doc.get("text", ""),
                "meta": meta,
            })

        try:
            req = RerankRequest(query=query, passages=passages)
            reranked = self.ranker.rerank(req)
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"FlashRank reranking failed ({e}). Returning un-reranked top-k.")
            return documents[:top_k]
