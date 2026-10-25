import os
from typing import Any, Dict, List, Optional

# pyrefly: ignore[missing-import]
import chromadb

from core.models import CodeSymbol
from llm.embeddings import LocalEmbeddingClient


class CodeVectorStore:
    """ChromaDB vector store managing persistent code_chunks and code_summaries collections

    with batch upserts and metadata filtering.
    """

    def __init__(self, persist_directory: str = "data/chromadb"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # 1. code_chunks collection
        self.code_chunks = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"description": "Source code chunks (functions, classes, modules)"},
        )

        # 2. code_summaries collection
        self.code_summaries = self.client.get_or_create_collection(
            name="code_summaries",
            metadata={"description": "LLM-generated function and module summaries"},
        )

    def index_symbols(
        self,
        repo_id: str,
        symbols: List[CodeSymbol],
        embedder: Optional[LocalEmbeddingClient] = None,
        batch_size: int = 100,
    ) -> None:
        """Indexes code symbols into the code_chunks collection in batches."""
        if not symbols:
            return
        if embedder is None:
            embedder = LocalEmbeddingClient()

        items = []
        for sym in symbols:
            # Combine signature, docstring, and code for semantic richness
            parts = []
            if sym.signature:
                parts.append(sym.signature)
            if sym.docstring:
                parts.append(sym.docstring)
            if sym.code_body:
                parts.append(sym.code_body)
            doc_text = "\n".join(parts) if parts else sym.name

            item_id = f"{repo_id}::code::{sym.symbol_id}"
            metadata = {
                "repo_id": repo_id,
                "file_path": sym.file_path,
                "symbol_id": sym.symbol_id,
                "name": sym.name,
                "start_line": int(sym.start_line),
                "end_line": int(sym.end_line),
                "type": "code",
            }
            items.append((item_id, doc_text, metadata))

        # Batch upsert
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            ids = [c[0] for c in chunk]
            documents = [c[1] for c in chunk]
            metadatas = [c[2] for c in chunk]
            embeddings = embedder.embed_texts(documents)

            self.code_chunks.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def index_summaries(
        self,
        repo_id: str,
        summaries: List[Dict[str, Any]],
        embedder: Optional[LocalEmbeddingClient] = None,
        batch_size: int = 100,
    ) -> None:
        """Indexes function summaries into the code_summaries collection in batches."""
        if not summaries:
            return
        if embedder is None:
            embedder = LocalEmbeddingClient()

        items = []
        for s in summaries:
            symbol_id = s.get("symbol_id", "")
            summary_text = s.get("summary") or s.get("summary_text", "")
            if not summary_text:
                continue

            # Extract fallback file_path and name from symbol_id
            parts = symbol_id.split("::")
            file_path = s.get("file_path") or (parts[0] if len(parts) > 0 else "")
            name = s.get("name") or (parts[-1] if len(parts) > 0 else "")

            item_id = f"{repo_id}::summary::{symbol_id}"
            metadata = {
                "repo_id": repo_id,
                "file_path": file_path,
                "symbol_id": symbol_id,
                "name": name,
                "type": "summary",
            }
            items.append((item_id, summary_text, metadata))

        # Batch upsert
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            ids = [c[0] for c in chunk]
            documents = [c[1] for c in chunk]
            metadatas = [c[2] for c in chunk]
            embeddings = embedder.embed_texts(documents)

            self.code_summaries.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def search(
        self,
        repo_id: str,
        query: str,
        n_results: int = 10,
        embedder: Optional[LocalEmbeddingClient] = None,
        target_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Performs semantic similarity search over code chunks and/or function summaries."""
        if embedder is None:
            embedder = LocalEmbeddingClient()

        query_vectors = embedder.embed_texts([query])
        if not query_vectors:
            return []
        query_vec = query_vectors[0]

        candidates: List[Dict[str, Any]] = []

        # 1. Search code_chunks (primary vector index)
        if target_collection in (None, "code_chunks"):
            count = self.code_chunks.count()
            if count > 0:
                k = min(n_results, count)
                try:
                    code_res = self.code_chunks.query(
                        query_embeddings=[query_vec],
                        n_results=k,
                        where={"repo_id": repo_id},
                    )
                    if code_res and code_res.get("ids") and code_res["ids"][0]:
                        for idx, item_id in enumerate(code_res["ids"][0]):
                            meta = code_res["metadatas"][0][idx] if code_res.get("metadatas") else {}
                            doc = code_res["documents"][0][idx] if code_res.get("documents") else ""
                            dist = code_res["distances"][0][idx] if code_res.get("distances") else 1.0
                            candidates.append({
                                "id": item_id,
                                "symbol_id": meta.get("symbol_id", ""),
                                "text": doc,
                                "metadata": meta,
                                "distance": dist,
                                "source_collection": "code_chunks",
                            })
                except Exception:
                    pass

        # 2. Search code_summaries (optional legacy fallback if summaries exist)
        if target_collection == "code_summaries" or (target_collection is None and self.code_summaries.count() > 0):
            count = self.code_summaries.count()
            if count > 0:
                k = min(n_results, count)
                try:
                    summary_res = self.code_summaries.query(
                        query_embeddings=[query_vec],
                        n_results=k,
                        where={"repo_id": repo_id},
                    )
                    if summary_res and summary_res.get("ids") and summary_res["ids"][0]:
                        for idx, item_id in enumerate(summary_res["ids"][0]):
                            meta = summary_res["metadatas"][0][idx] if summary_res.get("metadatas") else {}
                            doc = summary_res["documents"][0][idx] if summary_res.get("documents") else ""
                            dist = summary_res["distances"][0][idx] if summary_res.get("distances") else 1.0
                            candidates.append({
                                "id": item_id,
                                "symbol_id": meta.get("symbol_id", ""),
                                "text": doc,
                                "metadata": meta,
                                "distance": dist,
                                "source_collection": "code_summaries",
                            })
                except Exception:
                    pass

        # Sort candidates by distance (ascending)
        candidates.sort(key=lambda x: x["distance"])

        # Deduplicate by symbol_id keeping best score
        seen_symbols = set()
        deduped: List[Dict[str, Any]] = []
        for cand in candidates:
            sid = cand["symbol_id"]
            if sid and sid not in seen_symbols:
                seen_symbols.add(sid)
                deduped.append(cand)
            elif not sid:
                deduped.append(cand)

        return deduped[:n_results]

    def delete_repo(self, repo_id: str) -> None:
        """Deletes all chunks and summaries associated with a repository."""
        try:
            self.code_chunks.delete(where={"repo_id": repo_id})
        except Exception:
            pass
        try:
            self.code_summaries.delete(where={"repo_id": repo_id})
        except Exception:
            pass
