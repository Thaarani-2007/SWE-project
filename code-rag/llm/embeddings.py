import hashlib
import logging
import os
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class LocalEmbeddingClient:
    """Unified local embedding client supporting Ollama (/api/embed or /api/embeddings)

    and sentence-transformers with deterministic fallback.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self._st_model = None

        # Determine provider
        env_provider = os.environ.get("EMBEDDING_PROVIDER")
        if provider:
            self.provider = provider.lower()
        elif env_provider:
            self.provider = env_provider.lower()
        else:
            self.provider, detected_model = self._detect_provider()
            if detected_model and not model_name:
                model_name = detected_model

        if model_name:
            self.model_name = model_name
        elif self.provider == "ollama":
            self.model_name = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
        else:
            self.model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def _detect_provider(self) -> Tuple[str, Optional[str]]:
        """Checks if local Ollama is responsive AND has an embedding model; otherwise defaults to sentence-transformers."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name", "") for m in res.json().get("models", [])]
                    # Check if an embedding model is actually installed in Ollama
                    embed_models = [m for m in models if any(k in m for k in ("embed", "bge", "minilm"))]
                    if embed_models:
                        return "ollama", embed_models[0]
        except Exception:
            pass
        return "sentence-transformers", None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized vector embeddings for a list of text strings."""
        if not texts:
            return []

        if self.provider == "ollama":
            try:
                return self._embed_ollama(texts)
            except Exception as e:
                logger.warning(f"Ollama embedding failed ({e}), attempting sentence-transformers fallback...")
                try:
                    return self._embed_sentence_transformers(texts)
                except Exception as st_err:
                    logger.warning(f"sentence-transformers fallback also failed ({st_err}). Using deterministic fallback.")
                    return self._fallback_deterministic_embeddings(texts)
        else:
            try:
                return self._embed_sentence_transformers(texts)
            except Exception as e:
                logger.warning(f"sentence-transformers embedding failed ({e}). Attempting Ollama...")
                try:
                    return self._embed_ollama(texts)
                except Exception:
                    return self._fallback_deterministic_embeddings(texts)

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Calls Ollama's /api/embed or /api/embeddings endpoint."""
        with httpx.Client(timeout=self.timeout) as client:
            # Try new Ollama /api/embed (batch endpoint)
            try:
                res = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": texts},
                )
                if res.status_code == 200:
                    data = res.json()
                    if "embeddings" in data and len(data["embeddings"]) == len(texts):
                        return data["embeddings"]
            except Exception:
                pass

            # Fallback to legacy Ollama /api/embeddings (single item per request)
            results = []
            for text in texts:
                res = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                if res.status_code == 200:
                    results.append(res.json()["embedding"])
                else:
                    raise RuntimeError(f"Ollama embedding returned HTTP {res.status_code}: {res.text}")
            return results

    def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Uses sentence-transformers to generate embeddings locally."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            st_name = self.model_name if (self.provider == "sentence-transformers" and ("/" in self.model_name or "MiniLM" in self.model_name)) else "all-MiniLM-L6-v2"
            self._st_model = SentenceTransformer(st_name)
        embeddings = self._st_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _fallback_deterministic_embeddings(self, texts: List[str], dim: int = 384) -> List[List[float]]:
        """Deterministic pseudo-embedding for air-gapped test environments if both engines are unreachable."""
        embeddings: List[List[float]] = []
        for text in texts:
            vec: List[float] = []
            h = hashlib.sha256(text.encode("utf-8")).digest()
            for i in range(dim):
                byte_val = h[i % len(h)]
                vec.append((byte_val / 127.5) - 1.0)
            norm = sum(x * x for x in vec) ** 0.5
            norm = norm if norm > 1e-9 else 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings
