import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from core.models import CodeSymbol

logger = logging.getLogger(__name__)


class FunctionSummarizer:
    """Local LLM function summarizer using Ollama with concurrency limits and

    automatic fallback to docstrings and signatures on timeout/error.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
        self.timeout = timeout

    def _build_prompt(self, symbol: CodeSymbol) -> str:
        callees_str = ", ".join(symbol.callees) if symbol.callees else "none"
        doc_str = symbol.docstring.strip() if symbol.docstring else "None"
        code_str = symbol.code_body.strip() if symbol.code_body else symbol.signature

        return (
            "You are an expert static analysis assistant. Summarize the following Python function "
            "in exactly 2-3 sentences. Explain: 1) What it does, 2) Key inputs and outputs, "
            "3) Side effects or external dependencies.\n\n"
            f"Function Signature: {symbol.signature}\n"
            f"Docstring: {doc_str}\n"
            f"Callees: {callees_str}\n"
            "Source Code:\n"
            f"{code_str}\n\n"
            "Summary (2-3 concise sentences only):"
        )

    def _generate_fallback(self, symbol: CodeSymbol) -> str:
        """Fallback to docstring and function signature if LLM fails or times out."""
        sig = symbol.signature.strip() if symbol.signature else f"def {symbol.name}():"
        doc = symbol.docstring.strip() if symbol.docstring else ""
        if doc:
            return f"Function `{symbol.name}` ({sig}): {doc}"
        params_str = ", ".join(symbol.parameters) if symbol.parameters else "no parameters"
        return f"Function `{symbol.name}` with signature `{sig}` taking {params_str}."

    def summarize_symbol_sync(self, symbol: CodeSymbol) -> str:
        """Synchronously summarizes a single symbol via Ollama with automatic fallback."""
        prompt = self._build_prompt(symbol)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 150},
                    },
                )
                if res.status_code == 200:
                    text = res.json().get("response", "").strip()
                    if text:
                        return text
                logger.warning(
                    f"Ollama summarization returned status {res.status_code}: {res.text}. Using fallback."
                )
        except Exception as e:
            logger.warning(f"Ollama summarization call failed or timed out ({e}). Using fallback.")

        return self._generate_fallback(symbol)

    async def summarize_symbol(
        self, symbol: CodeSymbol, client: Optional[httpx.AsyncClient] = None
    ) -> str:
        """Asynchronously summarizes a single symbol via Ollama with automatic fallback."""
        prompt = self._build_prompt(symbol)
        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close = True

        try:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 150},
                },
            )
            if res.status_code == 200:
                text = res.json().get("response", "").strip()
                if text:
                    return text
            logger.warning(f"Ollama returned {res.status_code}. Using fallback.")
        except Exception as e:
            logger.warning(f"Ollama async call failed or timed out ({e}). Using fallback.")
        finally:
            if should_close:
                await client.aclose()

        return self._generate_fallback(symbol)

    async def summarize_symbols_batch(
        self,
        symbols: List[CodeSymbol],
        db: Optional[Any] = None,
        max_concurrency: int = 3,
    ) -> List[Dict[str, Any]]:
        """Processes a list of symbols concurrently using an asyncio.Semaphore(3) worker pool

        and optionally persists results into SQLite function_summaries table.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async def worker(sym: CodeSymbol) -> Dict[str, Any]:
                async with semaphore:
                    summary = await self.summarize_symbol(sym, client=client)
                    if db is not None:
                        try:
                            db.save_function_summary(sym.symbol_id, summary, self.model)
                        except Exception as db_err:
                            logger.warning(f"Failed to persist summary for {sym.symbol_id}: {db_err}")
                    return {
                        "symbol_id": sym.symbol_id,
                        "summary": summary,
                        "name": sym.name,
                        "file_path": sym.file_path,
                    }

            tasks = [worker(s) for s in symbols]
            results = await asyncio.gather(*tasks)

        return results

    def summarize_symbols_batch_sync(
        self,
        symbols: List[CodeSymbol],
        db: Optional[Any] = None,
        max_concurrency: int = 3,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for batch summarization."""
        return asyncio.run(self.summarize_symbols_batch(symbols, db=db, max_concurrency=max_concurrency))
