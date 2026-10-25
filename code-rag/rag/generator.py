import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

# Citation format: [file_path:start_line-end_line] or [file_path:line]
CITATION_PATTERN = re.compile(
    r"\[([a-zA-Z0-9_\-\./\\]+\.py):(\d+)(?:-(\d+))?\]"
)

SYSTEM_PROMPT = (
    "You are an expert software architect answering questions about a codebase.\n"
    "Answer the user's question strictly based on the provided source code evidence.\n"
    "Mandate citations in format: [file_path:start_line-end_line].\n"
    "Do not extrapolate or speculate beyond what is explicitly shown in the context."
)


def validate_citations(
    response_text: str,
    valid_files: Union[Dict[str, Tuple[int, int]], Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Validates citations in the LLM response against known repository files and line boundaries.

    Args:
        response_text: The LLM generated answer.
        valid_files: Dict mapping file_path -> (start_line, end_line) or DatabaseManager instance.

    Returns:
        Tuple of (validated_citations_list, is_grounded_bool).
    """
    matches = list(CITATION_PATTERN.finditer(response_text))
    if not matches:
        return [], True

    validated_citations: List[Dict[str, Any]] = []
    has_hallucinations = False

    # Extract valid file ranges
    file_bounds: Dict[str, Tuple[int, int]] = {}
    if isinstance(valid_files, dict):
        for k, v in valid_files.items():
            norm_k = k.replace("\\", "/")
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                file_bounds[norm_k] = (int(v[0]), int(v[1]))
            else:
                file_bounds[norm_k] = (1, 100000)
    elif hasattr(valid_files, "get_files_by_repo"):
        # Valid files passed as DatabaseManager
        pass

    for match in matches:
        full_raw = match.group(0)
        file_path = match.group(1).replace("\\", "/")
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line

        # Check if cited file exists in valid_files
        matched_file_key = None
        for k in file_bounds:
            if k == file_path or k.endswith("/" + file_path) or file_path.endswith("/" + k):
                matched_file_key = k
                break

        if matched_file_key is None:
            # File does not exist: hallucination
            has_hallucinations = True
            continue

        min_line, max_line = file_bounds[matched_file_key]
        # Check line intersection
        if start_line > max_line or end_line < min_line:
            has_hallucinations = True
            continue

        validated_citations.append({
            "file": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "raw": full_raw,
        })

    is_grounded = (not has_hallucinations) and (len(validated_citations) > 0)
    return validated_citations, is_grounded


def generate_answer(
    question: str,
    context: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> str:
    """Generates an answer from the local Ollama LLM with strict grounding and citations."""
    base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    model_name = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"SOURCE CODE EVIDENCE:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSWER (with [file_path:start_line-end_line] citations):"
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            return f"LLM error: HTTP {res.status_code} - {res.text}"
    except Exception as e:
        return f"LLM generation failed: {e}"
