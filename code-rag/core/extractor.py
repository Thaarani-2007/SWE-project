import os
import shutil
import zipfile
from pathlib import PurePath
from typing import List, Optional

MAX_UNCOMPRESSED_SIZE = 150 * 1024 * 1024  # 150 MB
MAX_FILE_COUNT = 1000

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".tox",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "site-packages",
}


def safe_extract_zip(
    zip_path: str,
    extract_target: Optional[str] = None,
    repo_id: Optional[str] = None
) -> List[str]:
    """Safely extracts Python files from a ZIP archive to a destination directory.

    Guards against:
    - Zip Slip / path traversal (leading slashes, '..' components, escaping sandbox root).
    - Resource exhaustion (maximum 1,000 files, maximum 150MB uncompressed).
    - Non-Python / unwanted files (filters to only .py, skips __pycache__, .git, .venv).

    Args:
        zip_path: Path to the uploaded zip archive.
        extract_target: Directory where files should be extracted.
        repo_id: Optional repo ID used when extract_target is not provided (defaults to data/repos/<repo_id>/).

    Returns:
        List of absolute file paths to the extracted Python files.
    """
    if extract_target is None:
        if repo_id:
            extract_target = os.path.join("data", "repos", repo_id)
        else:
            raise ValueError("Either extract_target or repo_id must be provided.")

    target_root = os.path.abspath(extract_target)
    os.makedirs(target_root, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()

        if len(infolist) > MAX_FILE_COUNT:
            raise ValueError(
                f"Zip archive exceeds maximum allowed files: {len(infolist)} > {MAX_FILE_COUNT}"
            )

        total_uncompressed_size = 0
        extracted_files: List[str] = []

        for member in infolist:
            # Skip directory entries
            if member.is_dir() or member.filename.endswith(("/", "\\")):
                continue

            raw_name = member.filename

            # 1. Reject leading slashes or drive letters
            if raw_name.startswith(("/", "\\")) or (len(raw_name) > 1 and raw_name[1] == ":"):
                raise ValueError(f"Path traversal attempt detected (leading slash/drive): {raw_name}")

            # 2. Reject path traversal with '..'
            pure_path = PurePath(raw_name)
            parts = pure_path.parts
            if ".." in parts:
                raise ValueError(f"Path traversal attempt detected ('..'): {raw_name}")

            # 3. Canonical path check against extract target root
            dest_path = os.path.abspath(os.path.join(target_root, *parts))
            try:
                common = os.path.commonpath([target_root, dest_path])
            except ValueError:
                raise ValueError(f"Zip slip path traversal detected: {raw_name}")

            if common != target_root:
                raise ValueError(f"Zip slip path traversal detected: {raw_name}")

            # 4. Filter excluded directories
            lower_parts = {part.lower() for part in parts}
            if any(exc in lower_parts for exc in EXCLUDED_DIR_NAMES):
                continue

            # 5. Filter only Python source files (.py)
            if not raw_name.lower().endswith(".py"):
                continue

            # 6. Size limits check
            total_uncompressed_size += member.file_size
            if total_uncompressed_size > MAX_UNCOMPRESSED_SIZE:
                raise ValueError(
                    f"Zip archive exceeds maximum uncompressed size of {MAX_UNCOMPRESSED_SIZE / (1024*1024)}MB."
                )

            # Ensure destination directory exists and write file
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zf.open(member) as source, open(dest_path, "wb") as target:
                shutil.copyfileobj(source, target)

            extracted_files.append(dest_path)

        return extracted_files
