import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.services import (
    db,
    get_status,
    run_ingestion_pipeline,
    stream_query_pipeline,
    update_status,
    vstore,
)

router = APIRouter(prefix="/api")


class QueryRequest(BaseModel):
    repo_id: str
    question: str


@router.get("/repos")
def list_repositories() -> List[Dict[str, Any]]:
    """Lists all indexed repositories."""
    return db.list_repositories()


@router.delete("/repos/{repo_id}")
def delete_repository(repo_id: str) -> Dict[str, Any]:
    """Deletes a repository and all its files, SQLite metadata, ChromaDB vector records, and graph caches."""
    repo = db.get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found.")

    # 1. Delete from SQLite (cascades to files, symbols, summaries)
    db.delete_repository(repo_id)

    # 2. Delete vectors from ChromaDB collections
    try:
        vstore.delete_repo(repo_id)
    except Exception:
        pass

    # 3. Clean up extracted repository files on disk
    repo_dir = os.path.abspath(os.path.join("data", "repos", repo_id))
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)

    # 4. Clean up graph JSON on disk
    graph_path = os.path.abspath(os.path.join("data", "graphs", f"{repo_id}.json"))
    if os.path.exists(graph_path):
        try:
            os.remove(graph_path)
        except Exception:
            pass

    return {"status": "deleted", "repo_id": repo_id}


@router.post("/upload")
async def upload_repository(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Uploads a ZIP codebase, creates the repository record, and triggers background ingestion."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archive files are supported.")

    repo_id = str(uuid.uuid4())[:8]
    repo_name = os.path.splitext(file.filename)[0]

    # Save uploaded archive to temporary file
    temp_dir = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_dir, f"{repo_id}.zip")

    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize SQLite repository entry
    db.save_repository_analysis(
        repo_id=repo_id,
        symbols=[],
        files=[],
        name=repo_name,
        status="EXTRACTING",
    )

    update_status(
        repo_id=repo_id,
        status="EXTRACTING",
        stage=1,
        percentage=5,
        message="Upload received, starting analysis pipeline...",
    )

    # Launch background worker
    background_tasks.add_task(run_ingestion_pipeline, repo_id, temp_zip_path, repo_name)

    return {
        "repo_id": repo_id,
        "name": repo_name,
        "message": "Upload successful, ingestion started.",
    }


@router.get("/status/{repo_id}")
def check_status(repo_id: str) -> Dict[str, Any]:
    """Returns ingestion progress and status for a repository."""
    status_info = get_status(repo_id)
    if status_info.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found.")
    return status_info


@router.post("/query")
async def query_codebase(request: QueryRequest):
    """Executes a hybrid RAG query with streaming tokens and validated citations via SSE."""
    repo = db.get_repo(request.repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repository '{request.repo_id}' not found.")

    return StreamingResponse(
        stream_query_pipeline(request.repo_id, request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/files/{repo_id}")
def get_repository_files(repo_id: str) -> List[Dict[str, Any]]:
    """Retrieves list of files for an indexed repository."""
    return db.get_files_by_repo(repo_id)


@router.get("/symbol/{symbol_id:path}")
def get_symbol_details(symbol_id: str) -> Dict[str, Any]:
    """Retrieves full code body and metadata for a symbol."""
    sym = db.get_symbol(symbol_id)
    if not sym:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol_id}' not found.")
    return sym.model_dump()


@router.get("/file_content")
def get_file_content(repo_id: str = Query(...), path: str = Query(...)) -> Dict[str, Any]:
    """Retrieves raw content of a repository source file for the citation viewer."""
    norm_path = path.replace("\\", "/").lstrip("/")
    full_path = os.path.abspath(os.path.join("data", "repos", repo_id, norm_path))
    repo_root = os.path.abspath(os.path.join("data", "repos", repo_id))

    if not full_path.startswith(repo_root):
        raise HTTPException(status_code=400, detail="Invalid file path.")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found.")

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return {"path": norm_path, "content": content}
