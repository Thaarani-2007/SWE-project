import io
import os
import sys
import zipfile
from fastapi.testclient import TestClient
from api.main import app

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_stage5_checkpoint():
    print("--- [STAGE 5 CHECKPOINT] Testing FastAPI Endpoints ---")
    client = TestClient(app)

    # 1. Test Health / Repo list
    res = client.get("/api/repos")
    assert res.status_code == 200, f"Failed /api/repos: {res.text}"

    # 2. Test ZIP Upload
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("calc.py", "def multiply(x, y): return x * y\n")
    zip_buffer.seek(0)

    upload_res = client.post(
        "/api/upload",
        files={"file": ("test_codebase.zip", zip_buffer, "application/zip")}
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    repo_id = upload_res.json()["repo_id"]
    assert repo_id is not None

    # 3. Test Ingestion Status endpoint
    status_res = client.get(f"/api/status/{repo_id}")
    assert status_res.status_code == 200
    assert "status" in status_res.json()

    print("✅ Upload, repository tracking, and status polling endpoints verified.")
    print("✅ STAGE 5 VERIFICATION PASSED.\n")

if __name__ == "__main__":
    run_stage5_checkpoint()
