import io
import os
import zipfile
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import db
from core.models import CodeSymbol


@pytest.fixture
def client():
    return TestClient(app)


def test_list_repos_endpoint(client):
    res = client.get("/api/repos")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_upload_non_zip_rejected(client):
    fake_txt = io.BytesIO(b"not a zip file")
    res = client.post(
        "/api/upload",
        files={"file": ("code.txt", fake_txt, "text/plain")},
    )
    assert res.status_code == 400
    assert "zip" in res.json()["detail"].lower()


def test_upload_valid_zip_and_status(client):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("sample.py", "def greet(name: str) -> str:\n    return f'Hello {name}'\n")
    zip_buf.seek(0)

    res = client.post(
        "/api/upload",
        files={"file": ("sample_project.zip", zip_buf, "application/zip")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "repo_id" in data
    repo_id = data["repo_id"]

    status_res = client.get(f"/api/status/{repo_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["repo_id"] == repo_id
    assert "status" in status_data


def test_status_non_existent_repo(client):
    res = client.get("/api/status/non_existent_999")
    assert res.status_code == 404


def test_symbol_endpoint(client):
    mock_sym = CodeSymbol(
        symbol_id="test_route.py::::sample_func",
        file_path="test_route.py",
        name="sample_func",
        symbol_type="function",
        start_line=1,
        end_line=5,
        signature="def sample_func():",
        docstring="Docstring test.",
        code_body="def sample_func():\n    return 42",
    )
    db.save_repository_analysis(
        repo_id="route_test_repo",
        symbols=[mock_sym],
        files=["test_route.py"],
    )

    res = client.get("/api/symbol/test_route.py::::sample_func")
    assert res.status_code == 200
    assert res.json()["name"] == "sample_func"

    res_404 = client.get("/api/symbol/not_found_sym")
    assert res_404.status_code == 404


def test_file_content_security_and_fetch(client):
    repo_id = "security_test"
    repo_dir = os.path.join("data", "repos", repo_id)
    os.makedirs(repo_dir, exist_ok=True)
    with open(os.path.join(repo_dir, "app.py"), "w", encoding="utf-8") as f:
        f.write("# valid code\nprint('hello')\n")

    # Valid fetch
    res = client.get(f"/api/file_content?repo_id={repo_id}&path=app.py")
    assert res.status_code == 200
    assert "print('hello')" in res.json()["content"]

    # Traversal attack
    res_bad = client.get(f"/api/file_content?repo_id={repo_id}&path=../../etc/passwd")
    assert res_bad.status_code in (400, 404)
