import os
import sqlite3
from typing import Any, Dict, List, Optional, Union

from core.models import CodeSymbol


class DatabaseManager:
    """Manages SQLite metadata storage for repositories, files, code symbols,

    and function summaries with atomic transactions.
    """

    def __init__(self, db_path: str = "data/metadata.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema()

    def init_schema(self) -> None:
        """Initializes database schema and indexes."""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    total_lines INTEGER DEFAULT 0,
                    FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE,
                    UNIQUE(repo_id, path)
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    file_id INTEGER,
                    name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    signature TEXT,
                    docstring TEXT,
                    code_body TEXT,
                    FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS function_summaries (
                    symbol_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
                CREATE INDEX IF NOT EXISTS idx_symbols_repo ON symbols(repo_id);
                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
            """)

    def save_repository_analysis(
        self,
        repo_id: str,
        symbols: List[CodeSymbol],
        files: Union[List[str], List[Dict[str, Any]]],
        name: Optional[str] = None,
        status: str = "READY",
    ) -> None:
        """Atomically saves repository metadata, file mappings, and extracted symbols."""
        repo_name = name or repo_id

        # Calculate max lines per file from symbols if not provided
        file_lines_map: Dict[str, int] = {}
        for sym in symbols:
            norm_path = sym.file_path.replace("\\", "/")
            file_lines_map[norm_path] = max(file_lines_map.get(norm_path, 0), sym.end_line)

        with self._conn:
            # 1. Upsert repository
            self._conn.execute(
                """
                INSERT INTO repositories (id, name, status)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status
                """,
                (repo_id, repo_name, status),
            )

            # 2. Insert files and build path -> file_id lookup
            file_id_map: Dict[str, int] = {}
            for item in files:
                if isinstance(item, str):
                    path = item.replace("\\", "/")
                    total_lines = file_lines_map.get(path, 0)
                else:
                    path = item.get("path", "").replace("\\", "/")
                    total_lines = item.get("total_lines", file_lines_map.get(path, 0))

                self._conn.execute(
                    """
                    INSERT INTO files (repo_id, path, total_lines)
                    VALUES (?, ?, ?)
                    ON CONFLICT(repo_id, path) DO UPDATE SET
                        total_lines = excluded.total_lines
                    """,
                    (repo_id, path, total_lines),
                )

                cur = self._conn.execute(
                    "SELECT id FROM files WHERE repo_id = ? AND path = ?",
                    (repo_id, path),
                )
                row = cur.fetchone()
                if row:
                    file_id_map[path] = row["id"]

            # 3. Upsert symbols
            for sym in symbols:
                norm_file_path = sym.file_path.replace("\\", "/")
                file_id = file_id_map.get(norm_file_path)

                # If file wasn't explicitly in files list, insert it now
                if file_id is None:
                    total_lines = file_lines_map.get(norm_file_path, sym.end_line)
                    self._conn.execute(
                        """
                        INSERT INTO files (repo_id, path, total_lines)
                        VALUES (?, ?, ?)
                        ON CONFLICT(repo_id, path) DO UPDATE SET
                            total_lines = excluded.total_lines
                        """,
                        (repo_id, norm_file_path, total_lines),
                    )
                    cur = self._conn.execute(
                        "SELECT id FROM files WHERE repo_id = ? AND path = ?",
                        (repo_id, norm_file_path),
                    )
                    row = cur.fetchone()
                    file_id = row["id"] if row else None
                    if file_id is not None:
                        file_id_map[norm_file_path] = file_id

                self._conn.execute(
                    """
                    INSERT INTO symbols (
                        id, repo_id, file_id, name, symbol_type,
                        start_line, end_line, signature, docstring, code_body
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        repo_id = excluded.repo_id,
                        file_id = excluded.file_id,
                        name = excluded.name,
                        symbol_type = excluded.symbol_type,
                        start_line = excluded.start_line,
                        end_line = excluded.end_line,
                        signature = excluded.signature,
                        docstring = excluded.docstring,
                        code_body = excluded.code_body
                    """,
                    (
                        sym.symbol_id,
                        repo_id,
                        file_id,
                        sym.name,
                        sym.symbol_type,
                        sym.start_line,
                        sym.end_line,
                        sym.signature,
                        sym.docstring,
                        sym.code_body,
                    ),
                )

    def get_symbol(self, symbol_id: str) -> Optional[CodeSymbol]:
        """Retrieves a single CodeSymbol by its symbol_id."""
        cur = self._conn.execute(
            """
            SELECT s.*, f.path as file_path
            FROM symbols s
            LEFT JOIN files f ON s.file_id = f.id
            WHERE s.id = ?
            """,
            (symbol_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        return CodeSymbol(
            symbol_id=row["id"],
            file_path=row["file_path"] or "",
            name=row["name"],
            symbol_type=row["symbol_type"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"] or "",
            docstring=row["docstring"] or "",
            code_body=row["code_body"] or "",
        )

    def get_symbols_by_repo(self, repo_id: str) -> List[CodeSymbol]:
        """Retrieves all CodeSymbols belonging to a repository."""
        cur = self._conn.execute(
            """
            SELECT s.*, f.path as file_path
            FROM symbols s
            LEFT JOIN files f ON s.file_id = f.id
            WHERE s.repo_id = ?
            ORDER BY s.file_id, s.start_line
            """,
            (repo_id,),
        )
        results: List[CodeSymbol] = []
        for row in cur.fetchall():
            results.append(
                CodeSymbol(
                    symbol_id=row["id"],
                    file_path=row["file_path"] or "",
                    name=row["name"],
                    symbol_type=row["symbol_type"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    signature=row["signature"] or "",
                    docstring=row["docstring"] or "",
                    code_body=row["code_body"] or "",
                )
            )
        return results

    def get_files_by_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        """Retrieves all file records for a repository."""
        cur = self._conn.execute(
            "SELECT * FROM files WHERE repo_id = ? ORDER BY path",
            (repo_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def save_function_summary(self, symbol_id: str, summary_text: str, model_name: str) -> None:
        """Saves or updates an LLM-generated summary for a function symbol."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO function_summaries (symbol_id, summary_text, model_name)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    model_name = excluded.model_name
                """,
                (symbol_id, summary_text, model_name),
            )

    def get_function_summary(self, symbol_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the summary for a given symbol_id."""
        cur = self._conn.execute(
            "SELECT * FROM function_summaries WHERE symbol_id = ?",
            (symbol_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_repositories(self) -> List[Dict[str, Any]]:
        """Lists all registered repositories."""
        cur = self._conn.execute("SELECT * FROM repositories ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]

    def get_repo(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves repository metadata by ID."""
        cur = self._conn.execute("SELECT * FROM repositories WHERE id = ?", (repo_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def update_repo_status(self, repo_id: str, status: str) -> None:
        """Updates the status of a repository."""
        with self._conn:
            self._conn.execute(
                "UPDATE repositories SET status = ? WHERE id = ?",
                (status, repo_id),
            )

    def delete_repository(self, repo_id: str) -> None:
        """Deletes a repository and all cascaded files, symbols, and summaries."""
        with self._conn:
            self._conn.execute("DELETE FROM repositories WHERE id = ?", (repo_id,))

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
