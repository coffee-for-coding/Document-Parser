"""SQLite persistence layer.

Two files:

* ``data/db/inputs.db`` — one row per ingested document (primary key = doc_id).
  Holds file-hash + paths to the on-disk chunks / pages / faiss artefacts so
  a second ingest of the *same* file short-circuits and reuses them.

* ``data/db/outputs.db`` — one row per search. Foreign-keys back to the input
  via ``doc_id`` and points at the per-search JSON files on disk.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path("./data/db")
INPUTS_DB = DB_DIR / "inputs.db"
OUTPUTS_DB = DB_DIR / "outputs.db"

_inputs_lock = threading.Lock()
_outputs_lock = threading.Lock()


def _ensure() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ schema
def _init_inputs() -> None:
    with sqlite3.connect(INPUTS_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id       TEXT PRIMARY KEY,
                filename     TEXT,
                file_hash    TEXT NOT NULL,
                file_size    INTEGER,
                num_pages    INTEGER,
                num_chunks   INTEGER,
                embed_model  TEXT,
                ingested_at  TEXT NOT NULL,
                chunks_path  TEXT,
                pages_path   TEXT,
                faiss_path   TEXT,
                faiss_meta_path TEXT,
                manifest_path TEXT
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_file_hash "
                   "ON documents(file_hash)")


def _init_outputs() -> None:
    with sqlite3.connect(OUTPUTS_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id       TEXT,
                query        TEXT NOT NULL,
                lang_hint    TEXT,
                use_llm      INTEGER,
                num_pages    INTEGER,
                timestamp    TEXT NOT NULL,
                duration_ms  INTEGER,
                session_json TEXT,
                elastic_json TEXT,
                top_pages    TEXT,
                llm_flags    TEXT
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_doc_id "
                   "ON search_logs(doc_id)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_ts "
                   "ON search_logs(timestamp)")


def init() -> None:
    _ensure()
    _init_inputs()
    _init_outputs()


# ------------------------------------------------------------------ inputs
def find_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    init()
    with _inputs_lock, sqlite3.connect(INPUTS_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM documents WHERE file_hash = ? "
                         "ORDER BY ingested_at DESC LIMIT 1",
                         (file_hash,)).fetchone()
        return dict(row) if row else None


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    init()
    with _inputs_lock, sqlite3.connect(INPUTS_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM documents WHERE doc_id = ?",
                         (doc_id,)).fetchone()
        return dict(row) if row else None


def upsert_document(rec: Dict[str, Any]) -> None:
    init()
    cols = ("doc_id filename file_hash file_size num_pages num_chunks "
            "embed_model ingested_at chunks_path pages_path faiss_path "
            "faiss_meta_path manifest_path").split()
    vals = [rec.get(c) for c in cols]
    placeholders = ",".join("?" * len(cols))
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "doc_id")
    sql = (f"INSERT INTO documents ({','.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT(doc_id) DO UPDATE SET {updates}")
    with _inputs_lock, sqlite3.connect(INPUTS_DB) as cx:
        cx.execute(sql, vals)


def list_documents() -> List[Dict[str, Any]]:
    init()
    with _inputs_lock, sqlite3.connect(INPUTS_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            "SELECT doc_id, filename, file_hash, num_pages, num_chunks, "
            "ingested_at FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------------ outputs
def log_search(rec: Dict[str, Any]) -> int:
    init()
    cols = ("doc_id query lang_hint use_llm num_pages timestamp duration_ms "
            "session_json elastic_json top_pages llm_flags").split()
    vals = [
        rec.get("doc_id"),
        rec.get("query"),
        rec.get("lang_hint"),
        int(bool(rec.get("use_llm"))),
        rec.get("num_pages"),
        rec.get("timestamp"),
        rec.get("duration_ms"),
        rec.get("session_json"),
        rec.get("elastic_json"),
        json.dumps(rec.get("top_pages") or [], ensure_ascii=False),
        json.dumps(rec.get("llm_flags") or {}, ensure_ascii=False),
    ]
    placeholders = ",".join("?" * len(cols))
    with _outputs_lock, sqlite3.connect(OUTPUTS_DB) as cx:
        cur = cx.execute(
            f"INSERT INTO search_logs ({','.join(cols)}) "
            f"VALUES ({placeholders})", vals)
        return cur.lastrowid


def list_logs(doc_id: Optional[str] = None, limit: int = 100
              ) -> List[Dict[str, Any]]:
    init()
    with _outputs_lock, sqlite3.connect(OUTPUTS_DB) as cx:
        cx.row_factory = sqlite3.Row
        if doc_id:
            rows = cx.execute(
                "SELECT * FROM search_logs WHERE doc_id = ? "
                "ORDER BY log_id DESC LIMIT ?", (doc_id, limit)).fetchall()
        else:
            rows = cx.execute(
                "SELECT * FROM search_logs ORDER BY log_id DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]
