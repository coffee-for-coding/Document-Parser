"""JSON output persistence.

Live (session-scoped) files — reset on server start:
  * ``data/session_output.json``  — appended every search during this run.
  * ``data/elastic_output.json``  — overwritten with the most recent iteration.

Archived (per-event) files — never reset, grow over time:
  * ``data/sessions/<iso-ts>__<slug>.json``   — one file per search.
  * ``data/elastic/<iso-ts>__<slug>.json``    — one file per search.
  * ``data/ingests/<doc-id>__<iso-ts>.json``  — one manifest per ingest.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

DATA_DIR = Path("./data")
SESSION_PATH = DATA_DIR / "session_output.json"
ELASTIC_PATH = DATA_DIR / "elastic_output.json"

SESSIONS_DIR = DATA_DIR / "sessions"
ELASTIC_DIR = DATA_DIR / "elastic"
INGESTS_DIR = DATA_DIR / "ingests"

_lock = Lock()


def _ensure():
    for d in (DATA_DIR, SESSIONS_DIR, ELASTIC_DIR, INGESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def _slug(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "_", (s or "").strip().lower())
    return s[:maxlen].strip("_") or "query"


# --------------------------------------------------------------- lifecycle
def reset_session() -> None:
    """Clear the live session files. Archived files are left untouched."""
    _ensure()
    with _lock:
        SESSION_PATH.write_text("[]", encoding="utf-8")
        ELASTIC_PATH.write_text(json.dumps(
            {"note": "Awaiting first search iteration.",
             "last_reset": datetime.utcnow().isoformat() + "Z"},
            indent=2), encoding="utf-8")


# --------------------------------------------------------------- search logs
def append_session(entry: Dict[str, Any]) -> str:
    """Append to the live session file AND write a timestamped archive copy.
    Returns the archive file path (str)."""
    _ensure()
    with _lock:
        try:
            current: List[Dict[str, Any]] = json.loads(
                SESSION_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = []
        current.append(entry)
        SESSION_PATH.write_text(
            json.dumps(current, indent=2, ensure_ascii=False),
            encoding="utf-8")

        archive = SESSIONS_DIR / f"{_ts()}__{_slug(entry.get('query', ''))}.json"
        archive.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False),
            encoding="utf-8")
        return str(archive)


def write_elastic(payload: Dict[str, Any]) -> str:
    """Overwrite live elastic file AND write a timestamped archive copy.
    Returns the archive file path."""
    _ensure()
    with _lock:
        ELASTIC_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

        archive = ELASTIC_DIR / f"{_ts()}__{_slug(payload.get('query', ''))}.json"
        archive.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return str(archive)


# --------------------------------------------------------------- ingest logs
def write_ingest_manifest(doc_id: str, manifest: Dict[str, Any]) -> str:
    """Persist chunk metadata + file info at ingest time. Returns path."""
    _ensure()
    path = INGESTS_DIR / f"{_slug(doc_id, 40)}__{_ts()}.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    return str(path)
