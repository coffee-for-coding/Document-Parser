"""Two JSON log files:

* ``session_output.json`` — appends every search result during the process
  lifetime. Reset to ``[]`` when the server starts (fresh session).
* ``elastic_output.json`` — the raw Elasticsearch response from the most
  recent search only. Overwritten per iteration; reset on server start.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

DATA_DIR = Path("./data")
SESSION_PATH = DATA_DIR / "session_output.json"
ELASTIC_PATH = DATA_DIR / "elastic_output.json"

_lock = Lock()


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def reset_session() -> None:
    _ensure()
    with _lock:
        SESSION_PATH.write_text("[]", encoding="utf-8")
        ELASTIC_PATH.write_text(json.dumps(
            {"note": "Awaiting first search iteration.",
             "last_reset": datetime.utcnow().isoformat() + "Z"},
            indent=2), encoding="utf-8")


def append_session(entry: Dict[str, Any]) -> None:
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


def write_elastic(payload: Dict[str, Any]) -> None:
    """Overwrites elastic_output.json with the latest iteration's raw ES data."""
    _ensure()
    with _lock:
        ELASTIC_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False,
                       default=str),
            encoding="utf-8")
