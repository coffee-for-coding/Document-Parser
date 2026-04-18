"""FastAPI server exposing ingest + search endpoints.

Run with:  uvicorn docsearch.server:app --reload
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import DocSearch
from . import output_log

app = FastAPI(title="DE/EN Search")

# Fresh session files on every (re)start.
output_log.reset_session()

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))
_ds: Optional[DocSearch] = None


def _get() -> DocSearch:
    global _ds
    if _ds is None:
        _ds = DocSearch()
        try:
            _ds.load()
        except FileNotFoundError:
            pass
    return _ds


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), doc_id: str = Form(...),
                 recreate: bool = Form(False)):
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        n = _get().ingest(tmp_path, doc_id, recreate=recreate)
        return {"doc_id": doc_id, "chunks": n}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/search")
def search(q: str, top: int = 20, lang: str = "auto", use_llm: bool = True):
    if lang not in ("auto", "de", "en"):
        lang = "auto"
    ds = _get()
    if ds.faiss is None:
        try:
            ds.load()
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="No document indexed yet.")
    ds.use_llm = bool(use_llm)
    ds.qx.use_llm = bool(use_llm)
    if use_llm and ds.llm is None:
        from .llm import Ollama
        ds.llm = Ollama()
        ds.qx._llm = ds.llm
    if not use_llm:
        ds.llm = None
        ds.qx._llm = None
    return JSONResponse(ds.search(q, top_pages=top, lang_hint=lang))


@app.get("/healthz")
def health():
    return {"status": "ok"}
