import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .config import CFG
from .parser import parse
from .preprocess import Chunk, chunk_page
from .embedder import Embedder
from .indexer_faiss import FaissIndex
from .query import ExpandedQuery, QueryExpander
from .ranker import chunk_scores, aggregate_pages, proximity_score
from .synonyms import expand_term, classify
from .llm import Ollama
from . import output_log
from . import db
from datetime import datetime
import hashlib
import time


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s or "").strip("_") or "doc"


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _highlight_all(text: str, terms, tag_class: str = "primary") -> str:
    """HTML-escape the full text and wrap every term occurrence in
    <mark class="<tag_class>">…</mark>. Matches are whole-word, case-insensitive."""
    if not text:
        return ""
    escaped = _html_escape(text)
    terms = [t for t in terms if t]
    if not terms:
        return escaped
    import re as _re
    pat = _re.compile(r"\b(" + "|".join(_re.escape(t) for t in terms) + r")\b",
                      _re.IGNORECASE)
    return pat.sub(lambda m: f'<mark class="{tag_class}">{m.group(0)}</mark>',
                   escaped)


def _highlight(text: str, terms: List[str], max_len: int = 180) -> str:
    if not terms:
        return text[:max_len]
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms if t) + r")\b",
                         re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return text[:max_len]
    start = max(0, m.start() - 60)
    end = min(len(text), m.end() + 100)
    snippet = text[start:end]
    snippet = pattern.sub(lambda x: f"<em>{x.group(0)}</em>", snippet)
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


class DocSearch:
    """Main orchestrator. ES is optional; FAISS + in-memory BM25 is the
    default so the app runs out-of-the-box without Elasticsearch."""

    def __init__(self, use_es: Optional[bool] = None,
                 use_llm: Optional[bool] = None):
        self.use_es = CFG.use_es if use_es is None else use_es
        self.use_llm = CFG.use_llm if use_llm is None else use_llm
        self.emb = Embedder()
        self.faiss: Optional[FaissIndex] = None
        self.chunks: List[Chunk] = []
        self.qx = QueryExpander(use_llm=self.use_llm)
        self.llm = Ollama() if self.use_llm else None

        self.pages: Dict[int, str] = {}
        self.doc_id: Optional[str] = None
        self.es = None
        if self.use_es:
            try:
                from .indexer_es import ESIndexer
                self.es = ESIndexer(dim=self.emb.dim)
                self.es.ensure_index()
            except Exception as e:
                print(f"[warn] Elasticsearch unavailable, falling back to "
                      f"in-memory BM25: {e}")
                self.es = None
                self.use_es = False

        # BM25 in-memory fallback
        self._bm25 = None
        self._bm25_tokens: List[List[str]] = []

    # ------------------------------------------------------------------ ingest
    def ingest(self, path: str, doc_id: str, recreate: bool = False,
               filename: Optional[str] = None) -> int:
        path = Path(path)
        filename = filename or path.name
        file_hash = _file_sha256(path)

        # Memoization: identical file content → reuse existing artefacts.
        existing = db.find_by_hash(file_hash)
        if existing and not recreate:
            if (Path(existing["chunks_path"]).exists()
                    and Path(existing["faiss_path"]).exists()):
                print(f"[cache] file hash {file_hash[:12]} already indexed "
                      f"as doc_id='{existing['doc_id']}' — reusing.")
                self._load_from_record(existing)
                # Re-register under the new doc_id if different, but share paths.
                if existing["doc_id"] != doc_id:
                    rec = dict(existing)
                    rec["doc_id"] = doc_id
                    rec["filename"] = filename
                    rec["ingested_at"] = datetime.utcnow().isoformat() + "Z"
                    db.upsert_document(rec)
                self.doc_id = doc_id
                return existing["num_chunks"]

        # Per-doc on-disk layout so multiple documents can coexist.
        base = Path(CFG.chunks_path).parent / "docs" / _safe(doc_id)
        base.mkdir(parents=True, exist_ok=True)
        chunks_path = base / "chunks.pkl"
        pages_path = base / "pages.pkl"
        faiss_path = base / "faiss.index"
        faiss_meta = base / "faiss_meta.pkl"

        # Parse + chunk
        all_chunks: List[Chunk] = []
        pages: Dict[int, str] = {}
        for page_no, text in parse(path):
            pages[page_no] = (pages.get(page_no, "") + "\n" + (text or "")).strip()
            all_chunks.extend(chunk_page(doc_id, page_no, text,
                                         CFG.chunk_tokens, CFG.chunk_overlap))
        if not all_chunks:
            print("[warn] no chunks extracted")
            return 0

        # Embed
        texts = [c.text for c in all_chunks]
        print(f"[info] embedding {len(texts)} chunks with {CFG.embed_model}")
        vecs = self.emb.encode(texts, batch_size=32)

        # Elasticsearch
        if self.use_es and self.es is not None:
            if recreate:
                self.es.ensure_index(recreate=True)
            self.es.bulk_index(all_chunks, vecs)

        # FAISS (per-doc)
        self.faiss = FaissIndex(self.emb.dim)
        meta = [{"chunk_id": c.chunk_id, "page_no": c.page_no,
                 "lang": c.lang, "tokens": c.tokens, "text": c.text,
                 "entities": c.entities, "lemma": c.lemma}
                for c in all_chunks]
        self.faiss.add(vecs, meta)
        self.faiss.save(path=str(faiss_path), meta_path=str(faiss_meta))
        # Also write canonical paths for the "last ingested" fallback loader.
        self.faiss.save()

        # Persist chunks + pages (per-doc + legacy canonical copies)
        self.chunks = all_chunks
        self.pages = pages
        Path(CFG.chunks_path).parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)
        with open(pages_path, "wb") as f:
            pickle.dump(pages, f)
        with open(CFG.chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)
        with open(CFG.pages_path, "wb") as f:
            pickle.dump(pages, f)

        self._build_bm25()

        # Ingest manifest + DB row
        ingested_at = datetime.utcnow().isoformat() + "Z"
        manifest = {
            "doc_id": doc_id,
            "filename": filename,
            "file_hash": file_hash,
            "file_size": path.stat().st_size,
            "ingested_at": ingested_at,
            "embed_model": CFG.embed_model,
            "num_pages": len(pages),
            "num_chunks": len(all_chunks),
            "chunks_path": str(chunks_path),
            "pages_path": str(pages_path),
            "faiss_path": str(faiss_path),
            "faiss_meta_path": str(faiss_meta),
            "chunks": [
                {"chunk_id": c.chunk_id, "page_no": c.page_no,
                 "lang": c.lang, "num_tokens": len(c.tokens),
                 "entities": c.entities,
                 "text_preview": c.text[:180]}
                for c in all_chunks
            ],
        }
        manifest_path = output_log.write_ingest_manifest(doc_id, manifest)
        db.upsert_document({
            "doc_id": doc_id, "filename": filename, "file_hash": file_hash,
            "file_size": path.stat().st_size,
            "num_pages": len(pages), "num_chunks": len(all_chunks),
            "embed_model": CFG.embed_model, "ingested_at": ingested_at,
            "chunks_path": str(chunks_path), "pages_path": str(pages_path),
            "faiss_path": str(faiss_path),
            "faiss_meta_path": str(faiss_meta),
            "manifest_path": manifest_path,
        })
        self.doc_id = doc_id
        print(f"[info] indexed doc_id='{doc_id}' ({len(all_chunks)} chunks); "
              f"manifest: {manifest_path}")
        return len(all_chunks)

    def _load_from_record(self, rec: Dict) -> None:
        """Rehydrate chunks/pages/faiss from the per-doc paths in a DB row."""
        with open(rec["chunks_path"], "rb") as f:
            self.chunks = pickle.load(f)
        if rec.get("pages_path") and Path(rec["pages_path"]).exists():
            with open(rec["pages_path"], "rb") as f:
                self.pages = pickle.load(f)
        self.faiss = FaissIndex.load(
            self.emb.dim,
            path=rec["faiss_path"],
            meta_path=rec["faiss_meta_path"],
        )
        self.doc_id = rec.get("doc_id")
        self._build_bm25()

    def load(self):
        if not Path(CFG.chunks_path).exists():
            raise FileNotFoundError("No ingested index found. Run `ingest` first.")
        with open(CFG.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        if Path(CFG.pages_path).exists():
            with open(CFG.pages_path, "rb") as f:
                self.pages = pickle.load(f)
        else:
            # Back-compat: rebuild page text from chunks (lossy but usable).
            self.pages = {}
            for c in self.chunks:
                self.pages[c.page_no] = (self.pages.get(c.page_no, "")
                                         + " " + c.text).strip()
        self.faiss = FaissIndex.load(self.emb.dim)
        self._build_bm25()

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        self._bm25_tokens = [c.tokens + c.lemma.split() for c in self.chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)

    # ------------------------------------------------------------------ search
    def _bm25_search_mem(self, eq: ExpandedQuery, k: int) -> Dict[str, Dict]:
        if self._bm25 is None:
            return {}
        query_terms = list(dict.fromkeys(
            eq.terms + eq.synonyms
            + [w.lower() for w in eq.translated.split() if w.isalpha()]
        ))
        scores = self._bm25.get_scores(query_terms)
        order = np.argsort(-scores)[:k]
        out = {}
        for idx in order:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            out[c.chunk_id] = {
                "score": float(scores[idx]),
                "src": {
                    "chunk_id": c.chunk_id, "page_no": c.page_no, "lang": c.lang,
                    "text": c.text, "tokens": c.tokens,
                    "entities": c.entities, "lemma": c.lemma,
                },
                "snippet": _highlight(c.text, query_terms),
            }
        return out

    def _count_matches(self, tokens: List[str], lang: str,
                       de_terms: set, en_terms: set):
        c = Counter(tokens)
        de_total = sum(c[t] for t in de_terms)
        en_total = sum(c[t] for t in en_terms)
        return de_total + en_total, de_total, en_total

    def search(self, query: str, top_pages: int = 20,
               lang_hint: str = "auto") -> Dict:
        if self.faiss is None:
            self.load()

        t_start = time.time()
        eq = self.qx.expand(query)
        if not eq.raw:
            return {"query": query, "expanded": eq.as_dict(),
                    "pages": [], "top_full": [], "page_counts": []}

        # Apply user-supplied language hint: forces bucket classification
        # so the primary query stays in its language and translations/synonyms
        # represent the "equivalent" in the other language.
        if lang_hint in ("de", "en"):
            eq.lang = lang_hint

        # Multi-vector query: raw + translation, mean-pooled & renormalized.
        q_texts = [eq.raw]
        if eq.translated and eq.translated.lower() != eq.raw.lower():
            q_texts.append(eq.translated)
        qvecs = self.emb.encode(q_texts)
        qvec = qvecs.mean(axis=0, keepdims=True)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)

        # BM25
        es_raw_dump: Dict = {"mode": "in-memory-bm25",
                             "timestamp": datetime.utcnow().isoformat() + "Z",
                             "query": query}
        if self.use_es and self.es is not None:
            bm25_hits = self.es.search(eq, CFG.top_k_bm25)
            es_raw_dump.update({
                "mode": "elasticsearch",
                "index": CFG.es_index,
                "es_host": CFG.es_host,
                "bm25_hits": [
                    {"id": k, "score": v["score"],
                     "page_no": v["src"].get("page_no"),
                     "lang": v["src"].get("lang"),
                     "snippet": v.get("snippet", "")[:400]}
                    for k, v in bm25_hits.items()
                ],
            })
        else:
            bm25_hits = self._bm25_search_mem(eq, CFG.top_k_bm25)
            es_raw_dump["bm25_hits"] = [
                {"id": k, "score": v["score"],
                 "page_no": v["src"].get("page_no"),
                 "lang": v["src"].get("lang"),
                 "snippet": v.get("snippet", "")[:400]}
                for k, v in bm25_hits.items()
            ]

        # Vector
        if self.use_es and self.es is not None:
            vec_hits = self.es.knn(qvec[0], CFG.top_k_vec)
        else:
            raw = self.faiss.search(qvec, CFG.top_k_vec)[0]
            vec_hits = {m["chunk_id"]: {"score": s, "src": m} for m, s in raw}

        all_ids = set(bm25_hits) | set(vec_hits)
        bm25 = {i: bm25_hits[i]["score"] for i in bm25_hits}
        sim = {i: vec_hits[i]["score"] for i in vec_hits}
        prox: Dict[str, float] = {}
        ent_match: Dict[str, int] = {}
        meta: Dict[str, Dict] = {}

        de_terms: set = set()
        en_terms: set = set()
        # Primary query terms go into the hinted language bucket (if hinted),
        # otherwise fall back to the classifier.
        for w in eq.terms:
            wl = w.lower()
            bucket = eq.lang if lang_hint in ("de", "en") else classify(wl)
            (de_terms if bucket == "de" else en_terms).add(wl)
        # Synonyms + translation represent the "equivalent" and use the classifier.
        equiv_pool = list(eq.synonyms) + [
            w.lower() for w in eq.translated.split() if w.isalpha()
        ]
        for w in equiv_pool:
            wl = w.lower()
            (de_terms if classify(wl) == "de" else en_terms).add(wl)

        prox_terms = list(dict.fromkeys(
            [t.lower() for t in eq.terms]
            + [w.lower() for w in eq.translated.split() if w.isalpha()]
        ))

        for cid in all_ids:
            src = (bm25_hits.get(cid) or vec_hits.get(cid))["src"]
            tokens = src.get("tokens") or re.findall(r"\w+", src["text"].lower())
            prox[cid] = proximity_score(tokens, prox_terms)
            ents = set(src.get("entities", []) or [])
            ent_match[cid] = int(any(e in ents for e in eq.entities))

            total, de_c, en_c = self._count_matches(
                tokens, src["lang"], de_terms, en_terms)
            snippet = (bm25_hits.get(cid, {}).get("snippet")
                       or _highlight(src["text"], prox_terms))
            meta[cid] = {
                "page_no": src["page_no"],
                "match_count": total,
                "de_count": de_c, "en_count": en_c,
                "snippet": snippet,
            }

        scores = chunk_scores(all_ids, bm25, sim, prox, ent_match)
        pages = aggregate_pages(meta, scores)

        # === LLM re-ranking (place #2) ==============================
        llm_rerank_used = False
        if self.llm is not None and self.llm.available() and pages:
            k = min(CFG.llm_rerank_k, len(pages))
            candidates = []
            for p in pages[:k]:
                snip = " ".join(p.get("snippets") or []) or \
                       (self.pages.get(p["page"], "")[:400])
                candidates.append({"id": f"p{p['page']}", "page": p["page"],
                                   "text": snip})
            llm_scores = self.llm.rerank(query, candidates)
            if llm_scores:
                llm_rerank_used = True
                # blend: hybrid_score * (1-w) + llm_score_norm * w * max_hybrid
                max_base = max((p["score"] for p in pages), default=1.0) or 1.0
                w = CFG.llm_rerank_weight
                for p in pages[:k]:
                    cid = f"p{p['page']}"
                    if cid in llm_scores:
                        p["llm_score"] = round(llm_scores[cid], 3)
                        p["score"] = round(
                            p["score"] * (1 - w) + llm_scores[cid] * max_base * w,
                            4,
                        )
                pages.sort(key=lambda r: r["score"], reverse=True)

        pages_out = pages[:top_pages]

        # ---- Section 2: top 5 pages with full page text (highlighted) ----
        top_full: List[Dict] = []
        primary_terms = de_terms if eq.lang == "de" else en_terms
        equiv_terms = en_terms if eq.lang == "de" else de_terms
        explain_k = min(CFG.llm_explain_k, len(pages_out))
        use_explain = self.llm is not None and self.llm.available()
        llm_explain_used = False
        for idx, p in enumerate(pages_out[:5]):
            raw = self.pages.get(p["page"], "")
            # === LLM explanation (place #3) =========================
            explanation = ""
            if use_explain and idx < explain_k:
                snippet_for_llm = " ".join(p.get("snippets") or []) or raw[:600]
                explanation = self.llm.explain(
                    query, p["page"], snippet_for_llm, lang=eq.lang)
                if explanation:
                    llm_explain_used = True
            top_full.append({
                "page": p["page"],
                "mentions": p["mentions"],
                "de": p["de"], "en": p["en"],
                "score": p["score"],
                "llm_score": p.get("llm_score"),
                "explanation": explanation,
                "primary_hits": _highlight_all(raw, primary_terms, "primary"),
                "equivalent_hits": _highlight_all(raw, equiv_terms, "equiv"),
            })

        # ---- Section 3: compact page counts ----
        page_counts = [
            {"page": p["page"], "mentions": p["mentions"],
             "de": p["de"], "en": p["en"]}
            for p in pages_out
        ]

        # ---- Section 2.5: vector hits dumped alongside bm25 ----
        es_raw_dump["vector_hits"] = [
            {"id": k, "score": v["score"],
             "page_no": v["src"].get("page_no"),
             "lang": v["src"].get("lang")}
            for k, v in vec_hits.items()
        ]
        elastic_archive = output_log.write_elastic(es_raw_dump)

        result = {
            "query": query,
            "lang_hint": lang_hint,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "expanded": eq.as_dict(),
            "pages": pages_out,
            "top_full": top_full,
            "page_counts": page_counts,
            "llm": {
                "enabled": self.use_llm,
                "available": bool(self.llm and self.llm.available()),
                "model": CFG.ollama_model if self.llm else None,
                "rerank_used": llm_rerank_used,
                "explain_used": llm_explain_used,
                "query_expansion_used": eq.llm_used,
            },
        }
        top_pages_summary = [
            {"page": p["page"], "mentions": p["mentions"],
             "de": p["de"], "en": p["en"], "score": p["score"]}
            for p in pages_out[:10]
        ]
        session_archive = output_log.append_session({
            "query": query, "lang_hint": lang_hint,
            "timestamp": result["timestamp"],
            "doc_id": self.doc_id,
            "top_pages": top_pages_summary,
            "llm": result["llm"],
        })
        duration_ms = int((time.time() - t_start) * 1000)
        db.log_search({
            "doc_id": self.doc_id,
            "query": query,
            "lang_hint": lang_hint,
            "use_llm": self.use_llm,
            "num_pages": len(pages_out),
            "timestamp": result["timestamp"],
            "duration_ms": duration_ms,
            "session_json": session_archive,
            "elastic_json": elastic_archive,
            "top_pages": top_pages_summary,
            "llm_flags": result["llm"],
        })
        result["duration_ms"] = duration_ms
        return result
