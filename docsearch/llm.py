"""Thin client for a local Ollama server. Used in exactly three places:

  1. query expansion   (query.py)
  2. re-ranking top-K  (pipeline.py)
  3. explanation gen   (pipeline.py)

Falls back silently when Ollama isn't reachable so the rest of the app
keeps working."""
from __future__ import annotations

import json
import re
from typing import List, Dict, Optional

import urllib.request
import urllib.error

from .config import CFG

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class Ollama:
    def __init__(self, host: str = None, model: str = None, timeout: int = None):
        self.host = (host or CFG.ollama_host).rstrip("/")
        self.model = model or CFG.ollama_model
        self.timeout = timeout or CFG.ollama_timeout
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as r:
                self._available = r.status == 200
        except Exception:
            self._available = False
        return self._available

    # ------------------------------------------------------------------
    def generate(self, prompt: str, *, system: str = None,
                 json_mode: bool = False,
                 temperature: float = 0.2) -> str:
        if not self.available():
            return ""
        body: Dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            body["system"] = system
        if json_mode:
            body["format"] = "json"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
                text = payload.get("response", "")
                # deepseek-r1 emits <think>…</think>; strip it.
                text = _THINK_RE.sub("", text).strip()
                return text
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self._available = False
            return ""

    # ------------------------------------------------------------------
    def generate_json(self, prompt: str, *, system: str = None,
                      temperature: float = 0.1) -> Optional[dict]:
        raw = self.generate(prompt, system=system, json_mode=True,
                            temperature=temperature)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = _JSON_RE.search(raw)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

    # ================================================================ #
    # 1) QUERY EXPANSION                                                 #
    # ================================================================ #
    def expand_query(self, query: str, lang: str) -> Dict[str, List[str]]:
        """Ask the LLM for cross-lingual synonyms, paraphrases, and the
        translation of the query. Returns {"synonyms": [...],
        "paraphrases": [...], "translation": "..."}."""
        other = "English" if lang == "de" else "German"
        src = "German" if lang == "de" else "English"
        prompt = (
            f"You are a bilingual {src}/{other} lexical assistant.\n"
            f'Input query: "{query}"\n'
            f"Source language: {src}.\n\n"
            "Return a JSON object with exactly these keys:\n"
            "  translation  : the query translated into " + other + "\n"
            "  synonyms     : 4-8 single-word synonyms (mixed "
            f"{src}/{other}), lowercase\n"
            "  paraphrases  : 2-4 short paraphrases of the full query "
            f"(mixed {src}/{other})\n\n"
            "Return ONLY the JSON, no prose."
        )
        out = self.generate_json(prompt, temperature=0.2) or {}
        return {
            "synonyms":   [s.lower() for s in (out.get("synonyms") or []) if s],
            "paraphrases": [s for s in (out.get("paraphrases") or []) if s],
            "translation": (out.get("translation") or "").strip(),
        }

    # ================================================================ #
    # 2) RE-RANKING                                                      #
    # ================================================================ #
    def rerank(self, query: str, candidates: List[Dict]) -> Dict[str, float]:
        """Score each candidate 0.0–1.0 for how well it answers `query`.
        `candidates`: [{"id": str, "page": int, "text": str}, ...].
        Returns {id: score}. Missing ids default to 0.5."""
        if not candidates:
            return {}
        lines = []
        for c in candidates:
            snippet = (c.get("text") or "").replace("\n", " ")[:400]
            lines.append(f'{{"id": "{c["id"]}", "page": {c["page"]}, '
                         f'"snippet": {json.dumps(snippet)} }}')
        prompt = (
            "You are a bilingual (German/English) search re-ranker.\n"
            f'User query: "{query}"\n\n'
            "Below are candidate page snippets. Score each from 0.0 to 1.0 "
            "for how directly it answers the query (semantically, not just "
            "lexically). Consider cross-lingual equivalence.\n\n"
            "Candidates:\n[" + ",\n".join(lines) + "]\n\n"
            'Return ONLY a JSON object mapping id -> score, e.g. '
            '{"abc": 0.8, "def": 0.2}. No prose.'
        )
        out = self.generate_json(prompt, temperature=0.1)
        if not isinstance(out, dict):
            return {}
        scored: Dict[str, float] = {}
        for k, v in out.items():
            try:
                f = float(v)
                scored[str(k)] = max(0.0, min(1.0, f))
            except (TypeError, ValueError):
                continue
        return scored

    # ================================================================ #
    # 3) EXPLANATION                                                     #
    # ================================================================ #
    def explain(self, query: str, page_no: int, snippet: str,
                lang: str = "auto") -> str:
        """One-sentence, user-facing explanation of why this page matches."""
        if not snippet:
            return ""
        reply_lang = {"de": "German", "en": "English"}.get(lang, "English")
        prompt = (
            f'User query: "{query}"\n'
            f"Page number: {page_no}\n"
            f"Page snippet:\n\"\"\"\n{snippet[:800]}\n\"\"\"\n\n"
            f"In ONE short sentence (max 25 words) in {reply_lang}, explain "
            "why this page is relevant to the query. Mention the specific "
            "German or English term(s) that support the match. No preamble."
        )
        text = self.generate(prompt, temperature=0.3)
        # keep it to one line
        return text.splitlines()[0].strip() if text else ""
