from dataclasses import dataclass, field
from typing import List
import re

from .config import CFG
from .preprocess import detect_lang, _load_spacy
from .synonyms import expand_term
from .llm import Ollama


@dataclass
class ExpandedQuery:
    raw: str
    lang: str
    translated: str = ""
    synonyms: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    paraphrases: List[str] = field(default_factory=list)
    llm_used: bool = False

    def as_dict(self):
        return {
            "raw": self.raw, "lang": self.lang,
            "translated": self.translated,
            "synonyms": self.synonyms,
            "entities": self.entities,
            "terms": self.terms,
            "paraphrases": self.paraphrases,
            "llm_used": self.llm_used,
        }


class QueryExpander:
    def __init__(self, use_llm: bool = None):
        self._mt = None
        if CFG.use_mt:
            self._init_mt()
        self.use_llm = CFG.use_llm if use_llm is None else use_llm
        self._llm = Ollama() if self.use_llm else None

    def _init_mt(self):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            self._tok_de_en = MarianTokenizer.from_pretrained(CFG.mt_de_en)
            self._mt_de_en = MarianMTModel.from_pretrained(CFG.mt_de_en)
            self._tok_en_de = MarianTokenizer.from_pretrained(CFG.mt_en_de)
            self._mt_en_de = MarianMTModel.from_pretrained(CFG.mt_en_de)
            self._mt = True
        except Exception:
            self._mt = None

    def _translate(self, text: str, src: str) -> str:
        if not self._mt:
            # Token-level dictionary fallback
            toks = re.findall(r"\w+|\S", text)
            out = []
            for t in toks:
                low = t.lower()
                syns = expand_term(low)
                if syns:
                    out.append(syns[0])
                else:
                    out.append(t)
            return " ".join(out)
        tok, mdl = (self._tok_de_en, self._mt_de_en) if src == "de" \
                   else (self._tok_en_de, self._mt_en_de)
        ids = tok([text], return_tensors="pt", padding=True)
        out = mdl.generate(**ids, max_new_tokens=64)
        return tok.decode(out[0], skip_special_tokens=True)

    def expand(self, q: str) -> ExpandedQuery:
        q = (q or "").strip()
        if not q:
            return ExpandedQuery(raw="", lang="en")
        lang = detect_lang(q)
        nlp = _load_spacy(lang)
        doc = nlp(q)
        terms, ents = [], []
        for t in doc:
            if getattr(t, "is_space", False):
                continue
            if not t.text.isalpha():
                continue
            if getattr(t, "is_stop", False):
                continue
            lemma = (getattr(t, "lemma_", "") or t.text).lower()
            terms.append(lemma)
        if getattr(doc, "ents", None):
            ents = [e.text for e in doc.ents
                    if e.label_ in {"PER", "PERSON", "GPE", "LOC", "ORG"}]
        if not terms:
            terms = [w.lower() for w in re.findall(r"\w+", q)]

        syns: List[str] = []
        for t in terms:
            syns.extend(expand_term(t))
        syns = list(dict.fromkeys(syns))  # dedupe, preserve order

        translated = self._translate(q, lang)
        paraphrases: List[str] = []
        llm_used = False

        # === LLM-driven expansion (place #1) ========================
        if self._llm is not None and self._llm.available():
            llm_out = self._llm.expand_query(q, lang)
            if llm_out:
                llm_used = True
                for s in llm_out.get("synonyms", []):
                    if s and s not in syns:
                        syns.append(s)
                paraphrases = llm_out.get("paraphrases", []) or []
                tr = llm_out.get("translation", "").strip()
                if tr and (not translated or translated.lower() == q.lower()):
                    translated = tr

        return ExpandedQuery(
            raw=q, lang=lang, translated=translated,
            synonyms=syns, entities=ents, terms=terms,
            paraphrases=paraphrases, llm_used=llm_used,
        )
