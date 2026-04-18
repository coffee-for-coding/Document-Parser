import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict

from .config import CFG

_NLP: Dict[str, object] = {}


def _load_spacy(lang: str):
    if lang in _NLP:
        return _NLP[lang]
    import spacy
    model = CFG.spacy_de if lang == "de" else CFG.spacy_en
    try:
        nlp = spacy.load(model, disable=["parser"])
    except OSError:
        # Fall back to a blank pipeline with a sentencizer so the app still runs
        # even when spaCy models aren't installed.
        nlp = spacy.blank(lang)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    _NLP[lang] = nlp
    return nlp


def detect_lang(text: str) -> str:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        code = detect(text[:2000])
        return "de" if code.startswith("de") else "en"
    except Exception:
        # Heuristic fallback: German-specific characters / stopwords.
        de_markers = re.findall(r"[äöüÄÖÜß]|\b(der|die|das|und|nicht|ist|mit|für)\b",
                                text.lower())
        return "de" if len(de_markers) > 3 else "en"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int
    lang: str
    text: str
    lemma: str
    tokens: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)


def _window_chunks(text: str, size: int, overlap: int) -> List[str]:
    toks = text.split()
    if not toks:
        return []
    if len(toks) <= size:
        return [text]
    out, step = [], max(1, size - overlap)
    for i in range(0, len(toks), step):
        window = toks[i:i + size]
        if not window:
            break
        out.append(" ".join(window))
        if i + size >= len(toks):
            break
    return out


def chunk_page(doc_id: str, page_no: int, text: str,
               size: int = CFG.chunk_tokens,
               overlap: int = CFG.chunk_overlap) -> List[Chunk]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    lang = detect_lang(text)
    nlp = _load_spacy(lang)
    pieces = _window_chunks(text, size, overlap)
    chunks: List[Chunk] = []
    for piece in pieces:
        doc = nlp(piece)
        tokens, lemmas, ents = [], [], []
        for t in doc:
            if getattr(t, "is_space", False):
                continue
            tokens.append(t.text.lower())
            lemma = getattr(t, "lemma_", "") or t.text
            lemmas.append(lemma.lower())
        if getattr(doc, "ents", None):
            ents = list({e.text for e in doc.ents
                         if e.label_ in {"PER", "PERSON", "GPE", "LOC", "ORG"}})
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id, page_no=page_no, lang=lang,
            text=piece, lemma=" ".join(lemmas),
            tokens=tokens, entities=ents,
        ))
    return chunks
