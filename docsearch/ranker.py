from collections import defaultdict, Counter
from math import log
from typing import Dict, List, Iterable

from .config import CFG


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def proximity_score(tokens: List[str], terms: List[str],
                    window: int = CFG.prox_window) -> float:
    """Smallest enclosing window over token positions of each unique term.
    Returns m / (1 + span) where m = len(unique terms) if all occur inside
    a window of ≤ `window` tokens; 0 otherwise.
    """
    uniq = [t for t in dict.fromkeys(terms) if t]
    if len(uniq) < 2:
        return 0.0
    positions = {t: [i for i, w in enumerate(tokens) if w == t] for t in uniq}
    if any(not positions[t] for t in uniq):
        return 0.0
    # flatten to (pos, term) sorted
    flat = sorted((p, t) for t, ps in positions.items() for p in ps)
    need = Counter(uniq)
    have: Counter = Counter()
    matched_kinds = 0
    best = 0.0
    l = 0
    for r in range(len(flat)):
        pr, tr = flat[r]
        if have[tr] == 0:
            matched_kinds += 1
        have[tr] += 1
        while matched_kinds == len(need):
            pl, tl = flat[l]
            span = pr - pl
            if span <= window:
                best = max(best, len(uniq) / (1.0 + span))
            have[tl] -= 1
            if have[tl] == 0:
                matched_kinds -= 1
            l += 1
    return best


def rrf_merge(rank_lists: Iterable[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for lst in rank_lists:
        for r, cid in enumerate(lst, start=1):
            scores[cid] += 1.0 / (k + r)
    return dict(scores)


def chunk_scores(candidates: Iterable[str],
                 bm25: Dict[str, float],
                 sim: Dict[str, float],
                 prox: Dict[str, float],
                 ent_match: Dict[str, int]) -> Dict[str, float]:
    ids = list(candidates)
    b = _minmax([bm25.get(i, 0.0) for i in ids])
    s = _minmax([sim.get(i, 0.0) for i in ids])
    p = _minmax([prox.get(i, 0.0) for i in ids])
    e = [ent_match.get(i, 0) for i in ids]

    # If a component is all-zero, redistribute its weight uniformly.
    weights = {"a": CFG.alpha, "b": CFG.beta, "g": CFG.gamma, "d": CFG.delta}
    if all(v == 0 for v in p):
        weights["a"] += weights["g"] / 3
        weights["b"] += weights["g"] / 3
        weights["d"] += weights["g"] / 3
        weights["g"] = 0.0

    return {cid: (weights["a"] * b[k] + weights["b"] * s[k]
                  + weights["g"] * p[k] + weights["d"] * e[k])
            for k, cid in enumerate(ids)}


def aggregate_pages(chunk_meta: Dict[str, Dict],
                    chunk_score: Dict[str, float],
                    lam: float = 0.3) -> List[Dict]:
    by_page = defaultdict(lambda: {"mentions": 0, "de": 0, "en": 0,
                                   "max": 0.0, "sum": 0.0, "n": 0,
                                   "snippets": []})
    for cid, meta in chunk_meta.items():
        pg = by_page[meta["page_no"]]
        pg["mentions"] += meta.get("match_count", 0)
        pg["de"] += meta.get("de_count", 0)
        pg["en"] += meta.get("en_count", 0)
        sc = chunk_score.get(cid, 0.0)
        pg["max"] = max(pg["max"], sc)
        pg["sum"] += sc
        pg["n"] += 1
        if meta.get("snippet"):
            pg["snippets"].append(meta["snippet"])

    out = []
    for page, v in by_page.items():
        score = log(1 + v["mentions"]) * v["max"] + lam * (v["sum"] / max(v["n"], 1))
        out.append({
            "page": page,
            "mentions": v["mentions"],
            "de": v["de"], "en": v["en"],
            "score": round(score, 4),
            "snippets": v["snippets"][:3],
        })
    return sorted(out, key=lambda r: r["score"], reverse=True)
