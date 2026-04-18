from typing import List, Dict, Any
import numpy as np

from .config import CFG

MAPPING: Dict[str, Any] = {
    "settings": {
        "analysis": {
            "filter": {
                "de_stop": {"type": "stop", "stopwords": "_german_"},
                "en_stop": {"type": "stop", "stopwords": "_english_"},
                "de_stem": {"type": "stemmer", "language": "light_german"},
                "en_stem": {"type": "stemmer", "language": "light_english"},
                "de_en_syn": {
                    "type": "synonym_graph",
                    "synonyms_path": "analysis/de_en_synonyms.txt",
                    "updateable": True,
                },
            },
            "analyzer": {
                "de_analyzer": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "de_stop", "de_stem"],
                },
                "en_analyzer": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "en_stop", "en_stem"],
                },
                "de_en_search": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "de_en_syn"],
                },
            },
        }
    },
    "mappings": {
        "properties": {
            "doc_id":   {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "page_no":  {"type": "integer"},
            "lang":     {"type": "keyword"},
            "text":     {"type": "text", "term_vector": "with_positions_offsets"},
            "text_de":  {"type": "text", "analyzer": "de_analyzer",
                         "search_analyzer": "de_en_search",
                         "term_vector": "with_positions_offsets"},
            "text_en":  {"type": "text", "analyzer": "en_analyzer",
                         "search_analyzer": "de_en_search",
                         "term_vector": "with_positions_offsets"},
            "lemma":    {"type": "text", "analyzer": "whitespace"},
            "entities": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 384,
                          "index": True, "similarity": "cosine"},
        }
    },
}


class ESIndexer:
    def __init__(self, dim: int = 384):
        from elasticsearch import Elasticsearch
        self.es = Elasticsearch(CFG.es_host)
        self.dim = dim
        MAPPING["mappings"]["properties"]["embedding"]["dims"] = dim

    def ensure_index(self, recreate: bool = False):
        if recreate and self.es.indices.exists(index=CFG.es_index):
            self.es.indices.delete(index=CFG.es_index)
        if not self.es.indices.exists(index=CFG.es_index):
            self.es.indices.create(index=CFG.es_index, body=MAPPING)

    def bulk_index(self, chunks: List, vectors: np.ndarray):
        from elasticsearch import helpers

        def gen():
            for ch, vec in zip(chunks, vectors):
                yield {
                    "_index": CFG.es_index,
                    "_id":    ch.chunk_id,
                    "_source": {
                        "doc_id": ch.doc_id, "chunk_id": ch.chunk_id,
                        "page_no": ch.page_no, "lang": ch.lang,
                        "text": ch.text,
                        "text_de": ch.text if ch.lang == "de" else "",
                        "text_en": ch.text if ch.lang == "en" else "",
                        "lemma": ch.lemma,
                        "entities": ch.entities,
                        "embedding": vec.tolist(),
                    },
                }

        helpers.bulk(self.es, gen(), chunk_size=500, request_timeout=120)
        self.es.indices.refresh(index=CFG.es_index)

    def search(self, eq, k: int) -> Dict[str, Dict[str, Any]]:
        should = [
            {"multi_match": {"query": eq.raw,
                             "fields": ["text_de", "text_en", "lemma^1.2"],
                             "type": "best_fields"}},
        ]
        if eq.translated and eq.translated.lower() != eq.raw.lower():
            should.append({"multi_match": {"query": eq.translated,
                                           "fields": ["text_de", "text_en", "lemma^1.2"]}})
        for s in eq.synonyms:
            should.append({"multi_match": {"query": s,
                                           "fields": ["text_de", "text_en", "lemma"]}})
        for ent in eq.entities:
            should.append({"term": {"entities": {"value": ent, "boost": 2.0}}})
        if len(eq.terms) > 1:
            should.append({"match_phrase": {"text_en":
                           {"query": eq.raw, "slop": 5, "boost": 1.5}}})
            should.append({"match_phrase": {"text_de":
                           {"query": eq.raw, "slop": 5, "boost": 1.5}}})

        body = {
            "size": k,
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
            "highlight": {"fields": {"text_de": {}, "text_en": {}, "text": {}},
                          "fragment_size": 160, "number_of_fragments": 1},
        }
        res = self.es.search(index=CFG.es_index, body=body)
        out: Dict[str, Dict[str, Any]] = {}
        for h in res["hits"]["hits"]:
            hl = h.get("highlight", {}) or {}
            snippet = (hl.get("text_de") or hl.get("text_en") or hl.get("text") or [""])[0]
            out[h["_id"]] = {
                "score": h["_score"],
                "src": h["_source"],
                "snippet": snippet,
            }
        return out

    def knn(self, qvec: np.ndarray, k: int) -> Dict[str, Dict[str, Any]]:
        body = {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": qvec.tolist(),
                "k": k,
                "num_candidates": max(k * 5, 200),
            },
        }
        res = self.es.search(index=CFG.es_index, body=body)
        return {h["_id"]: {"score": h["_score"], "src": h["_source"]}
                for h in res["hits"]["hits"]}
