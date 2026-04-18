import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from .config import CFG


class FaissIndex:
    """IndexFlatIP over L2-normalized vectors == cosine similarity.
    For larger corpora (>50k vectors) switch to IVFPQ."""

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict] = []

    def add(self, vectors: np.ndarray, meta: List[Dict]):
        if vectors.size == 0:
            return
        self.index.add(vectors.astype("float32"))
        self.meta.extend(meta)

    def search(self, q: np.ndarray, k: int) -> List[List[Tuple[Dict, float]]]:
        if self.index.ntotal == 0:
            return [[]]
        k = min(k, self.index.ntotal)
        D, I = self.index.search(q.astype("float32"), k)
        results = []
        for r in range(q.shape[0]):
            row = []
            for c, i in enumerate(I[r]):
                if i == -1 or i >= len(self.meta):
                    continue
                row.append((self.meta[i], float(D[r][c])))
            results.append(row)
        return results

    def save(self, path: str = None, meta_path: str = None):
        import faiss
        path = path or CFG.faiss_path
        meta_path = meta_path or CFG.meta_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    @classmethod
    def load(cls, dim: int, path: str = None, meta_path: str = None) -> "FaissIndex":
        import faiss
        path = path or CFG.faiss_path
        meta_path = meta_path or CFG.meta_path
        obj = cls(dim)
        obj.index = faiss.read_index(path)
        with open(meta_path, "rb") as f:
            obj.meta = pickle.load(f)
        return obj
