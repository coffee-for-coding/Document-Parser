from typing import List
import numpy as np

from .config import CFG


class Embedder:
    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or CFG.embed_model)
        self.dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        v = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return v.astype("float32")
