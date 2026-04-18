from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    es_host: str = os.getenv("ES_HOST", "http://localhost:9200")
    es_index: str = os.getenv("ES_INDEX", "doc_chunks")
    use_es: bool = os.getenv("USE_ES", "1") == "1"

    faiss_path: str = os.getenv("FAISS_PATH", "./data/faiss.index")
    meta_path: str = os.getenv("FAISS_META", "./data/faiss_meta.pkl")
    chunks_path: str = os.getenv("CHUNKS_PATH", "./data/chunks.pkl")
    pages_path: str = os.getenv("PAGES_PATH", "./data/pages.pkl")

    embed_model: str = os.getenv(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    mt_de_en: str = "Helsinki-NLP/opus-mt-de-en"
    mt_en_de: str = "Helsinki-NLP/opus-mt-en-de"
    use_mt: bool = os.getenv("USE_MT", "0") == "1"

    spacy_de: str = os.getenv("SPACY_DE", "de_core_news_sm")
    spacy_en: str = os.getenv("SPACY_EN", "en_core_web_sm")

    chunk_tokens: int = 350
    chunk_overlap: int = 50

    top_k_bm25: int = 200
    top_k_vec: int = 200

    alpha: float = 0.35
    beta: float = 0.40
    gamma: float = 0.20
    delta: float = 0.05

    prox_window: int = 30

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))
    use_llm: bool = os.getenv("USE_LLM", "1") == "1"
    llm_rerank_k: int = int(os.getenv("LLM_RERANK_K", "10"))
    llm_explain_k: int = int(os.getenv("LLM_EXPLAIN_K", "5"))
    llm_rerank_weight: float = float(os.getenv("LLM_RERANK_WEIGHT", "0.4"))


CFG = Config()
