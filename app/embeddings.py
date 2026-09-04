"""
Embeddings run locally via sentence-transformers so ingestion and
retrieval never cost API quota. The model weights (~80MB) download once
on first use and are cached by the library.
"""
from functools import lru_cache
from typing import List

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> List[float]:
    return embed_texts([text])[0]
