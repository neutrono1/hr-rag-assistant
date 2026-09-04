"""
Embeddings run locally via sentence-transformers so ingestion and
retrieval never cost API quota. The model weights (~80MB) download once
on first use and are cached by the library.
"""
import os

# sentence-transformers pulls in `transformers`, which by default tries
# to import an optional TensorFlow integration. If the environment also
# has TensorFlow + Keras 3 installed (common in Anaconda base envs from
# unrelated projects), that import fails with a Keras-3-incompatibility
# error even though we never use TensorFlow -- we only need the PyTorch
# backend. Setting USE_TF=0 before the first transformers import skips
# that code path entirely.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

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
