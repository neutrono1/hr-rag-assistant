"""
Embeddings via fastembed (pure ONNX Runtime, no PyTorch dependency).
Chosen specifically to keep the runtime memory footprint small enough
for constrained hosting (Render free/starter tier) -- see DESIGN.md.
"""
from functools import lru_cache
from typing import List

from app.config import EMBEDDING_MODEL

# fastembed's model naming differs from the sentence-transformers hub
# path. all-MiniLM-L6-v2 maps directly; if EMBEDDING_MODEL changes,
# confirm the equivalent name in fastembed's supported model list.
_FASTEMBED_MODEL_MAP = {
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


@lru_cache(maxsize=1)
def _get_model():
    from fastembed import TextEmbedding
    model_name = _FASTEMBED_MODEL_MAP.get(EMBEDDING_MODEL, EMBEDDING_MODEL)
    return TextEmbedding(model_name=model_name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = list(model.embed(texts))
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> List[float]:
    return embed_texts([text])[0]