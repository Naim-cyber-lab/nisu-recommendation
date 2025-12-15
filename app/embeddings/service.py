# app/embeddings/service.py
from typing import List
from sentence_transformers import SentenceTransformer
import math

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0 else [x / norm for x in vec]

def embed_text(text: str, normalize: bool = True) -> List[float]:
    if not text or not text.strip():
        return []
    model = _get_model()
    vec = model.encode(text).tolist()
    return _l2_normalize(vec) if normalize else vec
