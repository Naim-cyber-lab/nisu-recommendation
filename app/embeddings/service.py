# app/embeddings/service.py
from __future__ import annotations

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import os
import threading
import torch

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# ⚠️ IMPORTANT:
# - Ne PAS mettre ce device dans le constructeur SentenceTransformer (ça peut crasher avec meta tensors).
# - On l'applique dans encode().
_DEVICE_ENV = os.getenv("EMBEDDING_DEVICE", "").strip().lower()  # "cpu" / "cuda" / "" (auto)

_model: Optional[SentenceTransformer] = None
_lock = threading.Lock()


def get_model_name() -> str:
    return _MODEL_NAME


def _resolve_device() -> str:
    """
    Device choisi:
    - si EMBEDDING_DEVICE=cpu -> cpu
    - si EMBEDDING_DEVICE=cuda -> cuda (si dispo, sinon cpu)
    - sinon auto -> cuda si dispo sinon cpu
    """
    if _DEVICE_ENV == "cpu":
        return "cpu"
    if _DEVICE_ENV == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        # ✅ NE PAS passer device= ici (évite le self.to(device) qui peut crasher avec meta)
        _model = SentenceTransformer(_MODEL_NAME)
        return _model


def embed_text(text: str, normalize: bool = True) -> List[float]:
    text = (text or "").strip()
    if not text:
        return []

    model = _get_model()
    device = _resolve_device()

    # ✅ device appliqué ici (pas au constructeur)
    emb = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        device=device,
        show_progress_bar=False,
    )

    # JSON-safe
    return [float(x) for x in emb]
