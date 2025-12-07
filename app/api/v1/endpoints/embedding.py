from typing import List
from pydantic import BaseModel

from fastapi import APIRouter

# app/embedding.py

from typing import List
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None

router = APIRouter()

def _get_model() -> SentenceTransformer:
    """
    Charge le modèle globalement (lazy loading).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def get_embedding(text: str) -> List[float]:
    """
    Prend un texte en entrée et renvoie son embedding
    sous forme de liste de floats.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    model = _get_model()
    vector = model.encode(text)
    return vector.tolist()



class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
def create_embedding(payload: EmbeddingRequest):
    """
    Génère l'embedding du texte fourni.
    Visible et testable dans Swagger (/docs).
    """
    vec = get_embedding(payload.text)
    return EmbeddingResponse(embedding=vec, dimension=len(vec))