# app/api/conversation_activity.py
from fastapi import APIRouter, HTTPException
from typing import List, Any

from app.core.es import es_client

router = APIRouter()

INDEX_NAME = "conversation_activity"


@router.get(
    "/conversation-activity",
    response_model=List[dict]  # ou List[Any] / un Pydantic model si tu veux typer propre
)
def get_all_conversation_activity(size: int = 1000):
    """
    Retourne tous les documents (jusqu'à `size`) de l'index conversation_activity.
    """
    try:
        # requête très simple : tout l'index
        resp = es_client.search(
            index=INDEX_NAME,
            query={"match_all": {}},
            size=size,
        )

        hits = resp["hits"]["hits"]

        # On renvoie juste le _source + l'_id
        return [
            {
                "id": hit["_id"],
                **hit["_source"],
            }
            for hit in hits
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
