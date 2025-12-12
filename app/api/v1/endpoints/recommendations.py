# app/api/conversation_activity.py
from fastapi import APIRouter, HTTPException
from typing import List, Any
from app.core.db import *
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


from fastapi import HTTPException

@router.get("/get_rencontre_from_winker/{user_id}")
def get_profil_winker_raw(user_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM profil_winker WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail="Profil introuvable")

            # noms des colonnes dans l’ordre du SELECT
            columns = [desc[0] for desc in cur.description]

    return dict(zip(columns, row))
