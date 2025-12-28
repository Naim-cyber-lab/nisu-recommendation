from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.repositories.events import create_event, get_event

from app.core.es import *
from app.embeddings.service import embed_text

router = APIRouter()


@router.post("/", response_model=any)
def create_event_endpoint(event_in):
    return create_event(event_in)


@router.get("/{event_id}", response_model=any)
def get_event_endpoint(event_id: str):
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event




INDEX = "events"

def search_events(q: str, size: int = 20):
    vector = embed_text(q)

    query = {
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"titre": {"query": q, "boost": 2}}},
                            {"match": {"bio": q}},
                        ]
                    }
                },
                "script": {
                    "source": """
                        double score = 0;
                        score += cosineSimilarity(params.q, 'titre_vector') * 2;
                        score += cosineSimilarity(params.q, 'bio_vector');
                        score += cosineSimilarity(params.q, 'preferences_vector');
                        return score;
                    """,
                    "params": {"q": vector},
                },
            }
        },
    }

    res = es.search(index=INDEX, body=query)

    return [
        {
            **hit["_source"],
            "score": hit["_score"],
            "id": hit["_id"],
        }
        for hit in res["hits"]["hits"]
    ]

@router.get("/search")
def search(q: str = Query(..., min_length=1)):
    return search_events(q)