from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.core.es import *
from app.embeddings.service import embed_text

router = APIRouter()






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