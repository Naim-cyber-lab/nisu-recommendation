from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas import RecommendedWinker, RecommendedEvent
from app.services.recommendations import (
    recommend_winkers_for_winker,
    recommend_events_for_winker,
)

router = APIRouter()


@router.get("/winkers/{winker_id}", response_model=List[RecommendedWinker])
def recommend_winkers_endpoint(winker_id: int, size: int = 10):
    recos = recommend_winkers_for_winker(winker_id, size=size)
    if not recos:
        # à toi de voir si tu veux 200 ou 404
        return []
    return recos


@router.get("/events/{winker_id}", response_model=List[RecommendedEvent])
def recommend_events_endpoint(winker_id: int, size: int = 10):
    recos = recommend_events_for_winker(winker_id, size=size)
    if not recos:
        return []
    return recos
