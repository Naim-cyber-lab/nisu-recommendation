from fastapi import APIRouter
from .endpoints import indexing, recommendations, embedding, events

api_router = APIRouter()
api_router.include_router(indexing.router, prefix="/index", tags=["indexing"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(embedding.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
