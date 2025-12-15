from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
import requests
from elasticsearch import Elasticsearch
import os
from app.core.db import *
from app.mappings import *
from app.schemas import *
from app.api.v1.sql.fetch_events_with_relations_by_ids import *

router = APIRouter()

# ---- Config ----
ES_INDEX = "nisu_events"
ES_HOST = os.getenv("ES_HOST", "http://192.168.1.213:9200")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")

EMBEDDINGS_URL = "https://recommendation.nisu.fr/api/v1/recommendations/embeddings"
EMBEDDINGS_TIMEOUT = 60

es = Elasticsearch(
    [ES_HOST],
    basic_auth=(ES_USER, ES_PASS) if ES_USER and ES_PASS else None,
)

# ---- Helpers ----

def get_embedding(text: str) -> List[float]:
    if not text or not text.strip():
        return []
    r = requests.post(
        EMBEDDINGS_URL,
        json={"text": text},
        timeout=EMBEDDINGS_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding")
    if not isinstance(vec, list) or len(vec) == 0:
        return []
    return vec


def build_winker_profile_text(w: Dict[str, Any]) -> str:
    """
    Construit un texte stable qui représente le profil du winker.
    Tu peux l’ajuster selon tes champs vraiment utiles.
    """
    parts: List[str] = []

    bio = (w.get("bio") or "").strip()
    if bio:
        parts.append(bio)

    # localisation
    city = (w.get("city") or "").strip()
    region = (w.get("region") or "").strip()
    subregion = (w.get("subregion") or "").strip()
    if city or region or subregion:
        parts.append(" ".join([p for p in [city, subregion, region] if p]))

    # dernière recherche event (souvent super signal)
    last_search = (w.get("derniereRechercheEvent") or "").strip()
    if last_search and last_search not in ("{}", "[]", "null"):
        parts.append(last_search)

    # si tu stockes des tags/prefer dans d'autres champs, ajoute-les ici

    return " | ".join(parts).strip()


def parse_geo(w: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lat = w.get("lat")
    lon = w.get("lon")
    try:
        if lat is None or lon is None:
            return None
        return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        return None


# ---- Ta route existante (inchangée) ----
@router.get("/get_rencontre_from_winker/{user_id}")
def get_profil_winker_raw(user_id: int):

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM profil_winker WHERE id = %s", (user_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Profil introuvable")
            columns = [desc[0] for desc in cur.description]

    return dict(zip(columns, row))


# ---- Nouvelle route : top 4 events ----

@router.get("/get_events_for_winker/{user_id}", response_model=List[EventOut])
def get_events_for_winker(user_id: int) -> List[EventOut]:
    winker = get_profil_winker_raw(user_id)

    profile_text = build_winker_profile_text(winker)
    if not profile_text:
        raise HTTPException(status_code=400, detail="Profil trop vide pour recommander des events.")

    qvec = get_embedding(profile_text)
    if not qvec:
        raise HTTPException(status_code=500, detail="Impossible de générer l'embedding du profil.")

    user_geo = parse_geo(winker)

    knn_query: Dict[str, Any] = {
        "field": "embedding_vector",
        "query_vector": qvec,
        "k": 50,
        "num_candidates": 200
    }

    base_query: Dict[str, Any] = {"match_all": {}}
    if user_geo:
        base_query = {
            "bool": {
                "filter": [
                    {"geo_distance": {"distance": "200km", "localisation": user_geo}}
                ]
            }
        }

    body: Dict[str, Any] = {
        "size": 4,
        "query": base_query,
        "knn": knn_query,
        "rescore": {
            "window_size": 50,
            "query": {
                "rescore_query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "functions": [
                            {"field_value_factor": {"field": "boost", "factor": 0.05, "missing": 0}}
                        ],
                        "boost_mode": "sum",
                        "score_mode": "sum"
                    }
                },
                "query_weight": 1.0,
                "rescore_query_weight": 1.0
            }
        },
        "_source": False,  # on veut juste les ids
    }

    resp = es.search(index=ES_INDEX, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    event_ids: List[int] = []
    for h in hits:
        try:
            event_ids.append(int(h.get("_id")))
        except Exception:
            continue

    if not event_ids:
        return []

    # ====== ICI: tu remplaces par TON accès DB (SQLAlchemy / raw SQL / repo) ======
    # Objectif: récupérer Event + creatorWinker + participants + participants.participeWinker
    events_orm = fetch_events_with_relations_by_ids(event_ids)
    # ============================================================================

    # préserver l'ordre ES
    by_id = {e.get("id"): e for e in events_orm}
    ordered = [by_id[eid] for eid in event_ids if eid in by_id]

    return [to_event_out(e) for e in ordered]

