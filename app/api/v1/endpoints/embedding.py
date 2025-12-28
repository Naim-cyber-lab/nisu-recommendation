from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Any, Dict, List, Tuple
from elasticsearch import ApiError

from app.core.es import es_client
from app.embeddings.service import embed_text

# 👉 tu l'as déjà
from app.core.db import get_conn  # adapte si besoin
from app.api.v1.sql.fetch_events_with_relations_by_ids import *

router = APIRouter()

INDEX = "nisu_events"
VECTOR_DIMS = 768


def _to_float_list(vec) -> List[float]:
    """Convert embedding to JSON-safe list[float]."""
    if vec is None:
        return []
    if hasattr(vec, "tolist"):  # numpy / torch
        vec = vec.tolist()
    try:
        return [float(x) for x in vec]
    except Exception:
        return []


def _relevance_label(score: float) -> str:
    """
    Labels simples basés sur _score.
    Ajuste ces seuils après observation de tes scores réels.
    """
    if score >= 2.5:
        return "TRÈS_PERTINENT"
    if score >= 1.2:
        return "PERTINENT"
    if score >= 0.5:
        return "MOYEN"
    return "FAIBLE"


def _safe_int_id(es_id: Any) -> Optional[int]:
    """
    ES _id peut être str. On tente de le convertir en int.
    Si ce n'est pas possible, on ignore (ou adapte à ton modèle).
    """
    try:
        return int(es_id)
    except Exception:
        return None


def _build_es_query(
    q: str,
    from_: int,
    size: int,
    lat: Optional[float],
    lon: Optional[float],
    sigma_km: float,
    geo_weight: float,
    vec_weight: float,
) -> Dict[str, Any]:
    q = (q or "").strip()
    is_query_empty = len(q) == 0

    # Base query: on veut TOUJOURS des réponses => minimum_should_match = 0
    if is_query_empty:
        base_query: Dict[str, Any] = {"match_all": {}}
    else:
        base_query = {
            "bool": {
                "should": [
                    {"match": {"titre": {"query": q, "boost": 2}}},
                    {"match": {"bio": {"query": q}}},
                ],
                "minimum_should_match": 0,
            }
        }

    vector = _to_float_list(embed_text(q)) if not is_query_empty else []
    use_vectors = len(vector) == VECTOR_DIMS

    functions: List[Dict[str, Any]] = []

    # Vectors score (guard missing fields)
    vec_script = """
      double s = 0.0;
      if (params.useVec == false) return 0.0;

      if (doc['titre_vector'].size() != 0) {
        s += cosineSimilarity(params.q, 'titre_vector') * 2.0;
      }
      if (doc['bio_vector'].size() != 0) {
        s += cosineSimilarity(params.q, 'bio_vector');
      }
      if (doc['preferences_vector'].size() != 0) {
        s += cosineSimilarity(params.q, 'preferences_vector');
      }

      return s;
    """
    functions.append(
        {
            "script_score": {
                "script": {
                    "source": vec_script,
                    "params": {"useVec": use_vectors, "q": vector if use_vectors else []},
                }
            },
            "weight": vec_weight,
        }
    )

    # Geo gaussian boost
    has_geo = lat is not None and lon is not None
    if has_geo:
        geo_script = """
          if (doc['localisation'].size() == 0) return 0.0;

          double d = doc['localisation'].arcDistance(params.lat, params.lon); // meters
          double sigma = params.sigma_m; // meters
          if (sigma <= 0) return 0.0;

          double x = d / sigma;
          return Math.exp(-0.5 * x * x);
        """
        functions.append(
            {
                "script_score": {
                    "script": {
                        "source": geo_script,
                        "params": {
                            "lat": float(lat),
                            "lon": float(lon),
                            "sigma_m": float(sigma_km) * 1000.0,
                        },
                    }
                },
                "weight": geo_weight,
            }
        )

    # Optional doc boost field
    functions.append(
        {
            "field_value_factor": {
                "field": "boost",
                "missing": 0,
                "modifier": "sqrt",
            },
            "weight": 0.2,
        }
    )

    # Return distance_km as a computed field (if geo provided)
    script_fields: Dict[str, Any] = {}
    if has_geo:
        script_fields["distance_km"] = {
            "script": {
                "source": """
                  if (doc['localisation'].size() == 0) return null;
                  return doc['localisation'].arcDistance(params.lat, params.lon) / 1000.0;
                """,
                "params": {"lat": float(lat), "lon": float(lon)},
            }
        }

    body: Dict[str, Any] = {
        "track_total_hits": True,
        "from": from_,
        "size": size,
        "_source": False,  # on ne veut pas les sources ES, on récupère en DB
        "query": {
            "function_score": {
                "query": base_query,
                "functions": functions,
                "score_mode": "sum",
                # on veut garder un peu de score texte ES (quand il matche), sans bloquer
                "boost_mode": "sum",
            }
        },
    }

    if script_fields:
        body["script_fields"] = script_fields

    return body


def search_events_and_hydrate(
    q: str,
    page: int,
    per_page: int,
    lat: Optional[float],
    lon: Optional[float],
    sigma_km: float,
    geo_weight: float,
    vec_weight: float,
) -> Dict[str, Any]:
    if page < 1:
        page = 1
    per_page = max(1, min(per_page, 100))
    from_ = (page - 1) * per_page

    body = _build_es_query(
        q=q,
        from_=from_,
        size=per_page,
        lat=lat,
        lon=lon,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
    )

    try:
        res = es_client.search(index=INDEX, body=body)
    except ApiError as e:
        detail = getattr(e, "info", None) or str(e)
        raise HTTPException(status_code=400, detail={"elasticsearch_error": detail})

    hits = res.get("hits", {}).get("hits", [])
    total = res.get("hits", {}).get("total", {})
    total_value = int(total.get("value", 0)) if isinstance(total, dict) else int(total or 0)

    # 1) Build meta by id preserving ES order
    es_order_ids: List[int] = []
    meta_by_id: Dict[int, Dict[str, Any]] = {}

    for h in hits:
        es_id_raw = h.get("_id")
        event_id = _safe_int_id(es_id_raw)
        if event_id is None:
            continue

        score = float(h.get("_score") or 0.0)
        distance_km = None
        fields = h.get("fields") or {}
        if isinstance(fields, dict) and "distance_km" in fields and isinstance(fields["distance_km"], list) and fields["distance_km"]:
            try:
                distance_km = float(fields["distance_km"][0])
            except Exception:
                distance_km = None

        es_order_ids.append(event_id)
        meta_by_id[event_id] = {
            "score": score,
            "relevance": _relevance_label(score),
            "distance_km": distance_km,  # None si pas de geo ou si event n'a pas de localisation
        }

    # 2) Fetch full events from DB for those IDs
    events = fetch_events_with_relations_by_ids(es_order_ids)

    # 3) Map DB events by id (adapte si ton champ s'appelle autrement)
    # On essaye plusieurs clés possibles par sécurité.
    db_by_id: Dict[int, dict] = {}
    for ev in events:
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("eventId")
        try:
            ev_id_int = int(ev_id)
        except Exception:
            continue
        db_by_id[ev_id_int] = ev

    # 4) Merge in ES order + add relevance + distance_km
    merged: List[dict] = []
    for eid in es_order_ids:
        ev = db_by_id.get(eid)
        if not ev:
            continue
        meta = meta_by_id.get(eid, {})
        merged.append(
            {
                **ev,
                "score": meta.get("score", 0.0),
                "relevance": meta.get("relevance", "FAIBLE"),
                "distance_km": meta.get("distance_km", None),
            }
        )

    has_more = (from_ + per_page) < total_value

    return {
        "events": merged,
        "page": page,
        "per_page": per_page,
        "total_count": total_value,
        "has_more": has_more,
    }


@router.get("/search")
def search(
    q: str = Query("", description="Texte de recherche (peut être vide)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    sigma_km: float = Query(5.0, ge=0.1, le=100.0, description="Sigma gaussien en km"),
    geo_weight: float = Query(1.0, ge=0.0, le=10.0),
    vec_weight: float = Query(1.0, ge=0.0, le=10.0),
):
    return search_events_and_hydrate(
        q=q,
        page=page,
        per_page=per_page,
        lat=lat,
        lon=lon,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
    )
