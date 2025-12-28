from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Any, Dict, List
from elasticsearch import ApiError

from app.core.es import es_client
from app.embeddings.service import embed_text
from app.api.v1.sql.fetch_events_with_relations_by_ids import fetch_events_with_relations_by_ids

router = APIRouter()

INDEX = "nisu_events"
VECTOR_DIMS = 768


def _to_float_list(vec) -> List[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    try:
        return [float(x) for x in vec]
    except Exception:
        return []


def _relevance_label(score: float) -> str:
    if score >= 2.5:
        return "TRÈS_PERTINENT"
    if score >= 1.2:
        return "PERTINENT"
    if score >= 0.5:
        return "MOYEN"
    return "FAIBLE"


def _distance_penalty_label(distance_km: Optional[float], soft_radius_km: float) -> Optional[str]:
    if distance_km is None:
        return None
    return "LOIN" if distance_km > soft_radius_km else None


def _final_relevance(score: float, distance_km: Optional[float], soft_radius_km: float) -> str:
    base = _relevance_label(score)
    if distance_km is None:
        return base

    if distance_km > soft_radius_km * 2:
        return "HORS_ZONE"
    if distance_km > soft_radius_km:
        return "FAIBLE"
    return base


def _build_query(
    q: str,
    from_: int,
    size: int,
    lat: Optional[float],
    lon: Optional[float],
    sigma_km: float,
    geo_weight: float,
    vec_weight: float,
    soft_radius_km: float,  # utilisé pour labels côté API (pas ES)
    hard_max_radius_km: Optional[float],
) -> Dict[str, Any]:
    q = (q or "").strip()
    is_query_empty = len(q) == 0

    # Match-all garanti (même si aucun should ne matche)
    if is_query_empty:
        base_query: Dict[str, Any] = {"match_all": {}}
    else:
        base_query = {
            "bool": {
                "must": [{"match_all": {}}],
                "should": [
                    {"match": {"titre": {"query": q, "boost": 2}}},
                    {"match": {"bio": {"query": q}}},
                ],
                "minimum_should_match": 0,
            }
        }

    vector = _to_float_list(embed_text(q)) if not is_query_empty else []
    use_vectors = len(vector) == VECTOR_DIMS

    has_geo = lat is not None and lon is not None

    source_includes = ["event_id"]

    functions: List[Dict[str, Any]] = []

    # 1) vecteurs
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

    # 2) geo gaussian + distance_km en fields
    script_fields: Dict[str, Any] = {}
    if has_geo:
        geo_script = """
          if (doc['localisation'].size() == 0) return 0.0;

          double d = doc['localisation'].arcDistance(params.lat, params.lon); // meters
          double sigma = params.sigma_m;
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

        script_fields["distance_km"] = {
            "script": {
                "source": """
                  if (doc['localisation'].size() == 0) return null;
                  return doc['localisation'].arcDistance(params.lat, params.lon) / 1000.0;
                """,
                "params": {"lat": float(lat), "lon": float(lon)},
            }
        }

        # Option HARD (filtre dur)
        if hard_max_radius_km is not None:
            base_bool = base_query.get("bool") if isinstance(base_query, dict) else None
            if isinstance(base_bool, dict):
                base_bool.setdefault("filter", [])
                base_bool["filter"].append(
                    {
                        "geo_distance": {
                            "distance": f"{float(hard_max_radius_km)}km",
                            "localisation": {"lat": float(lat), "lon": float(lon)},
                        }
                    }
                )
            else:
                base_query = {
                    "bool": {
                        "must": [{"match_all": {}}],
                        "filter": [
                            {
                                "geo_distance": {
                                    "distance": f"{float(hard_max_radius_km)}km",
                                    "localisation": {"lat": float(lat), "lon": float(lon)},
                                }
                            }
                        ],
                    }
                }

    # 3) boost field
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

    # ✅ IMPORTANT:
    # - score_mode="sum": on additionne les contributions (vecteurs + geo + boost)
    # - boost_mode="multiply": la base query (texte) est multipliée par les fonctions
    #   => la distance devient une vraie pénalité (si loin => facteur proche de 0)
    body: Dict[str, Any] = {
        "track_total_hits": True,
        "from": from_,
        "size": size,
        "_source": {"includes": source_includes},
        "query": {
            "function_score": {
                "query": base_query,
                "functions": functions,
                "score_mode": "sum",
                "boost_mode": "multiply",  # ✅ au lieu de "sum"
            }
        },
    }

    if script_fields:
        body["script_fields"] = script_fields

    return body


def search_events_paginated(
    q: str,
    page: int,
    per_page: int,
    lat: Optional[float],
    lon: Optional[float],
    sigma_km: float,
    geo_weight: float,
    vec_weight: float,
    soft_radius_km: float,
    hard_max_radius_km: Optional[float],
) -> Dict[str, Any]:
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    from_ = (page - 1) * per_page

    body = _build_query(
        q=q,
        from_=from_,
        size=per_page,
        lat=lat,
        lon=lon,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
        soft_radius_km=soft_radius_km,
        hard_max_radius_km=hard_max_radius_km,
    )

    try:
        res = es_client.search(index=INDEX, body=body)
    except ApiError as e:
        detail = getattr(e, "info", None) or str(e)
        raise HTTPException(status_code=400, detail={"elasticsearch_error": detail})

    hits = res.get("hits", {}).get("hits", [])
    total = res.get("hits", {}).get("total", {})
    total_count = int(total.get("value", 0)) if isinstance(total, dict) else int(total or 0)

    event_ids: List[int] = []
    meta_by_id: Dict[int, Dict[str, Any]] = {}

    for h in hits:
        src = h.get("_source") or {}
        raw_event_id = src.get("event_id") or h.get("_id")
        try:
            eid = int(raw_event_id)
        except Exception:
            continue

        score = float(h.get("_score") or 0.0)

        distance_km = None
        fields = h.get("fields") or {}
        if isinstance(fields, dict) and "distance_km" in fields:
            v = fields.get("distance_km")
            if isinstance(v, list) and v:
                try:
                    distance_km = float(v[0])
                except Exception:
                    distance_km = None

        event_ids.append(eid)
        meta_by_id[eid] = {"score": score, "distance_km": distance_km}

    events_db = fetch_events_with_relations_by_ids(event_ids)

    db_by_id: Dict[int, dict] = {}
    for ev in events_db:
        raw_id = ev.get("id") or ev.get("event_id") or ev.get("eventId") or ev.get("_id")
        try:
            db_by_id[int(raw_id)] = ev
        except Exception:
            continue

    merged: List[dict] = []
    for eid in event_ids:
        ev = db_by_id.get(eid)
        if not ev:
            continue

        meta = meta_by_id.get(eid, {})
        score = float(meta.get("score", 0.0))
        distance_km = meta.get("distance_km", None)

        merged.append(
            {
                **ev,
                "score": score,
                "distance_km": distance_km,
                "relevance": _final_relevance(score, distance_km, soft_radius_km),
                "distance_label": _distance_penalty_label(distance_km, soft_radius_km),
            }
        )

    # (Optionnel) tri sécurité côté API
    merged.sort(key=lambda e: float(e.get("score") or 0.0), reverse=True)

    has_more = (from_ + per_page) < total_count

    return {
        "es_hits_count": len(hits),
        "merged_count": len(merged),
        "total_count": total_count,
        "has_more": has_more,
        "first_es_ids": [int(e["id"]) for e in merged[:10] if "id" in e],
        "events": merged,
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
    soft_radius_km: float = Query(30.0, ge=1.0, le=300.0, description="Au-delà => pertinence dégradée"),
    hard_max_radius_km: Optional[float] = Query(None, ge=1.0, le=1000.0, description="Optionnel: filtre dur"),
):
    return search_events_paginated(
        q=q,
        page=page,
        per_page=per_page,
        lat=lat,
        lon=lon,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
        soft_radius_km=soft_radius_km,
        hard_max_radius_km=hard_max_radius_km,
    )


@router.get("/debug/es")
def debug_es():
    info = es_client.info()
    count = es_client.count(index=INDEX)
    return {
        "INDEX": INDEX,
        "cluster_name": info.get("cluster_name"),
        "cluster_uuid": info.get("cluster_uuid"),
        "es_version": (info.get("version") or {}).get("number"),
        "count": count.get("count"),
    }
