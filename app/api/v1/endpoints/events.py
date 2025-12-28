from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Any, Dict, List

from elasticsearch import ApiError

from app.core.es import es_client
from app.embeddings.service import embed_text

router = APIRouter()

INDEX = "nisu_events"
VECTOR_DIMS = 768


def _to_float_list(vec) -> List[float]:
    """Ensure we send a plain JSON list[float] to Elasticsearch."""
    if vec is None:
        return []
    if hasattr(vec, "tolist"):  # numpy / torch
        vec = vec.tolist()
    try:
        return [float(x) for x in vec]
    except Exception:
        return []


def search_events(
    q: str,
    size: int = 20,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 25.0,
) -> List[Dict[str, Any]]:
    q = (q or "").strip()
    if not q:
        return []

    # --- embedding ---
    vector = _to_float_list(embed_text(q))

    # If dims mismatch, do a safe text-only fallback (prevents 500)
    use_vectors = len(vector) == VECTOR_DIMS

    # --- base bool query (text + optional geo filter) ---
    bool_query: Dict[str, Any] = {
        "should": [
            {"match": {"titre": {"query": q, "boost": 2}}},
            {"match": {"bio": {"query": q}}},
        ],
        "minimum_should_match": 1,
    }

    # Geo: filter by distance if lat/lon provided
    if lat is not None and lon is not None:
        bool_query["filter"] = [
            {
                "geo_distance": {
                    "distance": f"{radius_km}km",
                    "localisation": {"lat": lat, "lon": lon},
                }
            }
        ]

    if use_vectors:
        # ✅ script_score with guards for missing vectors + optional geo boost
        # NOTE: localisation is a geo_point in your mapping
        source = """
          double s = 0.0;

          // Vector similarity (guard missing vectors)
          if (doc['titre_vector'].size() != 0) {
            s += cosineSimilarity(params.q, 'titre_vector') * 2.0;
          }
          if (doc['bio_vector'].size() != 0) {
            s += cosineSimilarity(params.q, 'bio_vector');
          }
          if (doc['preferences_vector'].size() != 0) {
            s += cosineSimilarity(params.q, 'preferences_vector');
          }

          // Optional: boost closer events (smooth, bounded)
          if (params.hasGeo == true && doc['localisation'].size() != 0) {
            double d = doc['localisation'].arcDistance(params.lat, params.lon); // meters
            // boost in [~0, 1], closer => higher
            s += 1.0 / (1.0 + (d / 1000.0)); // distance in km
          }

          // Optional: include boost field if you index it
          if (doc['boost'].size() != 0) {
            s += (double)doc['boost'].value * 0.05;
          }

          return s;
        """

        query: Dict[str, Any] = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"bool": bool_query},
                    "script": {
                        "source": source,
                        "params": {
                            "q": vector,
                            "hasGeo": lat is not None and lon is not None,
                            "lat": float(lat) if lat is not None else 0.0,
                            "lon": float(lon) if lon is not None else 0.0,
                        },
                    },
                }
            },
        }
    else:
        # ✅ fallback text-only (still supports geo filter)
        query = {
            "size": size,
            "query": {"bool": bool_query},
        }

    try:
        res = es_client.search(index=INDEX, body=query)
    except ApiError as e:
        # Print ES details for debugging (runtime error root cause)
        detail = getattr(e, "info", None) or str(e)
        raise HTTPException(status_code=400, detail={"elasticsearch_error": detail})

    hits = res.get("hits", {}).get("hits", [])
    return [
        {
            **hit.get("_source", {}),
            "id": hit.get("_id"),
            "score": hit.get("_score"),
        }
        for hit in hits
    ]


@router.get("/search")
def search(
    q: str = Query(..., min_length=1),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    radius_km: float = Query(25.0, ge=0.1, le=200.0),
    size: int = Query(20, ge=1, le=100),
):
    return search_events(q=q, size=size, lat=lat, lon=lon, radius_km=radius_km)
