from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Any, Dict, List
from elasticsearch import ApiError, NotFoundError

from app.core.es import es_client
from app.api.v1.sql.fetch_events_with_relations_by_ids import fetch_events_with_relations_by_ids

router = APIRouter()

INDEX = "nisu_events"
VECTOR_DIMS = 768


# ─── helpers (identiques à events.py) ────────────────────────────────────────

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
    if score >= 0.85:
        return "TRÈS_PERTINENT"
    if score >= 0.65:
        return "PERTINENT"
    if score >= 0.40:
        return "MOYEN"
    return "FAIBLE"


def _distance_label(distance_km: Optional[float], soft_radius_km: float) -> Optional[str]:
    if distance_km is None:
        return None
    return "LOIN" if distance_km > soft_radius_km else None


# ─── fetch the source event from ES ──────────────────────────────────────────

def _get_source_event(event_id: str) -> Dict[str, Any]:
    """
    Récupère le document source depuis ES.
    Construit un vecteur combiné depuis titre_vector, bio_vector, preferences_vector
    (moyenne pondérée : titre x2, bio x1, preferences x1).
    """
    try:
        doc = es_client.get(
            index=INDEX,
            id=event_id,
            source_includes=["titre_vector", "bio_vector", "preferences_vector", "localisation"],
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' introuvable dans l'index ES.")
    except ApiError as e:
        raise HTTPException(status_code=400, detail={"elasticsearch_error": getattr(e, "info", str(e))})

    src = doc.get("_source") or {}

    titre_vec = _to_float_list(src.get("titre_vector"))
    bio_vec = _to_float_list(src.get("bio_vector"))
    pref_vec = _to_float_list(src.get("preferences_vector"))

    # Moyenne pondérée des vecteurs disponibles (titre compte double)
    weighted: List[List[float]] = []
    weights: List[float] = []
    if len(titre_vec) == VECTOR_DIMS:
        weighted.append(titre_vec)
        weights.append(2.0)
    if len(bio_vec) == VECTOR_DIMS:
        weighted.append(bio_vec)
        weights.append(1.0)
    if len(pref_vec) == VECTOR_DIMS:
        weighted.append(pref_vec)
        weights.append(1.0)

    if not weighted:
        raise HTTPException(
            status_code=422,
            detail=f"L'event '{event_id}' n'a aucun vecteur valide (titre_vector, bio_vector, preferences_vector).",
        )

    total_weight = sum(weights)
    combined = [
        sum(weighted[j][i] * weights[j] for j in range(len(weighted))) / total_weight
        for i in range(VECTOR_DIMS)
    ]

    localisation = src.get("localisation")
    lat: Optional[float] = None
    lon: Optional[float] = None
    if isinstance(localisation, dict):
        lat = localisation.get("lat")
        lon = localisation.get("lon")

    return {"vector": combined, "lat": lat, "lon": lon}


# ─── build the "more like this vector + geo" query ───────────────────────────

def _build_similar_query(
    source_vector: List[float],
    source_lat: Optional[float],
    source_lon: Optional[float],
    exclude_event_id: str,
    from_: int,
    size: int,
    soft_radius_km: float,
    sigma_km: float,
    geo_weight: float,
    vec_weight: float,
    hard_max_radius_km: Optional[float],
) -> Dict[str, Any]:

    has_geo = source_lat is not None and source_lon is not None

    # ── Filtre : exclure l'event lui-même ────────────────────────────────────
    base_query: Dict[str, Any] = {
        "bool": {
            "must": [{"match_all": {}}],
            "must_not": [{"term": {"event_id": exclude_event_id}}],
        }
    }

    # ── Filtre dur optionnel ─────────────────────────────────────────────────
    if has_geo and hard_max_radius_km is not None:
        base_query["bool"].setdefault("filter", [])
        base_query["bool"]["filter"].append(
            {
                "geo_distance": {
                    "distance": f"{float(hard_max_radius_km)}km",
                    "localisation": {"lat": float(source_lat), "lon": float(source_lon)},
                }
            }
        )

    functions: List[Dict[str, Any]] = []

    # ── 1) Score vecteur : cosine similarity sur titre_vector + bio_vector ──────
    # Moyenne des similarités disponibles (titre compte double).
    # cosineSimilarity renvoie [-1, 1] ; on ramène à [0, 1] via (s+1)/2.
    vec_script = """
      double total = 0.0;
      double weight = 0.0;
      if (doc['titre_vector'].size() != 0) {
        total += ((cosineSimilarity(params.vec, 'titre_vector') + 1.0) / 2.0) * 2.0;
        weight += 2.0;
      }
      if (doc['bio_vector'].size() != 0) {
        total += ((cosineSimilarity(params.vec, 'bio_vector') + 1.0) / 2.0) * 1.0;
        weight += 1.0;
      }
      if (doc['preferences_vector'].size() != 0) {
        total += ((cosineSimilarity(params.vec, 'preferences_vector') + 1.0) / 2.0) * 1.0;
        weight += 1.0;
      }
      if (weight == 0.0) return 0.0;
      return total / weight;
    """
    functions.append(
        {
            "script_score": {
                "script": {
                    "source": vec_script,
                    "params": {"vec": source_vector},
                }
            },
            "weight": vec_weight,
        }
    )

    # ── 2) Facteur géo : plateau puis chute gaussienne ────────────────────────
    # geo_factor ∈ [0, 1]
    # - distance ≤ soft_radius_km  → 1.0  (pas de pénalité, le vecteur compte à 100%)
    # - distance > soft_radius_km  → exp(-((d-soft)/sigma)²)  → tend vers 0
    #
    # score_mode="multiply" donc le score final = vec_score × geo_factor
    # Plus l'event est loin, plus il doit être vectoriellement similaire pour remonter.
    script_fields: Dict[str, Any] = {}

    if has_geo:
        geo_script = """
          if (doc['localisation'].size() == 0) return 1.0;

          double dKm = doc['localisation'].arcDistance(params.lat, params.lon) / 1000.0;

          if (dKm <= params.soft_km) return 1.0;

          double x = (dKm - params.soft_km) / params.falloff_km;
          return Math.exp(-1.0 * x * x);
        """
        functions.append(
            {
                "script_score": {
                    "script": {
                        "source": geo_script,
                        "params": {
                            "lat": float(source_lat),
                            "lon": float(source_lon),
                            "soft_km": float(soft_radius_km),
                            "falloff_km": float(max(sigma_km, 0.1)),
                        },
                    }
                },
                "weight": geo_weight,
            }
        )

        # Champ calculé : distance en km (pour l'UX)
        script_fields["distance_km"] = {
            "script": {
                "source": """
                  if (doc['localisation'].size() == 0) return null;
                  return doc['localisation'].arcDistance(params.lat, params.lon) / 1000.0;
                """,
                "params": {"lat": float(source_lat), "lon": float(source_lon)},
            }
        }

    body: Dict[str, Any] = {
        "track_total_hits": True,
        "from": from_,
        "size": size,
        "_source": {"includes": ["event_id"]},
        "query": {
            "function_score": {
                "query": base_query,
                "functions": functions,
                # sum des fonctions puis multiply sur le score de base (match_all = 1.0)
                # Résultat : vec_score (normalisé) * geo_factor
                "score_mode": "multiply",
                "boost_mode": "replace",  # remplace le score BM25 (inutile ici) par notre calcul
            }
        },
    }

    if script_fields:
        body["script_fields"] = script_fields

    return body


# ─── paginated service ────────────────────────────────────────────────────────

def find_similar_events_paginated(
    event_id: str,
    page: int,
    per_page: int,
    soft_radius_km: float,
    sigma_km: float,
    geo_weight: float,
    vec_weight: float,
    hard_max_radius_km: Optional[float],
) -> Dict[str, Any]:
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    from_ = (page - 1) * per_page

    # 1) Récupérer le vecteur + localisation de l'event source
    source = _get_source_event(event_id)

    # 2) Construire et exécuter la requête ES
    body = _build_similar_query(
        source_vector=source["vector"],
        source_lat=source["lat"],
        source_lon=source["lon"],
        exclude_event_id=event_id,
        from_=from_,
        size=per_page,
        soft_radius_km=soft_radius_km,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
        hard_max_radius_km=hard_max_radius_km,
    )

    try:
        res = es_client.search(index=INDEX, body=body)
    except ApiError as e:
        raise HTTPException(status_code=400, detail={"elasticsearch_error": getattr(e, "info", str(e))})

    hits = res.get("hits", {}).get("hits", [])
    total = res.get("hits", {}).get("total", {})
    total_count = int(total.get("value", 0)) if isinstance(total, dict) else int(total or 0)

    # 3) Parser les hits
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

        distance_km: Optional[float] = None
        fields = h.get("fields") or {}
        if isinstance(fields, dict) and "distance_km" in fields:
            v = fields["distance_km"]
            if isinstance(v, list) and v:
                try:
                    distance_km = float(v[0])
                except Exception:
                    pass

        event_ids.append(eid)
        meta_by_id[eid] = {"score": score, "distance_km": distance_km}

    # 4) Hydratation SQL
    # Sur la page 1 : on inclut aussi l'event source pour le récupérer en position 0.
    source_eid = int(event_id)
    ids_to_fetch = ([source_eid] + event_ids) if page == 1 else event_ids
    events_db = fetch_events_with_relations_by_ids(ids_to_fetch)

    db_by_id: Dict[int, dict] = {}
    for ev in events_db:
        raw_id = ev.get("id") or ev.get("event_id") or ev.get("eventId") or ev.get("_id")
        try:
            db_by_id[int(raw_id)] = ev
        except Exception:
            continue

    # 5) Merge ES meta + SQL data
    merged: List[dict] = []
    for eid in event_ids:
        ev = db_by_id.get(eid)
        if not ev:
            continue

        meta = meta_by_id[eid]
        score = float(meta["score"])
        distance_km = meta.get("distance_km")

        merged.append(
            {
                **ev,
                "score": round(score, 6),
                "distance_km": round(distance_km, 2) if distance_km is not None else None,
                "relevance": _relevance_label(score),
                "distance_label": _distance_label(distance_km, soft_radius_km),
            }
        )

    merged.sort(key=lambda e: float(e.get("score") or 0.0), reverse=True)

    # 6) Page 1 : injecter l'event source en position 0 (celui sur lequel l'user a cliqué).
    #    On le retire des similaires s'il y était déjà (must_not l'exclut normalement, mais sécurité).
    if page == 1:
        source_ev = db_by_id.get(source_eid)
        if source_ev:
            # Supprimer une éventuelle occurrence dans merged (sécurité)
            merged = [e for e in merged if int(e.get("id") or e.get("event_id") or 0) != source_eid]

            source_entry = {
                **source_ev,
                "score": 1.0,           # score symbolique maximal
                "distance_km": None,
                "relevance": "TRÈS_PERTINENT",
                "distance_label": None,
                "is_source": True,      # flag UX optionnel côté frontend
            }
            merged = [source_entry] + merged

    has_more = (from_ + per_page) < total_count

    return {
        "source_event_id": event_id,
        "source_has_geo": source["lat"] is not None,
        "es_hits_count": len(hits),
        "merged_count": len(merged),
        "total_count": total_count,
        "page": page,
        "per_page": per_page,
        "has_more": has_more,
        "events": merged,
    }


# ─── FastAPI route ────────────────────────────────────────────────────────────

@router.get("/{event_id}/similar")
def get_similar_events(
    event_id: str,
    page: int = Query(1, ge=1, description="Page courante (commence à 1)."),
    per_page: int = Query(20, ge=1, le=100, description="Nombre d'events par page."),
    soft_radius_km: float = Query(
        10.0,
        ge=1.0,
        le=500.0,
        description=(
            "Rayon plateau (km) : en-deçà, la distance ne pénalise pas du tout. "
            "Au-delà, la pénalité geo s'applique progressivement."
        ),
    ),
    sigma_km: float = Query(
        30.0,
        ge=0.1,
        le=500.0,
        description=(
            "Vitesse de décroissance de la pénalité géo après soft_radius_km. "
            "Grand sigma = pénalité douce (events lointains toujours visibles si très similaires). "
            "Petit sigma = pénalité sévère (seuls les events proches remontent)."
        ),
    ),
    geo_weight: float = Query(
        1.5,
        ge=0.0,
        le=10.0,
        description="Poids du facteur géo dans le score final. 0 = géo ignorée.",
    ),
    vec_weight: float = Query(
        1.0,
        ge=0.0,
        le=10.0,
        description="Poids de la similarité vectorielle dans le score final.",
    ),
    hard_max_radius_km: Optional[float] = Query(
        None,
        ge=1.0,
        le=5000.0,
        description="Optionnel : filtre dur — exclut tout event au-delà de cette distance.",
    ),
):
    """
    Retourne les events les plus similaires à `event_id`.

    ## Logique de scoring

    `score = vec_score × geo_factor`

    - **vec_score** : similarité cosinus sur `embedding_vector`, normalisée dans [0, 1].
    - **geo_factor** : facteur multiplicatif [0, 1] basé sur la distance.
        - `distance ≤ soft_radius_km` → **1.0** (pas de pénalité, seul le vecteur compte)
        - `distance > soft_radius_km` → `exp(-((d - soft) / sigma)²)` → tend vers 0

    Plus un event est loin, plus il doit être **vectoriellement proche** pour apparaître en tête.
    Si l'event source n'a pas de localisation, `geo_factor = 1.0` pour tous les résultats.

    ## Pagination (infinite scroll)
    Utilisez `page` et `per_page` ; continuez tant que `has_more = true`.
    """
    return find_similar_events_paginated(
        event_id=event_id,
        page=page,
        per_page=per_page,
        soft_radius_km=soft_radius_km,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
        hard_max_radius_km=hard_max_radius_km,
    )