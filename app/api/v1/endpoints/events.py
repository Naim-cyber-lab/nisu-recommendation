from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Any, Dict, List
from elasticsearch import ApiError

from app.core.es import es_client
from app.embeddings.service import embed_text

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
    Labels de pertinence basés sur _score.
    Ajuste ces seuils après quelques tests réels.
    """
    if score >= 2.5:
        return "TRÈS_PERTINENT"
    if score >= 1.2:
        return "PERTINENT"
    if score >= 0.5:
        return "MOYEN"
    return "FAIBLE"


def search_events(
    q: str,
    size: int = 20,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    sigma_km: float = 5.0,      # largeur gaussienne (plus grand => boost plus large)
    geo_weight: float = 1.0,    # poids du score géo dans le total
    vec_weight: float = 1.0,    # poids du score vecteur dans le total
    text_weight: float = 1.0,   # poids du score texte (celui d'ES)
) -> List[Dict[str, Any]]:
    q = (q or "").strip()

    # Toujours des réponses: si q vide, on bascule en match_all.
    is_query_empty = len(q) == 0

    # Embedding (si dims mismatch -> on ne casse pas, on met un vecteur vide et score vecteur = 0)
    vector = _to_float_list(embed_text(q)) if not is_query_empty else []
    use_vectors = len(vector) == VECTOR_DIMS

    # Base query: on NE force PAS minimum_should_match=1, sinon 0 résultats si rien ne match.
    # On préfère retourner des docs avec un score faible.
    if is_query_empty:
        base_query: Dict[str, Any] = {"match_all": {}}
    else:
        base_query = {
            "bool": {
                "should": [
                    {"match": {"titre": {"query": q, "boost": 2}}},
                    {"match": {"bio": {"query": q}}},
                ],
                "minimum_should_match": 0,  # ✅ garantit qu'on peut quand même retourner des docs
            }
        }

    # Function score = score texte + score vecteur + score geo gaussien
    # - score_mode="sum" => addition des fonctions
    # - boost_mode="replace" => on remplace le score ES par le score combiné (plus contrôlable)
    functions: List[Dict[str, Any]] = []

    # 1) Vecteurs (cosine similarity) -> script_score dans function_score
    # On "guard" chaque champ vectoriel pour éviter runtime error.
    # On met le vecteur à params.q seulement si correct, sinon script renvoie 0.
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

      // Clamp si tu veux éviter des extrêmes (optionnel)
      // if (s > 3.0) s = 3.0;

      return s;
    """

    functions.append({
        "script_score": {
            "script": {
                "source": vec_script,
                "params": {
                    "useVec": use_vectors,
                    "q": vector if use_vectors else [],
                }
            }
        },
        "weight": vec_weight
    })

    # 2) Localisation gaussienne (sur la distance)
    # score_geo = exp(-0.5 * (d/sigma)^2)
    # d en mètres, sigma en mètres
    # -> proche => ~1, loin => -> 0
    if lat is not None and lon is not None:
        geo_script = """
          if (doc['localisation'].size() == 0) return 0.0;

          double d = doc['localisation'].arcDistance(params.lat, params.lon); // meters
          double sigma = params.sigma_m; // meters

          if (sigma <= 0) return 0.0;

          double x = d / sigma;
          double g = Math.exp(-0.5 * x * x); // gaussian

          return g;
        """

        functions.append({
            "script_score": {
                "script": {
                    "source": geo_script,
                    "params": {
                        "lat": float(lat),
                        "lon": float(lon),
                        "sigma_m": float(sigma_km) * 1000.0,
                    }
                }
            },
            "weight": geo_weight
        })

    # 3) (Optionnel) champ boost du doc si présent
    # Très utile si tu as "boost" dans ton mapping.
    functions.append({
        "field_value_factor": {
            "field": "boost",
            "missing": 0,
            "modifier": "sqrt"
        },
        "weight": 0.2
    })

    query: Dict[str, Any] = {
        "size": size,
        "query": {
            "function_score": {
                "query": base_query,
                "functions": functions,
                "score_mode": "sum",
                # On veut garder une influence du score texte ES :
                # boost_mode="sum" => score_final = score_texte + sum(functions)
                # boost_mode="replace" => score_final = sum(functions)
                #
                # Ici, tu dis "en fonction du function score on affiche pertinence".
                # Je te propose SUM : tu gardes le texte (quand il match), sinon il reste faible.
                "boost_mode": "sum",
            }
        }
    }

    try:
        res = es_client.search(index=INDEX, body=query)
    except ApiError as e:
        detail = getattr(e, "info", None) or str(e)
        raise HTTPException(status_code=400, detail={"elasticsearch_error": detail})

    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict[str, Any]] = []

    for hit in hits:
        score = float(hit.get("_score") or 0.0)
        src = hit.get("_source", {}) or {}

        out.append({
            **src,
            "id": hit.get("_id"),
            "score": score,
            "relevance": _relevance_label(score),  # ✅ label de pertinence
        })

    return out


@router.get("/search")
def search(
    q: str = Query("", description="Texte de recherche (peut être vide)"),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    sigma_km: float = Query(5.0, ge=0.1, le=100.0, description="Sigma gaussien en km"),
    geo_weight: float = Query(1.0, ge=0.0, le=10.0),
    vec_weight: float = Query(1.0, ge=0.0, le=10.0),
    size: int = Query(20, ge=1, le=100),
):
    return search_events(
        q=q,
        size=size,
        lat=lat,
        lon=lon,
        sigma_km=sigma_km,
        geo_weight=geo_weight,
        vec_weight=vec_weight,
    )
