from typing import Any, Dict, List, Optional

def build_winker_profile_text(w: Dict[str, Any]) -> str:
    """
    Texte qui représente le winker DEMANDEUR (celui qui cherche des gens).
    """
    parts: List[str] = []

    bio = (w.get("bio") or "").strip()
    if bio:
        parts.append(bio)

    # préférences/centres d'intérêt
    prefs = w.get("listPreference") or w.get("preferences") or w.get("tags")
    if isinstance(prefs, str) and prefs.strip() not in ("", "null", "[]", "{}"):
        parts.append(prefs)
    elif isinstance(prefs, list) and prefs:
        parts.append(" ".join([str(x) for x in prefs]))

    # localisation (signal faible mais stable)
    city = (w.get("city") or "").strip()
    region = (w.get("region") or "").strip()
    if city or region:
        parts.append(" ".join([p for p in [city, region] if p]))

    # historique (super signal si pertinent)
    last_search = (w.get("derniereRechercheEvent") or "").strip()
    if last_search and last_search not in ("{}", "[]", "null"):
        parts.append(last_search)

    return " | ".join(parts).strip()


def build_candidate_winker_text(w: Dict[str, Any]) -> str:
    """
    Texte qui représente un winker CANDIDAT (dans l’index ES).
    Idéalement même structure que build_winker_profile_text, mais sans leakage.
    """
    parts: List[str] = []

    bio = (w.get("bio") or "").strip()
    if bio:
        parts.append(bio)

    prefs = w.get("listPreference") or w.get("preferences") or w.get("tags")
    if isinstance(prefs, str) and prefs.strip() not in ("", "null", "[]", "{}"):
        parts.append(prefs)
    elif isinstance(prefs, list) and prefs:
        parts.append(" ".join([str(x) for x in prefs]))

    city = (w.get("city") or "").strip()
    region = (w.get("region") or "").strip()
    if city or region:
        parts.append(" ".join([p for p in [city, region] if p]))

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
