from typing import List
from app.core.es import es
from app.core.config import INDEX_WINKERS, INDEX_EVENTS
from app.schemas import RecommendedWinker, RecommendedEvent, WinkerOut, EventOut
from app.repositories.winkers import get_winker_from_es
from app.repositories.events import get_event_from_es


def recommend_winkers_for_winker(winker_id: int, size: int = 10) -> List[RecommendedWinker]:
    user = get_winker_from_es(winker_id)
    if not user or not user.preference_vector:
        return []

    base_filter = {
        "bool": {
            "must": [],
            "must_not": [{"term": {"_id": winker_id}}],
        }
    }

    # Exemple : privilégier la même région
    if user.region:
        base_filter["bool"]["must"].append({"term": {"region": user.region}})

    query = {
        "script_score": {
            "query": base_filter,
            "script": {
                "source": "cosineSimilarity(params.qv, 'preference_vector') + 1.0",
                "params": {"qv": user.preference_vector},
            },
        }
    }

    res = es.search(index=INDEX_WINKERS, body={"query": query, "size": size})
    results: List[RecommendedWinker] = []

    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        latlon = src.get("latlon") or {}
        w = WinkerOut(
            id=int(hit["_id"]),
            username=src.get("username"),
            email=src.get("email"),
            sexe=src.get("sexe"),
            age=src.get("age"),
            city=src.get("city"),
            region=src.get("region"),
            subregion=src.get("subregion"),
            pays=src.get("pays"),
            lat=latlon.get("lat"),
            lon=latlon.get("lon"),
            visible_tags=src.get("visible_tags", []),
            preference_vector=src.get("preference_vector"),
            meet_eligible=src.get("meet_eligible"),
            mails_eligible=src.get("mails_eligible"),
            score=hit["_score"],
        )
        results.append(RecommendedWinker(winker=w, score=hit["_score"]))

    return results


def recommend_events_for_winker(winker_id: int, size: int = 10) -> List[RecommendedEvent]:
    user = get_winker_from_es(winker_id)
    if not user or not user.preference_vector:
        return []

    bool_filter = {"must": [], "filter": []}

    # Filtre région / pays
    if user.region:
        bool_filter["filter"].append({"term": {"region": user.region}})

    # Exemple : filtre d'âge si on connaît l'âge du winker
    if user.age is not None:
        bool_filter["filter"].append({
            "range": {"ageMinimum": {"lte": user.age}}
        })
        bool_filter["filter"].append({
            "range": {"ageMaximum": {"gte": user.age}}
        })

    query = {
        "script_score": {
            "query": {"bool": bool_filter},
            "script": {
                "source": "cosineSimilarity(params.qv, 'vectorPreferenceEvent') + 1.0",
                "params": {"qv": user.preference_vector},
            },
        }
    }

    res = es.search(index=INDEX_EVENTS, body={"query": query, "size": size})
    results: List[RecommendedEvent] = []

    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        latlon = src.get("latlon") or {}
        e = EventOut(
            id=int(hit["_id"]),
            titre=src.get("titre"),
            bioEvent=src.get("bioEvent"),
            city=src.get("city"),
            region=src.get("region"),
            subregion=src.get("subregion"),
            pays=src.get("pays"),
            codePostal=src.get("codePostal"),
            lat=latlon.get("lat"),
            lon=latlon.get("lon"),
            dateEvent=src.get("dateEvent"),
            datePublication=src.get("datePublication"),
            ageMinimum=src.get("ageMinimum"),
            ageMaximum=src.get("ageMaximum"),
            accessFille=src.get("accessFille"),
            accessGarcon=src.get("accessGarcon"),
            accessTous=src.get("accessTous"),
            hastagEvents=src.get("hastagEvents", []),
            meetEligible=src.get("meetEligible"),
            planTripElligible=src.get("planTripElligible"),
            currentNbParticipants=src.get("currentNbParticipants"),
            maxNumberParticipant=src.get("maxNumberParticipant"),
            isFull=src.get("isFull"),
            vectorPreferenceEvent=src.get("vectorPreferenceEvent"),
            score=hit["_score"],
        )
        results.append(RecommendedEvent(event=e, score=hit["_score"]))

    return results
