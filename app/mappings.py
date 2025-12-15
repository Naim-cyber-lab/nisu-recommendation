from app.schemas import *

def to_event_out(e) -> EventOut:
    return EventOut(
        id=e.get("id"),
        creatorWinker=WinkerOut.model_validate(e.get("creatorWinker")) if getattr(e, "creatorWinker", None) else None,
        participants=[
            ParticipeWinkerOut(
                id=p.id,
                rang=getattr(p, "rang", None),
                nbUnseen=getattr(p, "nbUnseen", None),
                dateOrder=str(getattr(p, "dateOrder", None)) if getattr(p, "dateOrder", None) else None,
                participeWinker=WinkerOut.model_validate(p.participeWinker) if getattr(p, "participeWinker", None) else None,
            )
            for p in (getattr(e, "participants", None) or [])
        ],

        titre=getattr(e, "titre", None),
        datePublication=str(getattr(e, "datePublication", None)) if getattr(e, "datePublication", None) else None,
        adresse=getattr(e, "adresse", None),
        city=getattr(e, "city", None),
        region=getattr(e, "region", None),
        subregion=getattr(e, "subregion", None),
        pays=getattr(e, "pays", None),
        codePostal=getattr(e, "codePostal", None),

        ageMinimum=getattr(e, "ageMinimum", None),
        ageMaximum=getattr(e, "ageMaximum", None),

        accessOuvert=getattr(e, "accessOuvert", None),
        accessTous=getattr(e, "accessTous", None),
        accessFille=getattr(e, "accessFille", None),
        accessGarcon=getattr(e, "accessGarcon", None),
        accessFollow=getattr(e, "accessFollow", None),
        accessFollower=getattr(e, "accessFollower", None),

        bioEvent=getattr(e, "bioEvent", None),
        bioEvent_fr=getattr(e, "bioEvent_fr", None),
        titre_fr=getattr(e, "titre_fr", None),

        moyenneAge=getattr(e, "moyenneAge", None),
        nbFille=getattr(e, "nbFille", None),
        nbGarcon=getattr(e, "nbGarcon", None),
        nbComment=getattr(e, "nbComment", None),
        numberView=getattr(e, "numberView", None),

        lon=getattr(e, "lon", None),
        lat=getattr(e, "lat", None),

        detailsAddress=getattr(e, "detailsAddress", None),
        accessComment=getattr(e, "accessComment", None),

        containReduction=getattr(e, "containReduction", None),
        prixInitial=getattr(e, "prixInitial", None),
        prixReduction=getattr(e, "prixReduction", None),
        textReduction=getattr(e, "textReduction", None),

        needReservation=getattr(e, "needReservation", None),
        linkReservation=getattr(e, "linkReservation", None),

        currentNbParticipants=getattr(e, "currentNbParticipants", None),
        maxNumberParticipant=getattr(e, "maxNumberParticipant", None),
        isFull=getattr(e, "isFull", None),

        website=getattr(e, "website", None),
        urlGoogleMapsAvis=getattr(e, "urlGoogleMapsAvis", None),
        urlAjoutGoogleMapsAvis=getattr(e, "urlAjoutGoogleMapsAvis", None),
        nb_conversations=getattr(e, "nb_conversations", None),
        nbStories=getattr(e, "nbStories", None),
    )
