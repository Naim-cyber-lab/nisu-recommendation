from app.core.db import *


FETCH_EVENTS_SQL = """
WITH input_ids AS (
    SELECT *
    FROM unnest(%s::int[]) WITH ORDINALITY AS t(id, ord)
)
SELECT
    e.id,
    e.titre,
    e."dateDebut",
    e."datePublication",
    e."numberDaysFromPublication",
    e."coeffDate",
    e."dateComplete",
    e.adresse,
    e.city,
    e.region,
    e.subregion,
    e.pays,
    e."codePostal",
    e."ageMinimum",
    e."ageMaximum",
    e."accessOuvert",
    e."accessTous",
    e."accessFille",
    e."accessGarcon",
    e."accessFollow",
    e."accessFollower",
    e."bioEvent",
    e."bioEvent_fr",
    e."titre_fr",
    e.audio,
    e."moyenneAge",
    e."nbFille",
    e."nbGarcon",
    e."nbLike",
    e."nbHaha",
    e."nbLove",
    e."nbAngry",
    e."nbSad",
    e."nbWow",
    e."numberView",
    e.lon,
    e.lat,
    e."accessComment",
    e."containReduction",
    e."prixInitial",
    e."prixReduction",
    e."firstPreference",
    e."firstPhotoLocalisation",
    e.remarque,
    e."priceEvent",
    e."horaireDebut",
    e."textForSearchingBar",
    e."vectorPreferenceEvent",
    e."hastagEvents",
    e."textReduction",
    e."currentLangue",
    e."importanteInformation",
    e."needReservation",
    e."linkReservation",
    e."isFull",
    e."detailsAddress",
    e."currentNbParticipants",
    e."maxNumberParticipant",
    e."nbComment",
    e.website,
    e."urlGoogleMapsAvis",
    e."urlAjoutGoogleMapsAvis",
    e."nb_conversations",
    e."nbStories",

    -- creatorWinker
    to_jsonb(cw.*) AS "creatorWinker",

    -- participants
    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'id', pw.id,
                'rang', pw.rang,
                'nbUnseen', pw."nbUnseen",
                'dateOrder', pw."dateOrder",
                'participeWinker', to_jsonb(wp.*)
            )
            ORDER BY pw.rang ASC, pw."dateOrder" ASC
        ) FILTER (WHERE pw.id IS NOT NULL),
        '[]'::jsonb
    ) AS participants

FROM input_ids i
JOIN profil_event e ON e.id = i.id

LEFT JOIN winker cw
    ON cw.id = e."creatorWinker_id"

LEFT JOIN participe_winker pw
    ON pw."event_id" = e.id
    AND pw."groupPrive_id" IS NULL

LEFT JOIN profil_winker wp
    ON wp.id = pw."participeWinker_id"

GROUP BY i.ord, e.id, cw.id
ORDER BY i.ord;
"""

def fetch_events_with_relations_by_ids(event_ids: list[int]) -> list[dict]:
    if not event_ids:
        return []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(FETCH_EVENTS_SQL, (event_ids,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]

