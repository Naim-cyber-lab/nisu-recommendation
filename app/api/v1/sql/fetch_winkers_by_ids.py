from typing import Any, Dict, List, Sequence
from app.core.db import *


def fetch_winkers_by_ids(ids: Sequence[int]) -> List[Dict[str, Any]]:
    """
    Récupère les winkers + leurs fichiers (profil_filesWinker) en 1 query.
    - Préserve l'ordre des ids d'entrée
    - Déduplique les ids
    - Ajoute un champ 'filesWinker' = liste de fichiers [{id, image}, ...]
    """
    if not ids:
        return []

    # Dedup + preserve order
    seen = set()
    ordered_ids: List[int] = []
    for i in ids:
        try:
            ii = int(i)
        except Exception:
            continue
        if ii not in seen:
            seen.add(ii)
            ordered_ids.append(ii)

    if not ordered_ids:
        return []

    # ⚠️ table join: tu m’as demandé "profil_filesWinker"
    # En SQL, les noms sont généralement snake_case => à adapter si besoin:
    # - profil_filesWinker
    # - profil_fileswinker
    # - files_winker
    # etc.
    sql = """
        SELECT
            w.*,
            COALESCE(
                json_agg(
                    json_build_object(
                        'id', f.id,
                        'image', f.image
                    )
                ) FILTER (WHERE f.id IS NOT NULL),
                '[]'::json
            ) AS "filesWinker"
        FROM profil_winker w
        LEFT JOIN profil_filesWinker f
            ON f.winker_id = w.id
        WHERE w.id = ANY(%s)
        GROUP BY w.id
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ordered_ids,))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]

    winkers = [dict(zip(cols, row)) for row in rows]

    # Re-order to match ES order
    by_id = {w.get("id"): w for w in winkers}
    return [by_id[i] for i in ordered_ids if i in by_id]
