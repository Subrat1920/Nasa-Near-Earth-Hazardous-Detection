"""
routers/asteroids.py — All asteroid-related endpoints (READ-ONLY SELECT queries)
"""
import json
import asyncio
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from database import get_pool

router = APIRouter(tags=["Asteroids"])


# ──────────────────────────────────────────────
# GET /api/stats
# ──────────────────────────────────────────────
@router.get("/stats")
async def get_stats(pool=Depends(get_pool)):
    """Global KPI stats for the hero counters."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT id)                                          AS unique_asteroids,
                COUNT(*)                                                    AS total_approaches,
                COUNT(DISTINCT id) FILTER (WHERE is_potentially_hazardous) AS unique_pho,
                COUNT(DISTINCT id) FILTER (WHERE is_sentry_object)         AS unique_sentry,
                MIN(miss_distance_km)                                       AS min_miss_distance_km,
                MAX(relative_velocity_kps)                                  AS max_velocity_kps,
                MAX(close_approach_date)                                    AS last_approach_date
            FROM train_neo
            """
        )
    return dict(row)


# ──────────────────────────────────────────────
# GET /api/asteroids/stream  (SSE — for 3D scene)
# ──────────────────────────────────────────────
@router.get("/asteroids_all")
async def get_all_asteroids(pool=Depends(get_pool)):
    """Fetches all 32,001 unique asteroids at once for the 3D scene."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (id)
                id, name,
                min_diameter_km, max_diameter_km,
                relative_velocity_kps, miss_distance_km,
                is_potentially_hazardous, is_sentry_object,
                close_approach_date, epoch_date_close_approach
            FROM train_neo
            ORDER BY id, epoch_date_close_approach DESC
            """
        )
    return [dict(r) for r in rows]



# ──────────────────────────────────────────────
# GET /api/asteroids  (paginated unique list)
# ──────────────────────────────────────────────
@router.get("/asteroids")
async def list_asteroids(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    hazardous: bool = Query(None),
    sentry: bool = Query(None),
    pool=Depends(get_pool),
):
    offset = (page - 1) * limit
    filters = []
    if hazardous is not None:
        filters.append(f"is_potentially_hazardous = {str(hazardous).lower()}")
    if sentry is not None:
        filters.append(f"is_sentry_object = {str(sentry).lower()}")

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH latest AS (
                SELECT DISTINCT ON (id)
                    id, name, min_diameter_km, max_diameter_km,
                    relative_velocity_kps, miss_distance_km,
                    is_potentially_hazardous, is_sentry_object,
                    close_approach_date, epoch_date_close_approach
                FROM train_neo
                ORDER BY id, epoch_date_close_approach DESC
            )
            SELECT * FROM latest
            {where_clause}
            ORDER BY id
            LIMIT {limit} OFFSET {offset}
            """
        )
        total = await conn.fetchval(
            f"""
            SELECT COUNT(DISTINCT id) FROM train_neo
            {where_clause.replace('is_potentially_hazardous', 'is_potentially_hazardous').replace('is_sentry_object', 'is_sentry_object')}
            """
        )
    return {"total": total, "page": page, "limit": limit, "data": [dict(r) for r in rows]}


# ──────────────────────────────────────────────
# GET /api/asteroids/{id}  (full detail + history)
# ──────────────────────────────────────────────
@router.get("/asteroids/{asteroid_id}")
async def get_asteroid(asteroid_id: int, pool=Depends(get_pool)):
    """Single asteroid: latest approach data + full close-approach history + prediction + risk."""
    async with pool.acquire() as conn:
        # Main data (latest approach for this asteroid)
        main = await conn.fetchrow(
            """
            SELECT DISTINCT ON (id)
                id, name, absolute_magnitude_h,
                min_diameter_km, max_diameter_km,
                is_potentially_hazardous, is_sentry_object, nasa_jpl_url
            FROM train_neo
            WHERE id = $1
            ORDER BY id, epoch_date_close_approach DESC
            """,
            asteroid_id,
        )
        if not main:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Asteroid not found")

        # All close approaches (timeline)
        approaches = await conn.fetch(
            """
            SELECT
                close_approach_date, close_approach_date_full,
                epoch_date_close_approach, relative_velocity_kps,
                miss_distance_km, orbiting_body
            FROM train_neo
            WHERE id = $1
            ORDER BY epoch_date_close_approach DESC
            """,
            asteroid_id,
        )

        # Prediction (if exists)
        pred = await conn.fetchrow(
            """
            SELECT
                probability_being_truely_hazardous,
                probability_being_falsely_hazardous,
                model_prediction
            FROM prediction_table
            WHERE id = $1
            LIMIT 1
            """,
            str(asteroid_id),
        )

        # Risk analysis (if exists)
        risk = await conn.fetchrow(
            """
            SELECT
                "RiskScorenormManual", "RiskCategoryManual",
                "RiskScorenormDDriven", "RiskCategoryDDriven",
                davg_m, mass_kg, impact_energy_j
            FROM risk_analysis
            WHERE id = $1
            LIMIT 1
            """,
            asteroid_id,
        )

    result = dict(main)
    result["approaches"] = [dict(a) for a in approaches]
    if pred:
        result.update({
            "probability_being_truely_hazardous": pred["probability_being_truely_hazardous"],
            "probability_being_falsely_hazardous": pred["probability_being_falsely_hazardous"],
            "model_prediction": pred["model_prediction"],
        })
    if risk:
        result.update({
            "risk_score_manual": risk["RiskScorenormManual"],
            "risk_category_manual": risk["RiskCategoryManual"],
            "risk_score_dd": risk["RiskScorenormDDriven"],
            "risk_category_dd": risk["RiskCategoryDDriven"],
            "davg_m": risk["davg_m"],
            "mass_kg": risk["mass_kg"],
            "impact_energy_j": risk["impact_energy_j"],
        })
    return result


# ──────────────────────────────────────────────
# GET /api/leaderboard
# ──────────────────────────────────────────────
_SORT_OPTIONS = {
    "risk":     ("r.\"RiskScorenormManual\"", "DESC NULLS LAST"),
    "velocity": ("t.relative_velocity_kps",   "DESC NULLS LAST"),
    "closest":  ("t.miss_distance_km",        "ASC NULLS LAST"),
    "size":     ("t.max_diameter_km",         "DESC NULLS LAST"),
}


@router.get("/leaderboard")
async def get_leaderboard(
    by: str = Query("risk", enum=["risk", "velocity", "closest", "size"]),
    top: int = Query(100, ge=10, le=1000),
    hazardous: bool = Query(None),
    sentry: bool = Query(None),
    name: str = Query(None, description="Search by asteroid name"),
    pool=Depends(get_pool),
):
    sort_col, sort_dir = _SORT_OPTIONS[by]
    pho_filter = ""
    if hazardous is not None:
        pho_filter += f" AND t.is_potentially_hazardous = {str(hazardous).lower()}"
    if sentry is not None:
        pho_filter += f" AND t.is_sentry_object = {str(sentry).lower()}"
    if name:
        safe_name = name.replace("'", "''").strip()
        pho_filter += f" AND t.name ILIKE '%{safe_name}%'"

    query = f"""
        WITH latest AS (
            SELECT DISTINCT ON (id)
                id, name, min_diameter_km, max_diameter_km,
                relative_velocity_kps, miss_distance_km,
                is_potentially_hazardous, is_sentry_object
            FROM train_neo
            ORDER BY id, epoch_date_close_approach DESC
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY {sort_col} {sort_dir}) AS rank,
            t.id, t.name, t.min_diameter_km, t.max_diameter_km,
            t.relative_velocity_kps, t.miss_distance_km,
            t.is_potentially_hazardous, t.is_sentry_object,
            r."RiskScorenormManual" AS risk_score_manual,
            r."RiskCategoryManual"  AS risk_category_manual
        FROM latest t
        LEFT JOIN risk_analysis r ON r.id = t.id
        WHERE 1=1 {pho_filter}
        ORDER BY {sort_col} {sort_dir}
        LIMIT {top}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# GET /api/compare
# ──────────────────────────────────────────────
@router.get("/compare")
async def compare_asteroids(
    ids: str = Query(..., description="Comma-separated asteroid IDs, max 6"),
    pool=Depends(get_pool),
):
    id_list = [int(i.strip()) for i in ids.split(",") if i.strip()][:6]
    results = []
    async with pool.acquire() as conn:
        for aid in id_list:
            row = await conn.fetchrow(
                """
                SELECT DISTINCT ON (id)
                    id, name, min_diameter_km, max_diameter_km,
                    relative_velocity_kps, miss_distance_km,
                    is_potentially_hazardous, is_sentry_object
                FROM train_neo
                WHERE id = $1
                ORDER BY id, epoch_date_close_approach DESC
                """,
                aid,
            )
            risk = await conn.fetchrow(
                """
                SELECT
                    "RiskScorenormManual", "RiskCategoryManual",
                    davg_m, mass_kg, impact_energy_j, "Phazardous"
                FROM risk_analysis WHERE id = $1 LIMIT 1
                """,
                aid,
            )
            pred = await conn.fetchrow(
                "SELECT probability_being_truely_hazardous FROM prediction_table WHERE id = $1 LIMIT 1",
                str(aid),
            )
            if row:
                entry = dict(row)
                if risk:
                    entry.update({
                        "risk_score_manual": risk["RiskScorenormManual"],
                        "risk_category_manual": risk["RiskCategoryManual"],
                        "davg_m": risk["davg_m"],
                        "mass_kg": risk["mass_kg"],
                        "impact_energy_j": risk["impact_energy_j"],
                        "phazardous": risk["Phazardous"],
                    })
                if pred:
                    entry["probability_being_truely_hazardous"] = pred["probability_being_truely_hazardous"]
                results.append(entry)
    return results
