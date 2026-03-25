"""
routers/risk.py — Risk analysis endpoints (READ-ONLY)
"""
from fastapi import APIRouter, Depends, Query
from database import get_pool

router = APIRouter(tags=["Risk Analysis"])


@router.get("/risk")
async def list_risk(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    category: str = Query(None, description="Filter: High Risk / Medium Risk / Low Risk / Very Low Risk"),
    pool=Depends(get_pool),
):
    """Paginated risk analysis leaderboard with asteroid names joined."""
    offset = (page - 1) * limit
    cat_filter = f"AND r.\"RiskCategoryManual\" = '{category}'" if category else ""

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH latest_names AS (
                SELECT DISTINCT ON (id) id, name, is_potentially_hazardous, is_sentry_object
                FROM train_neo
                ORDER BY id, epoch_date_close_approach DESC
            )
            SELECT
                r.id, n.name, r.davg_m, r.mass_kg, r.impact_energy_j,
                r."Enorm", r."Rmoid_norm", r."Phazardous",
                r."RiskScore_raw",
                r."RiskScorenormManual", r."RiskCategoryManual",
                r."RiskScorenormDDriven", r."RiskCategoryDDriven",
                r.risk_analysed_on_date,
                n.is_potentially_hazardous, n.is_sentry_object
            FROM risk_analysis r
            LEFT JOIN latest_names n ON n.id = r.id
            WHERE 1=1 {cat_filter}
            ORDER BY r."RiskScorenormManual" DESC
            LIMIT {limit} OFFSET {offset}
            """
        )
        total = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM risk_analysis r
            WHERE 1=1 {cat_filter}
            """
        )
        # Category breakdown
        cats = await conn.fetch(
            """
            SELECT "RiskCategoryManual" AS category, COUNT(*) AS count
            FROM risk_analysis
            GROUP BY "RiskCategoryManual"
            ORDER BY count DESC
            """
        )

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "category_breakdown": [dict(c) for c in cats],
        "data": [dict(r) for r in rows],
    }


@router.get("/risk/{asteroid_id}")
async def get_risk(asteroid_id: int, pool=Depends(get_pool)):
    """Risk data for a single asteroid."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT r.*, n.name, n.is_potentially_hazardous, n.is_sentry_object
            FROM risk_analysis r
            LEFT JOIN (
                SELECT DISTINCT ON (id) id, name, is_potentially_hazardous, is_sentry_object
                FROM train_neo ORDER BY id, epoch_date_close_approach DESC
            ) n ON n.id = r.id
            WHERE r.id = $1
            LIMIT 1
            """,
            asteroid_id,
        )
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Risk data not found for this asteroid")
    return dict(row)
