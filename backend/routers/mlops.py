"""
routers/mlops.py — MLOps monitoring endpoints (READ-ONLY)
"""
from fastapi import APIRouter, Depends
from database import get_pool

router = APIRouter(tags=["MLOps"])


@router.get("/mlops/models")
async def get_model_logs(pool=Depends(get_pool)):
    """All model training history."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT model_name, training_date, accuracy, precision, recall, f1_score, run_id, artifact_uri
            FROM model_training_logs
            ORDER BY training_date DESC
            """
        )
    return [dict(r) for r in rows]


@router.get("/mlops/drift")
async def get_drift_records(pool=Depends(get_pool)):
    """All drift detection records, grouped by run."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, feature, feature_type, method, psi,
                   chi2_stat, chi2_p_value, drift_status,
                   baseline_count, new_count,
                   created_at AT TIME ZONE 'UTC' AS created_at
            FROM neo_data_drift
            ORDER BY created_at DESC, feature
            """
        )
    return [dict(r) for r in rows]


@router.get("/mlops/drift/latest")
async def get_latest_drift(pool=Depends(get_pool)):
    """Latest drift detection run only."""
    async with pool.acquire() as conn:
        latest_run = await conn.fetchval(
            "SELECT run_id FROM neo_data_drift ORDER BY created_at DESC LIMIT 1"
        )
        if not latest_run:
            return {"run_id": None, "features": [], "summary": {}}

        rows = await conn.fetch(
            """
            SELECT run_id, feature, feature_type, method, psi,
                   chi2_stat, chi2_p_value, drift_status,
                   baseline_count, new_count,
                   created_at AT TIME ZONE 'UTC' AS created_at
            FROM neo_data_drift
            WHERE run_id = $1
            ORDER BY feature
            """,
            latest_run,
        )
        features = [dict(r) for r in rows]

        # Summary counts
        high = sum(1 for f in features if f["drift_status"] == "High Drift")
        moderate = sum(1 for f in features if f["drift_status"] == "Moderate Drift")
        no_drift = sum(1 for f in features if f["drift_status"] == "No Drift")

    return {
        "run_id": latest_run,
        "run_date": features[0]["created_at"] if features else None,
        "summary": {"high_drift": high, "moderate_drift": moderate, "no_drift": no_drift},
        "features": features,
    }
