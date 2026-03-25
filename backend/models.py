"""
models.py — Pydantic v2 response schemas for the NASA NEO API
"""
from pydantic import BaseModel, Field
from typing import Optional


class AsteroidBasic(BaseModel):
    """Minimal asteroid data for 3D scene streaming."""
    id: int
    name: str
    min_diameter_km: float
    max_diameter_km: float
    relative_velocity_kps: float
    miss_distance_km: float
    is_potentially_hazardous: bool
    is_sentry_object: bool
    close_approach_date: str
    epoch_date_close_approach: int


class CloseApproach(BaseModel):
    """Single close-approach event for the timeline."""
    close_approach_date: str
    close_approach_date_full: Optional[str]
    epoch_date_close_approach: int
    relative_velocity_kps: float
    miss_distance_km: float
    orbiting_body: Optional[str]


class AsteroidDetail(BaseModel):
    """Full asteroid card with all history."""
    id: int
    name: str
    absolute_magnitude_h: Optional[float]
    min_diameter_km: float
    max_diameter_km: float
    is_potentially_hazardous: bool
    is_sentry_object: bool
    nasa_jpl_url: Optional[str]
    approaches: list[CloseApproach] = []
    # Joined from prediction_table
    probability_being_truely_hazardous: Optional[float] = None
    probability_being_falsely_hazardous: Optional[float] = None
    model_prediction: Optional[bool] = None
    # Joined from risk_analysis
    risk_score_manual: Optional[float] = None
    risk_category_manual: Optional[str] = None
    risk_score_dd: Optional[float] = None
    risk_category_dd: Optional[str] = None
    davg_m: Optional[float] = None
    mass_kg: Optional[float] = None
    impact_energy_j: Optional[float] = None


class RiskEntry(BaseModel):
    id: int
    name: Optional[str] = None
    davg_m: float
    mass_kg: float
    impact_energy_j: float
    enorm: float = Field(alias="Enorm")
    rmoid_norm: float = Field(alias="Rmoid_norm")
    phazardous: float = Field(alias="Phazardous")
    risk_score_raw: float = Field(alias="RiskScore_raw")
    risk_score_manual: float = Field(alias="RiskScorenormManual")
    risk_category_manual: str = Field(alias="RiskCategoryManual")
    risk_score_dd: float = Field(alias="RiskScorenormDDriven")
    risk_category_dd: str = Field(alias="RiskCategoryDDriven")
    risk_analysed_on_date: Optional[str] = None
    is_potentially_hazardous: Optional[bool] = None
    is_sentry_object: Optional[bool] = None

    model_config = {"populate_by_name": True}


class ModelLog(BaseModel):
    model_name: str
    training_date: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    run_id: str
    artifact_uri: Optional[str] = None


class DriftEntry(BaseModel):
    run_id: str
    feature: str
    feature_type: str
    method: str
    psi: Optional[float]
    chi2_stat: Optional[float]
    chi2_p_value: Optional[float]
    drift_status: str
    baseline_count: int
    new_count: int
    created_at: str


class StatsResponse(BaseModel):
    unique_asteroids: int
    total_approaches: int
    unique_pho: int
    unique_sentry: int
    min_miss_distance_km: float
    max_velocity_kps: float
    last_approach_date: str


class LeaderboardEntry(BaseModel):
    rank: int
    id: int
    name: str
    min_diameter_km: float
    max_diameter_km: float
    relative_velocity_kps: float
    miss_distance_km: float
    is_potentially_hazardous: bool
    is_sentry_object: bool
    # risk fields (may be None if not in risk_analysis)
    risk_score_manual: Optional[float] = None
    risk_category_manual: Optional[str] = None
