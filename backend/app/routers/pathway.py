from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.services.pathway import compute_pathway_summary, PathwayFilters, SEGMENT_CODES

router = APIRouter(prefix="/api/v1/pathway", tags=["pathway"])


@router.get("/summary")
def get_pathway_summary(
    segment_code: Optional[int] = Query(None, ge=0, le=4, description="0-4, matches patients.segment_code"),
    family_support_tier: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    family_support_min: Optional[int] = Query(None, ge=0, le=10),
    family_support_max: Optional[int] = Query(None, ge=0, le=10),
    zone_id: Optional[str] = Query(None, description="e.g. 'Z02'"),
    diagnosis_group: Optional[str] = Query(None, description="e.g. 'Psychotic'"),
    diagnosis_id: Optional[str] = Query(None, description="e.g. 'F20' -- specific diagnosis, overrides diagnosis_group if both given"),
):
    """Powers the Pathway Explorer screen: transition matrix, fundamental
    matrix, expected months to absorption, and absorption probabilities --
    recomputed from only the matching transitions when any filter is given."""
    filters = PathwayFilters(
        segment_code=segment_code,
        family_support_tier=family_support_tier,
        family_support_min=family_support_min,
        family_support_max=family_support_max,
        zone_id=zone_id,
        diagnosis_group=diagnosis_group,
        diagnosis_id=diagnosis_id,
    )
    try:
        return compute_pathway_summary(filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/filter-options")
def get_filter_options():
    """Lets the frontend populate its filter dropdowns from real values
    rather than hardcoding them -- segment labels, zones, and diagnosis
    groups can all be added to or renamed in the data over time."""
    from sqlalchemy import text
    from app.db import engine
    with engine.connect() as conn:
        segments = conn.execute(text(
            "SELECT DISTINCT segment_code, segment_label FROM patients ORDER BY segment_code"
        )).fetchall()
        zones = conn.execute(text(
            "SELECT zone_id, zone_name FROM zones ORDER BY zone_name"
        )).fetchall()
        diag_groups = conn.execute(text(
            "SELECT DISTINCT diagnosis_group FROM diagnoses ORDER BY diagnosis_group"
        )).fetchall()
        diagnoses = conn.execute(text(
            "SELECT diagnosis_id, diagnosis_name, diagnosis_group FROM diagnoses ORDER BY diagnosis_name"
        )).fetchall()
    return {
        "segments": [{"code": r[0], "label": r[1]} for r in segments],
        "family_support_tiers": [
            {"value": "low", "label": "Low (0-3)"},
            {"value": "medium", "label": "Medium (4-7)"},
            {"value": "high", "label": "High (8-10)"},
        ],
        "zones": [{"id": r[0], "name": r[1]} for r in zones],
        "diagnosis_groups": [r[0] for r in diag_groups],
        "diagnoses": [{"id": r[0], "name": r[1], "group": r[2]} for r in diagnoses],
    }