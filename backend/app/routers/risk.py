from fastapi import APIRouter, HTTPException, Query

from app.services import risk

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.get("/readmission/worklist")
def get_readmission_worklist(top_pct: float = Query(10.0, gt=0, le=100)):
    """Powers the risk worklist screen: top N% of admissions by
    predicted 90-day readmission risk, plus the lift over the
    hospital-wide base rate that proves the list is worth acting on."""
    try:
        return risk.readmission_worklist(top_pct=top_pct)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/readmission/{admission_id}")
def get_readmission_score(admission_id: str):
    """Single-patient lookup: readmission risk score for one admission."""
    try:
        scored = risk.score_readmission_risk(admission_ids=[admission_id])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if scored.empty:
        raise HTTPException(status_code=404, detail=f"admission_id {admission_id} not found")
    return scored.iloc[0].to_dict()


@router.get("/nonattendance/worklist")
def get_nonattendance_worklist(top_pct: float = 10.0):
    """Top N% of upcoming visits by predicted non-attendance risk,
    plus the lift over the hospital-wide base rate."""
    try:
        return risk.nonattendance_worklist(top_pct=top_pct)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/nonattendance/{visit_id}")
def get_nonattendance_score(visit_id: str):
    """Single-visit lookup: non-attendance risk score for one
    scheduled outpatient visit."""
    try:
        scored = risk.score_nonattendance_risk(visit_ids=[visit_id])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if scored.empty:
        raise HTTPException(status_code=404, detail=f"visit_id {visit_id} not found")
    return scored.iloc[0].to_dict()
