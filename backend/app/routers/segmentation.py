from fastapi import APIRouter, HTTPException

from app.services import segmentation

router = APIRouter(prefix="/api/v1/segmentation", tags=["segmentation"])


@router.get("/summary")
def get_cluster_summary():
    """Powers a segmentation overview screen: size, share, and
    human-readable interpretation of each of the 5 patient clusters."""
    try:
        return segmentation.cluster_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/patient/{patient_id}")
def get_patient_cluster(patient_id: str):
    """Single-patient lookup: which cluster a patient belongs to."""
    try:
        scored = segmentation.score_patients(patient_ids=[patient_id])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if scored.empty:
        raise HTTPException(status_code=404, detail=f"patient_id {patient_id} not found")
    return scored.iloc[0].to_dict()
