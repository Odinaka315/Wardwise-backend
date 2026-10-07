from fastapi import APIRouter, HTTPException
from app.services.overview import get_units_baseline

router = APIRouter(prefix="/api/v1/overview", tags=["overview"])

@router.get("")
def get_overview():
    """Returns baseline metrics across all hospital units:
    capacity, occupancy, stay, staffing, and locked status."""
    try:
        return get_units_baseline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
