from fastapi import APIRouter, HTTPException
from app.services.longstay import get_longstay_residents

router = APIRouter(prefix="/api/v1/longstay", tags=["longstay"])

@router.get("/summary")
def get_longstay_summary():
    """Returns the long-stay resident register with clinical and
    resettlement feasibility details."""
    try:
        return get_longstay_residents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
