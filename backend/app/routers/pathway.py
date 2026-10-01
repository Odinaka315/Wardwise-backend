from fastapi import APIRouter, HTTPException

from app.services.pathway import compute_pathway_summary

router = APIRouter(prefix="/api/v1/pathway", tags=["pathway"])


@router.get("/summary")
def get_pathway_summary():
    """Powers the Pathway Explorer screen: transition matrix,
    fundamental matrix, expected months to absorption, and
    absorption probabilities.

    NOTE: this recomputes from the database on every call, which is
    fine at this data size. If you later add filtering by segment,
    zone or diagnosis (as the brief requires), this is the endpoint
    to extend with query parameters.
    """
    try:
        return compute_pathway_summary()
    except Exception as exc:
        # Once care_pathway_transitions is loaded and column names
        # confirmed, this should stop firing. Until then it tells you
        # exactly what's missing instead of a bare 500.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
