from fastapi import APIRouter, HTTPException

from app.services import forecast

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


@router.get("")
def get_all_forecasts():
    """Powers the forecast screen: next month's predicted admissions
    for every unit at once, each using whichever method won for it
    (XGBoost or the 3-month trailing average)."""
    try:
        return forecast.forecast_all_units()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{unit_id}")
def get_unit_forecast(unit_id: str):
    """Next month's predicted admissions for a single unit."""
    try:
        return forecast.forecast_unit(unit_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
