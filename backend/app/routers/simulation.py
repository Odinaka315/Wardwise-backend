from celery.result import AsyncResult
from fastapi import APIRouter
from typing import Optional, Dict, Literal
from app.celery_app import celery_app
from pydantic import BaseModel, Field
from app.services.simulation import run_scenarios, strike_trajectory, run_custom, baseline_defaults_payload


router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])


class EventSpec(BaseModel):
    type: Literal["none", "strike", "surge", "freeze"] = "none"
    start_month: int = 0
    duration_months: int = 0
    arrival_factor: Optional[float] = None
    discharge_factor: Optional[float] = None


class CustomScenarioRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=60)
    capacity_overrides: Optional[Dict[str, int]] = None
    demand_multiplier: Optional[float] = Field(None, ge=0.5, le=2.0)
    event: Optional[EventSpec] = None
    overcrowd_dropout: Optional[float] = Field(None, ge=0.0, le=1.0)
    horizon_months: Optional[int] = Field(None, ge=12, le=300)
    replications: Optional[int] = Field(None, ge=20, le=100)


@router.get("/baseline-defaults")
def get_baseline_defaults():
    return baseline_defaults_payload()


@router.post("/custom")
def trigger_custom(payload: CustomScenarioRequest):
    task = run_custom.delay(payload.model_dump())
    return {"task_id": task.id, "status": "submitted"}
@router.post("/solve")
def trigger_scenarios():
    task = run_scenarios.delay()
    return {"task_id": task.id, "status": "submitted"}


@router.post("/strike-trajectory")
def trigger_strike_trajectory(n_seeds: int = 40):
    task = strike_trajectory.delay(n_seeds=n_seeds)
    return {"task_id": task.id, "status": "submitted"}


@router.get("/result/{task_id}")
def get_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response