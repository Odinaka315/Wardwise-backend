from celery.result import AsyncResult
from fastapi import APIRouter

from app.celery_app import celery_app
from app.services.simulation import run_scenarios, strike_trajectory

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])


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