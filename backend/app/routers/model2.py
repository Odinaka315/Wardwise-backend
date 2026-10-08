from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel
from app.celery_app import celery_app
from app.services.model2 import solve_model2

# ── app/api/v1/model2.py ────────────────────────────────────────────────
router = APIRouter(prefix="/api/v1/model2", tags=["model2"])
 
 
class Model2Assumptions(BaseModel):
    demand_buffer_pct: float = 0.0
    budget_multiplier: float = 1.0
    nurse_ratio_multiplier: float = 1.0
    doctor_ratio_multiplier: float = 1.0
 
 
@router.post("/solve")
def trigger_model2_solve(assumptions: Model2Assumptions = Model2Assumptions()):
    task = solve_model2.delay(**assumptions.model_dump())
    return {"task_id": task.id, "status": "submitted", "assumptions": assumptions.model_dump()}
 
 
@router.get("/result/{task_id}")
def get_model2_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response