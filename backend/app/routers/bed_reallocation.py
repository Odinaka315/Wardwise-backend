from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel
from app.celery_app import celery_app
from app.services.bed_reallocation import solve_bed_reallocation

router = APIRouter(prefix="/api/v1/bed-reallocation", tags=["bed-reallocation"])
 
 
class BedReallocationAssumptions(BaseModel):
    lookback_months: int = 6
    demand_buffer_pct: float = 0.0
 
 
@router.post("/solve")
def trigger_bed_reallocation_solve(assumptions: BedReallocationAssumptions = BedReallocationAssumptions()):
    task = solve_bed_reallocation.delay(**assumptions.model_dump())
    return {"task_id": task.id, "status": "submitted", "assumptions": assumptions.model_dump()}
 
 
@router.get("/result/{task_id}")
def get_bed_reallocation_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response