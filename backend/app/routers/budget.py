from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel
from app.celery_app import celery_app
from app.services.budget import solve_budget

router = APIRouter(prefix="/api/v1/budget", tags=["budget"])
 
 
class BudgetAssumptions(BaseModel):
    personnel_floor_fraction: float = 0.70
    tier_tolerance: float = 1.02
 
 
@router.post("/solve")
def trigger_budget_solve(assumptions: BudgetAssumptions = BudgetAssumptions()):
    task = solve_budget.delay(**assumptions.model_dump())
    return {"task_id": task.id, "status": "submitted", "assumptions": assumptions.model_dump()}
 
 
@router.get("/result/{task_id}")
def get_budget_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response