from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel
from app.celery_app import celery_app
from app.services.roster import solve_roster

router = APIRouter(prefix="/api/v1/roster", tags=["roster"])
 
 
class RosterAssumptions(BaseModel):
    night_cap_bonus: int = 0
    compare_relaxed: bool = False
 
 
@router.post("/solve")
def trigger_roster_solve(assumptions: RosterAssumptions = RosterAssumptions()):
    task = solve_roster.delay(**assumptions.model_dump())
    return {"task_id": task.id, "status": "submitted", "assumptions": assumptions.model_dump()}
 
 
@router.get("/result/{task_id}")
def get_roster_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response