from celery.result import AsyncResult
from fastapi import APIRouter

from app.celery_app import celery_app
from app.services.roster import solve_roster

router = APIRouter(prefix="/api/v1/roster", tags=["roster"])


@router.post("/solve")
def trigger_solve(night_cap_bonus: int = 0, compare_relaxed: bool = False):
    """night_cap_bonus: test relaxing every staff member's night-shift
    cap by this many extra shifts. compare_relaxed: also solve a +2
    comparison and report the improvement (Week 4 finding: ~0.6%,
    confirming headcount, not policy, is the binding constraint)."""
    task = solve_roster.delay(night_cap_bonus=night_cap_bonus, compare_relaxed=compare_relaxed)
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
