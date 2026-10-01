from celery.result import AsyncResult
from fastapi import APIRouter

from app.celery_app import celery_app
from app.services.model2 import solve_model2

router = APIRouter(prefix="/api/v1/model2", tags=["model2"])


@router.post("/solve")
def trigger_solve():
    """Starts the bed & staffing optimisation as a background job and
    returns immediately with a task_id -- the actual solve never runs
    inside this request, per the brief's rule that long work must not
    block the browser."""
    task = solve_model2.delay()
    return {"task_id": task.id, "status": "submitted"}


@router.get("/result/{task_id}")
def get_solve_result(task_id: str):
    """Poll this with the task_id from /solve. status will be PENDING
    or STARTED while the worker is still solving, SUCCESS once the
    full board-facing comparison is ready in `result`, or FAILURE if
    the solve itself raised an error."""
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response
