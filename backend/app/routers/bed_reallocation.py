from celery.result import AsyncResult
from fastapi import APIRouter

from app.celery_app import celery_app
from app.services.bed_reallocation import solve_bed_reallocation

router = APIRouter(prefix="/api/v1/bed-reallocation", tags=["bed-reallocation"])


@router.post("/solve")
def trigger_solve():
    task = solve_bed_reallocation.delay()
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
