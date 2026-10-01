import os
from celery import Celery

celery_app = Celery(
    "fnph_model2",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Africa/Lagos",
    enable_utc=True,
    # Long solves (bed reallocation MIP, rosters) must never block a
    # web request, so this worker is where they actually run.
)

# Tasks are auto-discovered from app/services when they use @celery_app.task
celery_app.autodiscover_tasks(["app.services"])

# autodiscover_tasks only looks for a file literally named tasks.py
# inside each listed package by default -- our tasks instead live
# directly in their own named service files, so each must be imported
# explicitly here or the worker process never loads them and rejects
# the task with an "unregistered task" error, even though the API
# container can see it fine (this bit us once already with model2).
from app.services import model2, bed_reallocation, roster, budget, simulation  # noqa: E402,F401
