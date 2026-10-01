from fastapi import FastAPI
from sqlalchemy import text

from app.db import engine
from app.routers import pathway
from app.routers import risk
from app.routers import forecast
from app.routers import segmentation
from app.routers import model2
from app.routers import bed_reallocation, roster, budget
from app.routers import simulation

app = FastAPI(
    title="FNPH Yaba — Model 2 API",
    description="Bed & staffing configuration decision support API.",
    version="0.1.0",
)

app.include_router(pathway.router)
app.include_router(risk.router)
app.include_router(forecast.router)
app.include_router(segmentation.router)
app.include_router(model2.router)
app.include_router(bed_reallocation.router)
app.include_router(roster.router)
app.include_router(budget.router)
app.include_router(simulation.router)

@app.get("/health")
def health():
    """Confirms the API is up AND can actually reach the database —
    a bare 'ok' from the process isn't enough to trust the skeleton."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        db_status = f"unreachable: {exc}"
    return {"status": "ok", "database": db_status}
