"""
Compulsory bed reallocation: LP (for shadow prices / sensitivity) then
MIP (for the actual integer recommendation), scoped to the flexible
bed pool only (DAY, DRU, LSR) -- the other 7 units are locked at the
unit level (beds.locked_flag), confirmed independently by
shift_requirements.requires_locked_ward_competency showing the exact
same 7 units. Reallocating those beds would not be physically
implementable, so they are excluded here entirely rather than left
to look like a valid recommendation.
"""

import numpy as np
import pandas as pd
import pulp
from sqlalchemy import text

from app.celery_app import celery_app
from app.db import engine

FLEXIBLE_UNITS = ["DAY", "DRU", "LSR"]


def _load_inputs():
    with engine.connect() as conn:
        beds = pd.read_sql(text("SELECT * FROM beds"), conn)
        census = pd.read_sql(text("SELECT * FROM daily_census"), conn)

    census["census_date"] = pd.to_datetime(census["census_date"])
    recent_cutoff = census["census_date"].max() - pd.DateOffset(months=6)
    recent = census[census["census_date"] > recent_cutoff]
    demand = recent.groupby("unit_id")["occupied_beds"].mean()

    bed_cost = beds.groupby("unit_id")["daily_cost_ngn"].mean()
    current_beds = beds.groupby("unit_id")["bed_id"].count()
    flexible_pool = int(beds[beds["unit_id"].isin(FLEXIBLE_UNITS)]["bed_id"].nunique())

    return demand, bed_cost, current_beds, flexible_pool


@celery_app.task(name="bed_reallocation.solve")
def solve_bed_reallocation() -> dict:
    demand, bed_cost, current_beds, flexible_pool = _load_inputs()

    # --- Stage 1: LP relaxation, for shadow prices ---
    lp = pulp.LpProblem("Bed_Reallocation_LP", pulp.LpMinimize)
    beds_lp = {u: pulp.LpVariable(f"beds_lp_{u}", lowBound=0, cat="Continuous") for u in FLEXIBLE_UNITS}
    lp += pulp.lpSum([beds_lp[u] * bed_cost[u] * 365 for u in FLEXIBLE_UNITS])

    demand_constraints = {}
    for u in FLEXIBLE_UNITS:
        demand_constraints[u] = beds_lp[u] >= np.ceil(demand[u])
        lp += demand_constraints[u], f"Demand_{u}"
    lp += pulp.lpSum([beds_lp[u] for u in FLEXIBLE_UNITS]) == flexible_pool, "Pool_Conservation"
    lp.solve(pulp.PULP_CBC_CMD(msg=0))

    shadow_prices = {
        name: {"shadow_price_ngn_per_year": round(c.pi, 1), "slack": round(c.slack, 2)}
        for name, c in lp.constraints.items()
    }

    # --- Stage 2: MIP, the actual implementable recommendation ---
    mip = pulp.LpProblem("Bed_Reallocation_MIP", pulp.LpMinimize)
    beds_mip = {u: pulp.LpVariable(f"beds_mip_{u}", lowBound=0, cat="Integer") for u in FLEXIBLE_UNITS}
    mip += pulp.lpSum([beds_mip[u] * bed_cost[u] * 365 for u in FLEXIBLE_UNITS])
    for u in FLEXIBLE_UNITS:
        mip += beds_mip[u] >= np.ceil(demand[u]), f"Demand_{u}"
    mip += pulp.lpSum([beds_mip[u] for u in FLEXIBLE_UNITS]) == flexible_pool, "Pool_Conservation"
    mip.solve(pulp.PULP_CBC_CMD(msg=0))

    by_unit = []
    for u in FLEXIBLE_UNITS:
        recommended = int(beds_mip[u].value())
        by_unit.append({
            "unit_id": u,
            "recommended_beds": recommended,
            "current_beds": int(current_beds[u]),
            "bed_change": recommended - int(current_beds[u]),
        })

    return {
        "lp_status": pulp.LpStatus[lp.status],
        "mip_status": pulp.LpStatus[mip.status],
        "flexible_pool_size": flexible_pool,
        "scope_note": "Recommendation covers only DAY, DRU, LSR -- the other "
                       "7 units have locked_flag=1 (structurally fixed beds, "
                       "confirmed via shift_requirements.requires_locked_ward_"
                       "competency) and are not reallocatable.",
        "by_unit": by_unit,
        "sensitivity": shadow_prices,
    }
