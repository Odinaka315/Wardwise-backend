"""
Compulsory 28-day staff roster: an integer program assigning nurses
and doctors to shifts, respecting availability/leave and each staff
member's night-shift cap, minimising total penalised understaffing
(not requiring 100% coverage as a hard constraint -- the dataset's
understaffing_penalty_weight column signals full coverage isn't
achievable with the current staff pool, confirmed during Week 4
validation).

night_cap_bonus lets a caller test the sensitivity finding live: how
much relaxing the night-shift cap actually helps (answer, from Week 4
validation: only ~0.6% -- the real bottleneck is headcount, not policy).
"""

import numpy as np
import pandas as pd
import pulp
from sqlalchemy import text

from app.celery_app import celery_app
from app.db import engine

NURSE_ROLES = ["Psychiatric Nurse"]
DOCTOR_ROLES = ["Consultant Psychiatrist", "Senior Registrar", "Registrar"]


def _load_inputs():
    with engine.connect() as conn:
        shift_req = pd.read_sql(text("SELECT * FROM shift_requirements"), conn)
        staff = pd.read_sql(text("SELECT * FROM staff"), conn)
        staff_avail = pd.read_sql(text("SELECT * FROM staff_availability"), conn)

    shift_req["roster_date"] = pd.to_datetime(shift_req["roster_date"])
    staff_avail["roster_date"] = pd.to_datetime(staff_avail["roster_date"])
    staff_avail["eligible"] = (staff_avail["available_flag"] == 1) & (staff_avail["requested_leave_flag"] == 0)

    roster_staff = staff[staff["role"].isin(NURSE_ROLES + DOCTOR_ROLES) & (staff["home_unit_id"] != "OPD")].copy()
    roster_staff["staff_type"] = np.where(roster_staff["role"].isin(NURSE_ROLES), "nurse", "doctor")

    return shift_req, roster_staff, staff_avail


def _build_and_solve(shift_req, roster_staff, staff_avail, night_cap_bonus: int):
    eligibility = staff_avail.set_index(["staff_id", "roster_date"])["eligible"].to_dict()
    staff_unit = roster_staff.set_index("staff_id")["home_unit_id"].to_dict()
    staff_type = roster_staff.set_index("staff_id")["staff_type"].to_dict()
    max_nights = roster_staff.set_index("staff_id")["max_night_shifts_per_month"].to_dict()

    dates = sorted(shift_req["roster_date"].unique())
    shifts = ["M", "A", "N"]
    staff_ids = roster_staff["staff_id"].tolist()
    req_lookup = shift_req.set_index(["unit_id", "roster_date", "shift_code"])

    suffix = f"b{night_cap_bonus}"
    x = {(s, d, sh): pulp.LpVariable(f"x_{s}_{d}_{sh}_{suffix}", cat="Binary")
         for s in staff_ids for d in dates for sh in shifts}
    shortfall = {}
    for key in req_lookup.index:
        shortfall[(key, "nurse")] = pulp.LpVariable(f"short_n_{key}_{suffix}", lowBound=0)
        shortfall[(key, "doctor")] = pulp.LpVariable(f"short_d_{key}_{suffix}", lowBound=0)

    prob = pulp.LpProblem(f"Roster_{suffix}", pulp.LpMinimize)
    prob += pulp.lpSum([
        (shortfall[(key, "nurse")] + shortfall[(key, "doctor")]) * req_lookup.loc[key, "understaffing_penalty_weight"]
        for key in req_lookup.index
    ])

    for key in req_lookup.index:
        u, d, sh = key
        req = req_lookup.loc[key]
        nurses_assigned = pulp.lpSum([x[(s, d, sh)] for s in staff_ids if staff_unit[s] == u and staff_type[s] == "nurse"])
        doctors_assigned = pulp.lpSum([x[(s, d, sh)] for s in staff_ids if staff_unit[s] == u and staff_type[s] == "doctor"])
        prob += nurses_assigned + shortfall[(key, "nurse")] >= req["nurses_required"]
        prob += doctors_assigned + shortfall[(key, "doctor")] >= req["doctors_required"]

    for s in staff_ids:
        for d in dates:
            prob += pulp.lpSum([x[(s, d, sh)] for sh in shifts]) <= 1
            if not eligibility.get((s, d), False):
                for sh in shifts:
                    prob += x[(s, d, sh)] == 0
        prob += pulp.lpSum([x[(s, d, "N")] for d in dates]) <= max_nights[s] + night_cap_bonus

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    shortfall_rows = []
    for key in req_lookup.index:
        u, d, sh = key
        n_val, d_val = shortfall[(key, "nurse")].value(), shortfall[(key, "doctor")].value()
        if n_val > 0 or d_val > 0:
            shortfall_rows.append({"unit_id": u, "nurse_shortfall": n_val, "doctor_shortfall": d_val})

    per_unit = (
        pd.DataFrame(shortfall_rows).groupby("unit_id")[["nurse_shortfall", "doctor_shortfall"]].sum().round(1)
        if shortfall_rows else pd.DataFrame(columns=["nurse_shortfall", "doctor_shortfall"])
    )
    night_at_cap = sum(
        1 for s in staff_ids
        if sum(x[(s, d, "N")].value() for d in dates) >= max_nights[s] + night_cap_bonus
    )

    return {
        "status": pulp.LpStatus[prob.status],
        "total_penalised_shortfall": round(pulp.value(prob.objective), 2),
        "total_unfilled_nurse_shifts": round(sum(r["nurse_shortfall"] for r in shortfall_rows), 0),
        "total_unfilled_doctor_shifts": round(sum(r["doctor_shortfall"] for r in shortfall_rows), 0),
        "staff_at_night_cap": night_at_cap,
        "total_roster_staff": len(staff_ids),
        "by_unit": per_unit.reset_index().to_dict(orient="records"),
    }


@celery_app.task(name="roster.solve")
def solve_roster(night_cap_bonus: int = 0, compare_relaxed: bool = False) -> dict:
    """night_cap_bonus: extra night shifts allowed per staff member,
    beyond their normal cap -- 0 reproduces the validated baseline.
    compare_relaxed: if True, also solves with +2 nights and reports
    the improvement, reproducing the Week 4 sensitivity finding
    (~0.6% -- night cap is not the real bottleneck, headcount is)."""
    shift_req, roster_staff, staff_avail = _load_inputs()
    baseline = _build_and_solve(shift_req, roster_staff, staff_avail, night_cap_bonus)

    if compare_relaxed:
        relaxed = _build_and_solve(shift_req, roster_staff, staff_avail, night_cap_bonus + 2)
        improvement = round(
            (1 - relaxed["total_penalised_shortfall"] / baseline["total_penalised_shortfall"]) * 100, 2
        )
        baseline["night_cap_sensitivity"] = {
            "relaxed_by": 2,
            "relaxed_total_penalised_shortfall": relaxed["total_penalised_shortfall"],
            "improvement_pct": improvement,
        }

    return baseline
