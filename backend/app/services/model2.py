# REPLACES app/services/model2.py in full.
#
# What changed: _solve() and solve_model2() now accept four optional
# assumption parameters. Each is a bounded multiplier/buffer on a
# constraint that already exists in the LP -- nothing new is invented.
# Calling solve_model2() with no arguments reproduces the exact original
# behaviour (all defaults are the identity value: buffer=0, multiplier=1).
#
#   demand_buffer_pct    -- inflate each unit's demand_beds by this % before
#                            the Demand_{u} constraint. "What if occupancy
#                            runs X% hotter/colder than the last 6 months?"
#   budget_multiplier     -- scale the Budget_Ceiling constraint.
#                            "What if the fiscal-year budget were cut/raised
#                            by X%?"
#   nurse_ratio_multiplier / doctor_ratio_multiplier
#                          -- scale the per-bed staffing-ratio floor
#                            (Nurse_Ratio_{u} / Doctor_Ratio_{u}).
#                            "What if the minimum safe staffing ratio were
#                            raised/lowered by X%?"

import numpy as np
import pandas as pd
import pulp
from sqlalchemy import text

from app.celery_app import celery_app
from app.db import engine


def _load_inputs() -> pd.DataFrame:
    with engine.connect() as conn:
        beds = pd.read_sql(text("SELECT * FROM beds"), conn)
        staff = pd.read_sql(text("SELECT * FROM staff"), conn)
        budget = pd.read_sql(text("SELECT * FROM budget_lines"), conn)
        census = pd.read_sql(text("SELECT * FROM daily_census"), conn)

    census["census_date"] = pd.to_datetime(census["census_date"])
    recent_cutoff = census["census_date"].max() - pd.DateOffset(months=6)
    recent_census = census[census["census_date"] > recent_cutoff]

    census_summary = recent_census.groupby("unit_id").agg(
        avg_occupied_beds=("occupied_beds", "mean"),
        avg_nurses_on_duty=("nurses_on_duty", "mean"),
        avg_doctors_on_duty=("doctors_on_duty", "mean"),
    )

    bed_cost = beds.groupby("unit_id")["daily_cost_ngn"].mean().rename("avg_daily_bed_cost")

    nurse_cost = (
        staff[staff["role"] == "Psychiatric Nurse"]
        .groupby("home_unit_id")["monthly_cost_ngn"].mean()
    )
    doctor_cost = (
        staff[staff["role"].str.contains("Psychiatrist|Registrar", case=False, na=False)]
        .groupby("home_unit_id")["monthly_cost_ngn"].mean()
    )

    inputs = pd.DataFrame({
        "demand_beds": census_summary["avg_occupied_beds"],
        "nurse_per_bed": census_summary["avg_nurses_on_duty"] / census_summary["avg_occupied_beds"],
        "doctor_per_bed": census_summary["avg_doctors_on_duty"] / census_summary["avg_occupied_beds"],
    })
    inputs = inputs.join(bed_cost)
    inputs["nurse_monthly_cost"] = nurse_cost
    inputs["doctor_monthly_cost"] = doctor_cost

    total_bed_stock = int(beds["bed_id"].nunique())
    latest_year = int(budget["fiscal_year"].max())
    total_budget = float(
        budget.loc[budget["fiscal_year"] == latest_year, "amount_allocated_ngn"].sum()
    )
    current_beds = beds.groupby("unit_id")["bed_id"].count().rename("current_beds")

    return inputs, total_bed_stock, total_budget, current_beds, latest_year


def _solve(
    inputs: pd.DataFrame,
    total_bed_stock: int,
    total_budget: float,
    demand_buffer_pct: float = 0.0,
    nurse_ratio_multiplier: float = 1.0,
    doctor_ratio_multiplier: float = 1.0,
):
    units = inputs.index.tolist()
    prob = pulp.LpProblem("FNPH_Bed_Staff_Configuration", pulp.LpMinimize)

    beds_var = {u: pulp.LpVariable(f"beds_{u}", lowBound=0, cat="Integer") for u in units}
    nurses_var = {u: pulp.LpVariable(f"nurses_{u}", lowBound=0, cat="Integer") for u in units}
    doctors_var = {u: pulp.LpVariable(f"doctors_{u}", lowBound=0, cat="Integer") for u in units}

    prob += pulp.lpSum([
        beds_var[u] * inputs.loc[u, "avg_daily_bed_cost"] * 365
        + nurses_var[u] * inputs.loc[u, "nurse_monthly_cost"] * 12
        + doctors_var[u] * inputs.loc[u, "doctor_monthly_cost"] * 12
        for u in units
    ]), "Total_Annual_Cost"

    buffer_mult = 1.0 + (demand_buffer_pct / 100.0)
    for u in units:
        buffered_demand = inputs.loc[u, "demand_beds"] * buffer_mult
        prob += beds_var[u] >= np.ceil(buffered_demand), f"Demand_{u}"

    prob += pulp.lpSum([beds_var[u] for u in units]) <= total_bed_stock, "Total_Bed_Stock"

    for u in units:
        prob += nurses_var[u] >= beds_var[u] * inputs.loc[u, "nurse_per_bed"] * nurse_ratio_multiplier, f"Nurse_Ratio_{u}"
        prob += doctors_var[u] >= beds_var[u] * inputs.loc[u, "doctor_per_bed"] * doctor_ratio_multiplier, f"Doctor_Ratio_{u}"

    prob += pulp.lpSum([
        beds_var[u] * inputs.loc[u, "avg_daily_bed_cost"] * 365
        + nurses_var[u] * inputs.loc[u, "nurse_monthly_cost"] * 12
        + doctors_var[u] * inputs.loc[u, "doctor_monthly_cost"] * 12
        for u in units
    ]) <= total_budget, "Budget_Ceiling"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, beds_var, nurses_var, doctors_var


@celery_app.task(name="model2.solve")
def solve_model2(
    demand_buffer_pct: float = 0.0,
    budget_multiplier: float = 1.0,
    nurse_ratio_multiplier: float = 1.0,
    doctor_ratio_multiplier: float = 1.0,
) -> dict:
    inputs, total_bed_stock, total_budget, current_beds, latest_year = _load_inputs()
    adjusted_budget = total_budget * budget_multiplier

    prob, beds_var, nurses_var, doctors_var = _solve(
        inputs, total_bed_stock, adjusted_budget,
        demand_buffer_pct=demand_buffer_pct,
        nurse_ratio_multiplier=nurse_ratio_multiplier,
        doctor_ratio_multiplier=doctor_ratio_multiplier,
    )

    status = pulp.LpStatus[prob.status]
    assumptions = {
        "demand_buffer_pct": demand_buffer_pct,
        "budget_multiplier": budget_multiplier,
        "nurse_ratio_multiplier": nurse_ratio_multiplier,
        "doctor_ratio_multiplier": doctor_ratio_multiplier,
        "is_baseline": (
            demand_buffer_pct == 0.0 and budget_multiplier == 1.0
            and nurse_ratio_multiplier == 1.0 and doctor_ratio_multiplier == 1.0
        ),
    }

    if status != "Optimal":
        return {
            "status": status,
            "assumptions": assumptions,
            "message": "No feasible configuration exists under these assumptions "
                       "(budget/bed stock/ratios) -- this is itself a real finding, "
                       "not an error, and should be reported as such.",
        }

    units = inputs.index.tolist()
    results = pd.DataFrame({
        "recommended_beds": {u: int(beds_var[u].value()) for u in units},
        "recommended_nurses": {u: int(nurses_var[u].value()) for u in units},
        "recommended_doctors": {u: int(doctors_var[u].value()) for u in units},
    }).join(current_beds)
    results["bed_change"] = results["recommended_beds"] - results["current_beds"]
    results["pct_change"] = (results["bed_change"] / results["current_beds"] * 100).round(1)

    optimal_cost = pulp.value(prob.objective)
    current_cost = float((current_beds * inputs["avg_daily_bed_cost"] * 365).sum())

    return {
        "status": status,
        "assumptions": assumptions,
        "fiscal_year": latest_year,
        "total_bed_stock": total_bed_stock,
        "total_budget_ngn": adjusted_budget,
        "optimal_annual_cost_ngn": round(optimal_cost, 0),
        "current_annual_cost_ngn": round(current_cost, 0),
        "annual_savings_ngn": round(current_cost - optimal_cost, 0),
        "annual_savings_pct": round((current_cost - optimal_cost) / current_cost * 100, 1),
        "by_unit": results.reset_index().rename(columns={"index": "unit_id"}).to_dict(orient="records"),
    }