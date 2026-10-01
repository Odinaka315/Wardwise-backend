"""
Model 2: bed & staffing configuration optimisation.

Minimises total annual cost (beds + nurses + doctors, per unit)
subject to: meeting each unit's 6-month average observed occupancy
(from daily_census, not a forecast -- see MODELING_LOG.md for why),
not exceeding the hospital's real physical bed stock, maintaining
each unit's own currently-observed nurse/doctor-per-bed ratio as a
safety floor, and staying within the latest fiscal year's allocated
budget.

Unlike the Week 1-3 models, there is nothing to "train" here -- every
solve rebuilds its inputs fresh from the live database, so the
recommendation always reflects current data, not a stale snapshot.
"""

import numpy as np
import pandas as pd
import pulp
from sqlalchemy import text

from app.celery_app import celery_app
from app.db import engine


def _load_inputs() -> pd.DataFrame:
    """Rebuilds the full Model 2 input table (demand, costs, staffing
    ratios per unit) live from the database -- the same logic
    validated in Colab, kept in one place so a solve always reflects
    current data.
    """
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
    # Doctor-equivalent grades: Consultant Psychiatrist, Senior
    # Registrar, Registrar -- a plain "Psychiatrist" filter misses the
    # registrar grades and silently drops any unit staffed only by
    # registrars (caught during Week 4 validation).
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


def _solve(inputs: pd.DataFrame, total_bed_stock: int, total_budget: float):
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

    for u in units:
        prob += beds_var[u] >= np.ceil(inputs.loc[u, "demand_beds"]), f"Demand_{u}"

    prob += pulp.lpSum([beds_var[u] for u in units]) <= total_bed_stock, "Total_Bed_Stock"

    for u in units:
        prob += nurses_var[u] >= beds_var[u] * inputs.loc[u, "nurse_per_bed"], f"Nurse_Ratio_{u}"
        prob += doctors_var[u] >= beds_var[u] * inputs.loc[u, "doctor_per_bed"], f"Doctor_Ratio_{u}"

    prob += pulp.lpSum([
        beds_var[u] * inputs.loc[u, "avg_daily_bed_cost"] * 365
        + nurses_var[u] * inputs.loc[u, "nurse_monthly_cost"] * 12
        + doctors_var[u] * inputs.loc[u, "doctor_monthly_cost"] * 12
        for u in units
    ]) <= total_budget, "Budget_Ceiling"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, beds_var, nurses_var, doctors_var


@celery_app.task(name="model2.solve")
def solve_model2() -> dict:
    """The actual background job. Rebuilds inputs from the live
    database, solves, and returns a full board-facing comparison --
    recommended vs. current beds, staffing, and the cost delta.
    """
    inputs, total_bed_stock, total_budget, current_beds, latest_year = _load_inputs()
    prob, beds_var, nurses_var, doctors_var = _solve(inputs, total_bed_stock, total_budget)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {
            "status": status,
            "message": "No feasible configuration exists within the current "
                       "budget and bed stock -- this is itself a real finding, "
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
        "fiscal_year": latest_year,
        "total_bed_stock": total_bed_stock,
        "total_budget_ngn": total_budget,
        "optimal_annual_cost_ngn": round(optimal_cost, 0),
        "current_annual_cost_ngn": round(current_cost, 0),
        "annual_savings_ngn": round(current_cost - optimal_cost, 0),
        "annual_savings_pct": round((current_cost - optimal_cost) / current_cost * 100, 1),
        "by_unit": results.reset_index().rename(columns={"index": "unit_id"}).to_dict(orient="records"),
    }
