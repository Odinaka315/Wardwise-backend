"""
Compulsory budget goal-programming model: reallocates the latest
fiscal year's already-decided total budget ceiling across 88
unit/cost_category goals, in strict tier order (1 = highest priority),
weighted by strategic_priority_weight within each tier.

Includes the Week 4 fix: a hard floor on Personnel lines (70% of
current year-to-date spend) regardless of tier, after an initial
naive strict-lock version produced a mathematically optimal but
operationally impossible 0%-funded Personnel line at one unit. Tier
locks use a 2% tolerance rather than an exact lock, for the same
reason -- strict locking leaves the solver no incentive to protect
anything outside the tier currently being optimised.
"""

import pandas as pd
import pulp
from sqlalchemy import text

from app.celery_app import celery_app
from app.db import engine

PERSONNEL_FLOOR_FRACTION = 0.70
TIER_TOLERANCE = 1.02


def _load_inputs():
    with engine.connect() as conn:
        budget = pd.read_sql(text("SELECT * FROM budget_lines"), conn)
    latest_year = int(budget["fiscal_year"].max())
    goals = budget[budget["fiscal_year"] == latest_year].reset_index(drop=True)
    total_budget_ceiling = float(goals["amount_allocated_ngn"].sum())
    return goals, total_budget_ceiling, latest_year


@celery_app.task(name="budget.solve")
def solve_budget() -> dict:
    goals, total_budget_ceiling, latest_year = _load_inputs()
    n = len(goals)

    prob = pulp.LpProblem("Budget_Goal_Programming", pulp.LpMinimize)
    alloc = {i: pulp.LpVariable(f"alloc_{i}", lowBound=0) for i in range(n)}
    under = {i: pulp.LpVariable(f"under_{i}", lowBound=0) for i in range(n)}

    for i in range(n):
        prob += alloc[i] + under[i] == goals.loc[i, "amount_requested_ngn"]
        if goals.loc[i, "cost_category"] == "Personnel":
            floor = goals.loc[i, "amount_spent_ngn"] * PERSONNEL_FLOOR_FRACTION
            prob += alloc[i] >= floor, f"PersonnelFloor_{i}"

    prob += pulp.lpSum([alloc[i] for i in range(n)]) <= total_budget_ceiling

    tier_idx = {t: goals[goals["goal_programming_tier"] == t].index for t in sorted(goals["goal_programming_tier"].unique())}

    stage_results = {}
    prev_tier_obj = None
    for tier in sorted(tier_idx.keys()):
        idx = tier_idx[tier]
        if prev_tier_obj is not None:
            prob += pulp.lpSum(
                [under[i] * goals.loc[i, "strategic_priority_weight"] for i in prev_idx]
            ) <= prev_tier_obj * TIER_TOLERANCE + 1
        prob.setObjective(pulp.lpSum([under[i] * goals.loc[i, "strategic_priority_weight"] for i in idx]))
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        prev_tier_obj = pulp.value(prob.objective)
        prev_idx = idx
        stage_results[int(tier)] = {"status": pulp.LpStatus[prob.status], "weighted_shortfall": round(prev_tier_obj, 0)}

    goals["final_allocation"] = pd.Series({i: alloc[i].value() for i in range(n)})
    goals["pct_funded"] = (goals["final_allocation"] / goals["amount_requested_ngn"] * 100).round(1)

    tier_summary = goals.groupby("goal_programming_tier").agg(
        total_requested=("amount_requested_ngn", "sum"),
        total_allocated=("final_allocation", "sum"),
        avg_pct_funded=("pct_funded", "mean"),
    ).round(1).reset_index().to_dict(orient="records")

    return {
        "fiscal_year": latest_year,
        "total_requested_ngn": float(goals["amount_requested_ngn"].sum()),
        "total_budget_ceiling_ngn": total_budget_ceiling,
        "structural_shortfall_pct": round(
            (1 - total_budget_ceiling / goals["amount_requested_ngn"].sum()) * 100, 1
        ),
        "stage_results": stage_results,
        "tier_summary": tier_summary,
        "by_line_item": goals[["unit_id", "cost_category", "goal_programming_tier",
                                "amount_requested_ngn", "final_allocation", "pct_funded"]]
        .to_dict(orient="records"),
    }
