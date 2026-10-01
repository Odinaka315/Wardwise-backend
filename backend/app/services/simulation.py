"""
Week 5 simulation service: builds the calibrated baseline config live
from the database, defines the 6 board scenarios, and runs them as a
background job. See MODELING_LOG.md for the mechanism behind each
scenario and the validation this baseline passed before being trusted
here (held-out absorption-mix test, independent overflow check against
crisis_unit_hourly).
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.celery_app import celery_app
from app.db import engine
from app.services.sim_engine import (
    SimConfig, calibrate_phi, modify_P, run_replications, simulate, analytic_headcount,
)

TRANSIENT = ["Acute inpatient", "Rehabilitation inpatient", "Day hospital",
             "Outpatient follow-up", "Community relapse"]
ABSORBING = ["Deceased", "Long stay institutional care", "Lost to follow-up",
             "Recovered and discharged from service", "Transferred to another facility"]
ALLST = TRANSIENT + ABSORBING
N_T = 5
ACUTE, REHAB, DAY, OPD, RELAPSE = 0, 1, 2, 3, 4

N_REPS = 30  # background job, no request timeout to worry about


def _build_baseline():
    with engine.connect() as conn:
        trans = pd.read_sql(text("SELECT * FROM care_pathway_transitions"), conn)
        census = pd.read_sql(text("SELECT * FROM daily_census"), conn)
        visits = pd.read_sql(text("SELECT * FROM outpatient_visits"), conn)
        budget = pd.read_sql(text("SELECT * FROM budget_lines"), conn)
        units = pd.read_sql(text("SELECT * FROM units"), conn)

    trans["period_date"] = pd.to_datetime(trans["period_date"])
    census["census_date"] = pd.to_datetime(census["census_date"])

    counts = (trans.groupby(["state_from", "state_to"]).size()
              .unstack(fill_value=0).reindex(index=ALLST, columns=ALLST, fill_value=0))
    for s in ABSORBING:
        if counts.loc[s].sum() == 0:
            counts.loc[s, s] = 1
    P = counts.div(counts.sum(axis=1), axis=0).to_numpy()

    first = trans[trans["month_index"] == 0]
    start_probs = (first["state_from"].value_counts().reindex(TRANSIENT, fill_value=0)
                   / len(first)).to_numpy()

    arrivals_m = first.groupby(first["period_date"].dt.to_period("M")).size()
    lam = arrivals_m[arrivals_m.index < "2024-02"].mean()

    census["ym"] = census["census_date"].dt.to_period("M")
    occ = census.pivot_table(index="ym", columns="unit_id", values="occupied_beds", aggfunc="mean")
    steady = occ.loc["2023-04":"2025-12"]
    census_means = {ACUTE: (steady["ACM"] + steady["ACF"]).mean(),
                     REHAB: (steady["DRU"] + steady["LSR"]).mean(),
                     DAY: steady["DAY"].mean()}

    noshow_rate = visits["did_not_attend_flag"].mean()
    drug_diesel_share = (budget.loc[budget["cost_category"].isin(
        ["Psychotropic drugs", "Utilities and diesel"]), "amount_spent_ngn"].sum()
        / budget["amount_spent_ngn"].sum())
    ltw_capacity = int(units.loc[units["unit_id"] == "LTW", "bed_capacity"].iloc[0])
    ltw_occupied = occ["LTW"].loc["2023-04":"2025-12"].mean()

    cfg_raw = SimConfig(
        P=P, start_probs=start_probs, arrivals_per_month=lam, seasonal_index=np.ones(12),
        phi={ACUTE: 1, REHAB: 1, DAY: 1},
        capacity={ACUTE: 115, REHAB: 85, DAY: 30},
        bed_cost_per_day={ACUTE: 41000, REHAB: 38000, DAY: 19000},
        community_cost_per_month={OPD: 25000, RELAPSE: 15000},
        start_month=0, horizon=200, warmup=24, entry_window=60,
    )
    baseline_cfg = calibrate_phi(cfg_raw, census_means)

    extras = dict(
        noshow_rate=float(noshow_rate),
        drug_diesel_share=float(drug_diesel_share),
        ltw_capacity=ltw_capacity,
        ltw_occupied=float(ltw_occupied),
        visits_total=int(len(visits)),
        avg_fare=float(visits["travel_fare_ngn"].mean()),
        lam=float(lam),
        P=P,
    )
    return baseline_cfg, extras


def _build_scenarios(baseline_cfg, extras):
    head = analytic_headcount(baseline_cfg)
    census_means = {g: baseline_cfg.phi[g] * head[g] for g in (ACUTE, REHAB, DAY)}

    scenarios = {}
    scenarios["1_demand_plus_12pct"] = SimConfig(**{**baseline_cfg.__dict__, "demand_multiplier": 1.12})

    scenarios["2a_plus_12_rehab_beds"] = SimConfig(**{
        **baseline_cfg.__dict__, "capacity": {**baseline_cfg.capacity, REHAB: baseline_cfg.capacity[REHAB] + 12}})

    P_2b = modify_P(extras["P"], src=RELAPSE, dst=N_T + 2, factor=0.75)
    cfg_2b = SimConfig(**{**baseline_cfg.__dict__, "P": P_2b, "phi": {ACUTE: 1, REHAB: 1, DAY: 1}})
    scenarios["2b_5th_community_team"] = calibrate_phi(cfg_2b, census_means)

    scenarios["2c_double_day_hospital"] = SimConfig(**{
        **baseline_cfg.__dict__, "capacity": {**baseline_cfg.capacity, DAY: baseline_cfg.capacity[DAY] * 2}})

    P_3 = modify_P(extras["P"], src=OPD, dst=N_T + 2, factor=2 / 3)
    cfg_3 = SimConfig(**{**baseline_cfg.__dict__, "P": P_3, "phi": {ACUTE: 1, REHAB: 1, DAY: 1}})
    scenarios["3_nonattendance_minus_third"] = calibrate_phi(cfg_3, census_means)

    scenarios["4_strike"] = SimConfig(**{
        **baseline_cfg.__dict__, "events": {baseline_cfg.warmup + 12: (0.532, 0.891)}})

    mult = 1 + 0.40 * extras["drug_diesel_share"]
    scenarios["5_drug_diesel_plus_40pct"] = SimConfig(**{
        **baseline_cfg.__dict__,
        "bed_cost_per_day": {k: v * mult for k, v in baseline_cfg.bed_cost_per_day.items()},
        "community_cost_per_month": {k: v * mult for k, v in baseline_cfg.community_cost_per_month.items()},
    })
    return scenarios


KEY_METRICS = ["occupancy_rate_0", "occupancy_rate_1", "overflow_beds_1", "wait_days_1",
               "abs_Lost to follow-up", "abs_Recovered and discharged from service",
               "monthly_relapse_events", "monthly_cost_ngn", "months_in_system"]


@celery_app.task(name="simulation.run_scenarios")
def run_scenarios() -> dict:
    baseline_cfg, extras = _build_baseline()
    scenarios = _build_scenarios(baseline_cfg, extras)

    results = {"0_baseline": run_replications(baseline_cfg, n_reps=N_REPS, seed0=0)["mean"]}
    for i, (name, cfg) in enumerate(scenarios.items()):
        results[name] = run_replications(cfg, n_reps=N_REPS, seed0=(i + 1) * 100)["mean"]

    comparison = pd.DataFrame(results).T[KEY_METRICS]

    avg_fare = extras["avg_fare"]
    monthly_intervention_cost = avg_fare * extras["visits_total"] / 24
    extra_retained_prop = (results["3_nonattendance_minus_third"]["abs_Recovered and discharged from service"]
                            - results["0_baseline"]["abs_Recovered and discharged from service"])
    extra_retained = extra_retained_prop * extras["lam"]
    cost_per_retained = monthly_intervention_cost / extra_retained if extra_retained > 0 else None

    released_beds = 8
    ltw_bed_cost_per_day = 26052.0
    annual_bed_cost_saved = released_beds * ltw_bed_cost_per_day * 365
    annual_community_cost_added = released_beds * baseline_cfg.community_cost_per_month[OPD] * 12
    scenario_6 = {
        "ltw_occupancy_before": extras["ltw_occupied"],
        "ltw_occupancy_after": extras["ltw_occupied"] - released_beds,
        "ltw_capacity": extras["ltw_capacity"],
        "annual_bed_cost_saved_ngn": annual_bed_cost_saved,
        "annual_community_cost_added_ngn": annual_community_cost_added,
        "net_annual_saving_ngn": annual_bed_cost_saved - annual_community_cost_added,
        "absorption_probability_change": "none -- Long stay institutional care is an "
            "absorbing state; resettlement acts on already-absorbed residents, not on "
            "the transition process leading there. P is unchanged by this scenario.",
    }

    return {
        "comparison": comparison.round(4).reset_index().rename(columns={"index": "scenario"}).to_dict(orient="records"),
        "scenario_3_cost_per_patient_retained_ngn": cost_per_retained,
        "scenario_6": scenario_6,
        "real_data_inputs_used": {
            "noshow_rate": extras["noshow_rate"],
            "drug_diesel_share_of_spend": extras["drug_diesel_share"],
        },
    }


@celery_app.task(name="simulation.strike_trajectory")
def strike_trajectory(n_seeds: int = 40) -> dict:
    """Separate task: scenario 4's effect is transient, so a full-horizon
    average (what run_scenarios reports for it) hides the real impact --
    see MODELING_LOG.md. This instead averages many paired baseline/strike
    runs and reports the month-by-month trajectory around the event."""
    baseline_cfg, extras = _build_baseline()
    cfg_s4 = SimConfig(**{**baseline_cfg.__dict__, "events": {baseline_cfg.warmup + 12: (0.532, 0.891)}})

    event_month = baseline_cfg.warmup + 12
    window = slice(event_month - 6, event_month + 18)
    months = (np.arange(event_month - 6, event_month + 18) - event_month).tolist()

    base_runs, strike_runs = [], []
    for seed in range(n_seeds):
        rb = simulate(baseline_cfg, seed=seed)
        rs = simulate(cfg_s4, seed=seed)
        base_runs.append(baseline_cfg.phi[REHAB] * rb["counts"][window, REHAB] / baseline_cfg.capacity[REHAB])
        strike_runs.append(cfg_s4.phi[REHAB] * rs["counts"][window, REHAB] / cfg_s4.capacity[REHAB])

    base_avg = np.mean(base_runs, axis=0)
    strike_avg = np.mean(strike_runs, axis=0)
    gap = (strike_avg - base_avg)

    trough_i = int(np.argmin(gap))
    recovered = next((months[i] for i in range(trough_i, len(months)) if abs(gap[i]) < 0.005), None)

    return {
        "months_relative_to_strike": months,
        "rehab_occupancy_baseline": base_avg.round(4).tolist(),
        "rehab_occupancy_strike": strike_avg.round(4).tolist(),
        "gap": gap.round(4).tolist(),
        "deepest_dip_month_offset": months[trough_i],
        "deepest_dip_gap": round(float(gap[trough_i]), 4),
        "recovery_month_offset": recovered,
        "note": "A gap near zero in the first month after the strike is expected, not "
                "an error -- new arrivals enter Acute or Outpatient only, never Rehab "
                "directly, so Rehab only feels the shock once patients who would have "
                "moved in are missing.",
    }