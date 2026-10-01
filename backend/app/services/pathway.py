"""
Absorbing Markov chain over the FNPH Yaba care pathway.

Column and state names below are confirmed against the real
care_pathway_transitions table (verified in Colab, week 2), replacing
the original placeholder assumptions.

Feeds Model 2 (bed & staffing configuration) with two of its key
parameters: expected months per state (from N), which combines with
forecast demand to size each ward, and absorption probabilities
(from B), which tell you how many beds are effectively locked into
long-stay care versus turning over.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

TRANSIENT_STATES = [
    "Acute inpatient",
    "Rehabilitation inpatient",
    "Day hospital",
    "Outpatient follow-up",
    "Community relapse",
]

ABSORBING_STATES = [
    "Deceased",
    "Long stay institutional care",
    "Lost to follow-up",
    "Recovered and discharged from service",
    "Transferred to another facility",
]

ALL_STATES = TRANSIENT_STATES + ABSORBING_STATES


def load_transitions() -> pd.DataFrame:
    """Pull raw monthly patient moves from care_pathway_transitions."""
    query = text(
        """
        SELECT patient_id, month_index, period_date, state_from, state_to,
               absorbed_flag
        FROM care_pathway_transitions
        ORDER BY patient_id, month_index
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def build_transition_matrix(transitions: pd.DataFrame) -> pd.DataFrame:
    """Count state_from -> state_to moves and normalise each row to
    probabilities. Rows are indexed/columned by ALL_STATES so the
    canonical Q/R split below is just array slicing.
    """
    counts = (
        transitions.groupby(["state_from", "state_to"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ALL_STATES, columns=ALL_STATES, fill_value=0)
    )
    row_sums = counts.sum(axis=1)
    # Absorbing states have no outgoing moves in principle, but pin them
    # to stay put in case of any zero-row edge case, so normalising
    # below never divides by zero.
    for state in ABSORBING_STATES:
        if row_sums[state] == 0:
            counts.loc[state, state] = 1
    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums, axis=0)
    return probs


def fundamental_matrix(P: pd.DataFrame):
    """Split P into Q (transient-to-transient) and R (transient-to-
    absorbing), then compute N = (I - Q)^-1 and B = N @ R.

    Returns (N, B) as DataFrames indexed/columned for readability.
    """
    Q = P.loc[TRANSIENT_STATES, TRANSIENT_STATES].to_numpy()
    R = P.loc[TRANSIENT_STATES, ABSORBING_STATES].to_numpy()

    I = np.eye(len(TRANSIENT_STATES))
    N = np.linalg.inv(I - Q)
    B = N @ R

    N_df = pd.DataFrame(N, index=TRANSIENT_STATES, columns=TRANSIENT_STATES)
    B_df = pd.DataFrame(B, index=TRANSIENT_STATES, columns=ABSORBING_STATES)
    return N_df, B_df


def expected_months_to_absorption(N_df: pd.DataFrame) -> pd.Series:
    """Row sums of N: the true expected length of care from each
    starting state, in months, before the patient is absorbed."""
    return N_df.sum(axis=1)


def compute_pathway_summary() -> dict:
    """One entry point the FastAPI endpoint calls. Returns everything
    the pathway explorer screen needs, JSON-serialisable."""
    transitions = load_transitions()
    P = build_transition_matrix(transitions)
    N_df, B_df = fundamental_matrix(P)
    expected_months = expected_months_to_absorption(N_df)

    # Censoring note: a small fraction of patients (~1.4% in this
    # dataset) have no absorbed_flag=1 row yet because their care is
    # still ongoing as of the data extract date. This doesn't affect
    # the chain math above, which only counts observed moves, but is
    # worth surfacing to the user rather than hiding.
    last_rows = transitions.sort_values("month_index").groupby("patient_id").tail(1)
    censored_fraction = float((last_rows["absorbed_flag"] == 0).mean())

    return {
        "transition_matrix": P.round(4).to_dict(orient="index"),
        "fundamental_matrix": N_df.round(3).to_dict(orient="index"),
        "expected_months_to_absorption": expected_months.round(2).to_dict(),
        "absorption_probabilities": B_df.round(4).to_dict(orient="index"),
        "censored_patient_fraction": round(censored_fraction, 4),
    }


if __name__ == "__main__":
    # Run this directly during development to sanity-check the
    # arithmetic against what you know about the wards, before
    # trusting the endpoint. e.g. `python -m app.services.pathway`
    summary = compute_pathway_summary()
    for key, value in summary.items():
        print(f"\n--- {key} ---")
        if isinstance(value, dict):
            print(pd.DataFrame(value))
        else:
            print(value)
