"""
Absorbing Markov chain over the FNPH Yaba care pathway.

... (existing module docstring unchanged) ...

Filtering: segment_code and family_support_score are denormalized directly
onto care_pathway_transitions, so those filters need no join. zone_id and
diagnosis are NOT on this table -- zone comes from patients (via
patient_id), and diagnosis comes from each patient's EARLIEST admission
(their "index" diagnosis), since a patient can carry different diagnoses
across multiple admissions and the pathway is modeled per-patient, not
per-admission. This is a stated modeling choice, not a neutral default --
see PathwayFilters docstring.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
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

# Below this many filtered transitions, the per-state row probabilities are
# too noisy to present with any confidence -- a project-chosen heuristic
# (roughly 15+ transitions per origin-state row across 10 states), not a
# formal power calculation. The UI should still show the result, but the
# sample size and this flag travel with it so the user can judge for
# themselves rather than mistake a thin estimate for a solid one.
MIN_RELIABLE_TRANSITIONS = 150

SEGMENT_CODES = [0, 1, 2, 3, 4]  # matches patients.segment_code / segment_label, Section 3.5

FAMILY_SUPPORT_TIERS = {
    "low": (0, 3),
    "medium": (4, 7),
    "high": (8, 10),
}  # project-chosen split of the 0-10 score into thirds; family_support_min/max
   # below overrides this with an exact custom range if the caller supplies one


@dataclass
class PathwayFilters:
    segment_code: int | None = None
    family_support_tier: str | None = None       # "low" | "medium" | "high"
    family_support_min: int | None = None         # overrides the tier bounds if set
    family_support_max: int | None = None
    zone_id: str | None = None
    diagnosis_group: str | None = None
    diagnosis_id: str | None = None

    def family_support_bounds(self):
        if self.family_support_min is not None or self.family_support_max is not None:
            return self.family_support_min, self.family_support_max
        if self.family_support_tier:
            return FAMILY_SUPPORT_TIERS[self.family_support_tier.lower()]
        return None, None


def load_transitions(filters: PathwayFilters | None = None) -> pd.DataFrame:
    """Pull raw monthly patient moves from care_pathway_transitions, optionally
    filtered. segment_code and family_support_score are columns on this table
    directly. zone_id requires a join to patients; diagnosis requires a join
    to each patient's earliest admission (their index diagnosis)."""
    filters = filters or PathwayFilters()
    fs_min, fs_max = filters.family_support_bounds()

    needs_patients = filters.zone_id is not None
    needs_diagnosis = filters.diagnosis_group is not None or filters.diagnosis_id is not None

    query = """
        SELECT t.patient_id, t.month_index, t.period_date, t.state_from,
               t.state_to, t.absorbed_flag
        FROM care_pathway_transitions t
    """
    if needs_patients:
        query += " LEFT JOIN patients p ON t.patient_id = p.patient_id "
    if needs_diagnosis:
        query += """
            LEFT JOIN (
                SELECT DISTINCT ON (a.patient_id) a.patient_id, a.diagnosis_id
                FROM admissions a
                ORDER BY a.patient_id, a.admit_date ASC
            ) idx_diag ON t.patient_id = idx_diag.patient_id
        """
        if filters.diagnosis_group:
            query += " LEFT JOIN diagnoses d ON idx_diag.diagnosis_id = d.diagnosis_id "

    conditions, params = [], {}
    if filters.segment_code is not None:
        conditions.append("t.segment_code = :segment_code")
        params["segment_code"] = filters.segment_code
    if fs_min is not None:
        conditions.append("t.family_support_score >= :fs_min")
        params["fs_min"] = fs_min
    if fs_max is not None:
        conditions.append("t.family_support_score <= :fs_max")
        params["fs_max"] = fs_max
    if filters.zone_id:
        conditions.append("p.zone_id = :zone_id")
        params["zone_id"] = filters.zone_id
    if filters.diagnosis_id:
        conditions.append("idx_diag.diagnosis_id = :diagnosis_id")
        params["diagnosis_id"] = filters.diagnosis_id
    if filters.diagnosis_group:
        conditions.append("d.diagnosis_group = :diagnosis_group")
        params["diagnosis_group"] = filters.diagnosis_group

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY t.patient_id, t.month_index"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def build_transition_matrix(transitions: pd.DataFrame):
    """Count state_from -> state_to moves and normalise each row to
    probabilities. Returns (P, states_with_no_data): states_with_no_data
    lists any TRANSIENT state that had zero observed transitions in this
    (possibly filtered) data -- that row cannot be estimated and is pinned
    to self-loop *only* so the matrix stays mathematically well-formed; the
    caller must still surface states_with_no_data to the user rather than
    silently presenting that pin as a real finding (a 100%-self-loop row is
    an absence of data, not a discovery that patients never leave that
    state).
    """
    counts = (
        transitions.groupby(["state_from", "state_to"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ALL_STATES, columns=ALL_STATES, fill_value=0)
    )
    row_sums = counts.sum(axis=1)

    states_with_no_data = []
    for state in ALL_STATES:
        if row_sums[state] == 0:
            counts.loc[state, state] = 1
            if state in TRANSIENT_STATES:
                states_with_no_data.append(state)

    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums, axis=0)
    return probs, states_with_no_data


def fundamental_matrix(P: pd.DataFrame):
    """Unchanged from the original -- split P into Q/R, compute N and B."""
    Q = P.loc[TRANSIENT_STATES, TRANSIENT_STATES].to_numpy()
    R = P.loc[TRANSIENT_STATES, ABSORBING_STATES].to_numpy()

    I = np.eye(len(TRANSIENT_STATES))
    N = np.linalg.inv(I - Q)
    B = N @ R

    N_df = pd.DataFrame(N, index=TRANSIENT_STATES, columns=TRANSIENT_STATES)
    B_df = pd.DataFrame(B, index=TRANSIENT_STATES, columns=ABSORBING_STATES)
    return N_df, B_df


def expected_months_to_absorption(N_df: pd.DataFrame) -> pd.Series:
    return N_df.sum(axis=1)


def compute_pathway_summary(filters: PathwayFilters | None = None) -> dict:
    """One entry point the FastAPI endpoint calls. filters=None (or an
    all-None PathwayFilters) reproduces the original unfiltered behaviour
    exactly."""
    filters = filters or PathwayFilters()
    transitions = load_transitions(filters)
    n_transitions = len(transitions)

    if n_transitions == 0:
        return {
            "error": "no_data",
            "message": "No transitions match this filter combination.",
            "sample_size": 0,
            "filters_applied": filters.__dict__,
        }

    P, states_with_no_data = build_transition_matrix(transitions)
    N_df, B_df = fundamental_matrix(P)
    expected_months = expected_months_to_absorption(N_df)

    last_rows = transitions.sort_values("month_index").groupby("patient_id").tail(1)
    censored_fraction = float((last_rows["absorbed_flag"] == 0).mean())

    return {
        "transition_matrix": P.round(4).to_dict(orient="index"),
        "fundamental_matrix": N_df.round(3).to_dict(orient="index"),
        "expected_months_to_absorption": expected_months.round(2).to_dict(),
        "absorption_probabilities": B_df.round(4).to_dict(orient="index"),
        "censored_patient_fraction": round(censored_fraction, 4),
        "sample_size": n_transitions,
        "patient_count": int(transitions["patient_id"].nunique()),
        "reliable": n_transitions >= MIN_RELIABLE_TRANSITIONS,
        "min_reliable_transitions": MIN_RELIABLE_TRANSITIONS,
        "states_with_no_data": states_with_no_data,
        "filters_applied": {k: v for k, v in filters.__dict__.items() if v is not None},
    }