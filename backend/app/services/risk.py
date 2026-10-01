"""
Risk scoring service for the two Week 3 classifiers: 90-day readmission
and outpatient non-attendance.

Both models were trained in Colab as logistic regression (scaled,
class-balanced), chosen over XGBoost on ROC-AUC/PR-AUC in both cases.
Model artefacts (the .pkl files, plus the exact training column list
each model expects) must be present in MODEL_ARTIFACT_DIR before this
service will work — see README for how to get them there from Drive.
"""

import json
import os

import joblib
import pandas as pd
from sqlalchemy import text

from app.db import engine

ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "/app/model_artifacts")

READMISSION_FEATURES = [
    "age_at_admission", "sex", "legal_status", "referral_source",
    "prior_admissions", "family_support_score", "distance_km",
    "payment_source", "severity_on_admission", "severity_on_discharge",
    "medication_adherence_score", "seclusion_episodes",
    "one_to_one_nursing_days", "psychology_sessions",
    "occupational_therapy_sessions", "social_work_contacts",
    "length_of_stay_days", "ect_course_flag",
]
READMISSION_ONEHOT_COLS = ["sex", "legal_status", "referral_source", "payment_source"]

NONATTENDANCE_FEATURES = [
    "clinic", "scheduled_slot_hour", "booking_lead_days", "visit_type",
    "reminder_sent_flag", "distance_km", "travel_fare_ngn",
]
NONATTENDANCE_ONEHOT_COLS = ["clinic", "visit_type"]


def _load_artifact_set(prefix: str):
    """Loads {prefix}_model.pkl, {prefix}_scaler.pkl and
    {prefix}_columns.json from ARTIFACT_DIR. Raises a clear error if
    any piece is missing, rather than a confusing downstream failure."""
    model_path = os.path.join(ARTIFACT_DIR, f"{prefix}_model.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, f"{prefix}_scaler.pkl")
    columns_path = os.path.join(ARTIFACT_DIR, f"{prefix}_columns.json")

    for path in (model_path, scaler_path, columns_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing model artefact: {path}. Download it from Drive "
                f"into {ARTIFACT_DIR} before calling this endpoint."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    columns = json.load(open(columns_path))
    return model, scaler, columns


def _prepare_features(raw: pd.DataFrame, feature_cols, onehot_cols, training_columns) -> pd.DataFrame:
    """Rebuilds the exact same one-hot-encoded, column-aligned feature
    matrix the model was trained on, from raw rows straight out of the
    database. This is the step that keeps live predictions consistent
    with training — get it wrong and the model silently misreads
    which number means what.
    """
    df = raw[feature_cols].copy()
    df = pd.get_dummies(df, columns=onehot_cols)
    # reindex to the exact training-time columns: adds any missing
    # dummy column as all-zero (a category not present in this batch),
    # and drops/reorders anything unexpected, in one step
    df = df.reindex(columns=training_columns, fill_value=0)
    return df


def score_readmission_risk(admission_ids: list[str] | None = None) -> pd.DataFrame:
    """Scores admissions for 90-day readmission risk. Pass specific
    admission_ids to score just those, or leave None to score every
    admission currently in the database."""
    model, scaler, columns = _load_artifact_set("readmission")

    query = "SELECT * FROM admissions"
    params = {}
    if admission_ids:
        query += " WHERE admission_id = ANY(:ids)"
        params["ids"] = admission_ids

    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params)

    X = _prepare_features(raw, READMISSION_FEATURES, READMISSION_ONEHOT_COLS, columns)
    X_scaled = scaler.transform(X)
    scores = model.predict_proba(X_scaled)[:, 1]

    result = raw[["admission_id", "patient_id"]].copy()
    result["readmission_risk_score"] = scores
    return result.sort_values("readmission_risk_score", ascending=False)


def readmission_worklist(top_pct: float = 10.0) -> dict:
    """The board-facing version: everyone scored, ranked, and the top
    N% marked as the actionable follow-up list, plus the lift number
    that proves the list is better than a random selection.
    """
    scored = score_readmission_risk()
    cutoff = scored["readmission_risk_score"].quantile(1 - top_pct / 100)
    scored["flagged_high_risk"] = scored["readmission_risk_score"] >= cutoff

    with engine.connect() as conn:
        outcomes = pd.read_sql(
            text("SELECT admission_id, readmitted_90d_flag FROM admissions"), conn
        )
    merged = scored.merge(outcomes, on="admission_id", how="left")
    flagged = merged[merged["flagged_high_risk"]]

    return {
        "threshold_score": round(float(cutoff), 4),
        "flagged_count": int(len(flagged)),
        "flagged_pct_of_total": round(len(flagged) / len(merged) * 100, 2),
        "flagged_actual_readmission_rate": round(float(flagged["readmitted_90d_flag"].mean()), 4)
        if len(flagged) else None,
        "hospital_wide_base_rate": round(float(merged["readmitted_90d_flag"].mean()), 4),
        "worklist": flagged[["admission_id", "patient_id", "readmission_risk_score"]]
        .to_dict(orient="records"),
    }


def score_nonattendance_risk(visit_ids: list[str] | None = None) -> pd.DataFrame:
    """Scores outpatient visits for non-attendance risk. Pass specific
    visit_ids to score just those, or leave None to score every visit
    currently in the database."""
    model, scaler, columns = _load_artifact_set("nonattendance")

    query = "SELECT * FROM outpatient_visits"
    params = {}
    if visit_ids:
        query += " WHERE visit_id = ANY(:ids)"
        params["ids"] = visit_ids

    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params)

    raw["scheduled_dayofweek"] = pd.to_datetime(raw["scheduled_date"]).dt.dayofweek

    feature_cols = NONATTENDANCE_FEATURES + ["scheduled_dayofweek"]
    X = _prepare_features(raw, feature_cols, NONATTENDANCE_ONEHOT_COLS, columns)
    X_scaled = scaler.transform(X)
    scores = model.predict_proba(X_scaled)[:, 1]

    result = raw[["visit_id", "patient_id", "scheduled_date"]].copy()
    result["nonattendance_risk_score"] = scores
    return result.sort_values("nonattendance_risk_score", ascending=False)


def nonattendance_worklist(top_pct: float = 10.0) -> dict:
    """Same pattern as readmission_worklist: top N% of upcoming visits
    by predicted no-show risk, plus the lift over the base rate."""
    scored = score_nonattendance_risk()
    cutoff = scored["nonattendance_risk_score"].quantile(1 - top_pct / 100)
    scored["flagged_high_risk"] = scored["nonattendance_risk_score"] >= cutoff

    with engine.connect() as conn:
        outcomes = pd.read_sql(
            text("SELECT visit_id, did_not_attend_flag FROM outpatient_visits"), conn
        )
    merged = scored.merge(outcomes, on="visit_id", how="left")
    flagged = merged[merged["flagged_high_risk"]]

    return {
        "threshold_score": round(float(cutoff), 4),
        "flagged_count": int(len(flagged)),
        "flagged_pct_of_total": round(len(flagged) / len(merged) * 100, 2),
        "flagged_actual_noshow_rate": round(float(flagged["did_not_attend_flag"].mean()), 4)
        if len(flagged) else None,
        "hospital_wide_base_rate": round(float(merged["did_not_attend_flag"].mean()), 4),
        "worklist": flagged[["visit_id", "patient_id", "scheduled_date", "nonattendance_risk_score"]]
        .to_dict(orient="records"),
    }
