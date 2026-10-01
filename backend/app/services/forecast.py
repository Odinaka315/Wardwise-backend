"""
Per-unit admissions forecasting, using whichever method actually won
in Week 3 evaluation for each unit: pooled XGBoost for units where it
beat the naive baseline (DAY, DRU, FOR, GER), and a 3-month trailing
average everywhere else (ACF, ACM, CAU, CTU, LSR, LTW) -- see
MODELING_LOG.md for the full comparison that justified this split.

Forecasts the single next calendar month after the latest month
present in the admissions table, per unit.
"""

import json
import os

import joblib
import pandas as pd
from sqlalchemy import text

from app.db import engine

ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "/app/model_artifacts")

_model = None
_columns = None
_champion_map = None


def _load_artifacts():
    """Lazy-loads the XGBoost model, its training columns, and the
    champion map once per process, rather than on every request."""
    global _model, _columns, _champion_map
    if _model is None:
        model_path = os.path.join(ARTIFACT_DIR, "forecast_xgb_model.pkl")
        columns_path = os.path.join(ARTIFACT_DIR, "forecast_xgb_columns.json")
        champion_path = os.path.join(ARTIFACT_DIR, "forecast_champion_map.json")
        for path in (model_path, columns_path, champion_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing forecast artefact: {path}. Download it from "
                    f"Drive into {ARTIFACT_DIR} before calling this endpoint."
                )
        _model = joblib.load(model_path)
        _columns = json.load(open(columns_path))
        _champion_map = json.load(open(champion_path))
    return _model, _columns, _champion_map


def _get_unit_monthly_series(unit_id: str) -> pd.Series:
    """Rebuilds one unit's full monthly admissions history from the
    database, filling any zero-admission months that would otherwise
    be silently absent -- the same gap-filling done in Colab, applied
    live here so a quiet month never breaks the lag features.
    """
    with engine.connect() as conn:
        raw = pd.read_sql(
            text("SELECT admit_date FROM admissions WHERE unit_id = :uid"),
            conn, params={"uid": unit_id},
        )
    if raw.empty:
        raise ValueError(f"No admissions found for unit_id {unit_id}")

    raw["admit_date"] = pd.to_datetime(raw["admit_date"])
    monthly = raw.set_index("admit_date").resample("MS").size()

    # reindex across the FULL date range seen for this hospital overall,
    # not just this unit's own min/max, so lag_12 always has something
    # to look back on even for a unit with a short history
    with engine.connect() as conn:
        bounds = conn.execute(
            text("SELECT MIN(admit_date), MAX(admit_date) FROM admissions")
        ).fetchone()
    full_range = pd.date_range(bounds[0], bounds[1], freq="MS")
    monthly = monthly.reindex(full_range, fill_value=0)
    return monthly


def _forecast_naive(monthly: pd.Series) -> float:
    """Trailing 3-month average -- the method that won for low-volume
    or noisy units, where XGBoost couldn't beat this simple baseline."""
    return float(monthly.iloc[-3:].mean())


def _forecast_xgb(unit_id: str, monthly: pd.Series) -> float:
    """Builds the same lag_1 / lag_12 / month_num / one-hot-unit
    feature row the pooled model was trained on, for the single next
    month after the latest data point."""
    model, columns, _ = _load_artifacts()

    target_month = monthly.index[-1] + pd.DateOffset(months=1)
    lag_1 = float(monthly.iloc[-1])
    lag_12 = float(monthly.iloc[-12]) if len(monthly) >= 12 else float(monthly.mean())

    row = {col: 0 for col in columns}
    row["lag_1"] = lag_1
    row["lag_12"] = lag_12
    row["month_num"] = target_month.month
    unit_col = f"unit_{unit_id}"
    if unit_col in row:
        row[unit_col] = 1

    X = pd.DataFrame([row])[columns]
    return float(model.predict(X)[0])


def forecast_unit(unit_id: str) -> dict:
    """Forecasts next month's admissions for one unit, using whichever
    method won for it in evaluation."""
    _, _, champion_map = _load_artifacts()
    if unit_id not in champion_map:
        raise ValueError(f"Unknown unit_id {unit_id}. Known units: {list(champion_map)}")

    monthly = _get_unit_monthly_series(unit_id)
    method = champion_map[unit_id]
    raw_forecast = _forecast_xgb(unit_id, monthly) if method == "xgb" else _forecast_naive(monthly)
    target_month = monthly.index[-1] + pd.DateOffset(months=1)

    # Admissions can never be negative. XGBoost has no inherent floor,
    # so low-volume units occasionally predict below zero -- clip as a
    # safety net, and say so explicitly rather than silently changing
    # the number.
    clipped = raw_forecast < 0
    forecast = max(0.0, raw_forecast)

    return {
        "unit_id": unit_id,
        "forecast_month": target_month.strftime("%Y-%m-01"),
        "predicted_admissions": round(forecast, 1),
        "method_used": method,
        "based_on_data_through": monthly.index[-1].strftime("%Y-%m-01"),
        "clipped_from_negative": clipped,
    }


def forecast_all_units() -> list[dict]:
    """Forecasts every admitting unit at once, for the forecast screen."""
    _, _, champion_map = _load_artifacts()
    return [forecast_unit(unit_id) for unit_id in champion_map]
