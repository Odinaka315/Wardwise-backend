"""
Patient segmentation service, using the K-Prototypes model fitted in
Week 3 (see MODELING_LOG.md for the full method-selection story: plain
K-means was dominated by one-hot categorical columns, unscaled
K-Prototypes was dominated by distance_km alone; this final version
uses K-Prototypes with scaled numeric features and travel_fare_ngn
dropped as redundant with distance_km).

segment_label / segment_code are never used as inputs here -- only as
the post-hoc ARI/NMI validation already recorded in the log.
"""

import json
import os

import joblib
import pandas as pd
from sqlalchemy import text

from app.db import engine

ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "/app/model_artifacts")

_model = None
_scaler = None
_config = None
_cluster_labels = None


def _load_artifacts():
    global _model, _scaler, _config, _cluster_labels
    if _model is None:
        paths = {
            "model": os.path.join(ARTIFACT_DIR, "segmentation_model.pkl"),
            "scaler": os.path.join(ARTIFACT_DIR, "segmentation_scaler.pkl"),
            "config": os.path.join(ARTIFACT_DIR, "segmentation_config.json"),
            "labels": os.path.join(ARTIFACT_DIR, "segmentation_cluster_labels.json"),
        }
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing segmentation artefact ({name}): {path}. "
                    f"Download it from Drive into {ARTIFACT_DIR} first."
                )
        _model = joblib.load(paths["model"])
        _scaler = joblib.load(paths["scaler"])
        _config = json.load(open(paths["config"]))
        _cluster_labels = json.load(open(paths["labels"]))
    return _model, _scaler, _config, _cluster_labels


def _prepare_features(raw: pd.DataFrame, config: dict, scaler) -> pd.DataFrame:
    """Rebuilds the exact feature matrix the model was fit on: same
    column order, same education_level fill, same numeric scaling.
    """
    df = raw[config["features"]].copy()
    df["education_level"] = df["education_level"].fillna("Unknown")
    df[config["numeric_cols"]] = scaler.transform(df[config["numeric_cols"]])
    return df


def score_patients(patient_ids: list[str] | None = None) -> pd.DataFrame:
    """Assigns a cluster (and its human-readable label) to patients.
    Pass specific patient_ids to score just those, or leave None to
    score every patient in the database."""
    model, scaler, config, cluster_labels = _load_artifacts()

    query = "SELECT * FROM patients"
    params = {}
    if patient_ids:
        query += " WHERE patient_id = ANY(:ids)"
        params["ids"] = patient_ids

    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params)

    X = _prepare_features(raw, config, scaler)
    labels = model.predict(X.values, categorical=config["categorical_indices"])

    result = raw[["patient_id"]].copy()
    result["cluster"] = labels
    result["cluster_label"] = result["cluster"].astype(str).map(cluster_labels)
    return result


def cluster_summary() -> dict:
    """Powers a segmentation overview screen: size and share of every
    cluster, with its human-readable interpretation."""
    scored = score_patients()
    counts = scored["cluster"].value_counts().sort_index()
    _, _, _, cluster_labels = _load_artifacts()

    return {
        "total_patients": int(len(scored)),
        "clusters": [
            {
                "cluster": int(cluster),
                "label": cluster_labels.get(str(cluster), "Unknown"),
                "patient_count": int(count),
                "pct_of_total": round(float(count) / len(scored) * 100, 1),
            }
            for cluster, count in counts.items()
        ],
    }
