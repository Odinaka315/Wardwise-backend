"""
One-off maintenance script: updates only the 4 columns actually changed
during Week 1 cleaning (length_of_stay_days, family_support_score,
medication_adherence_score, total_cost_ngn) in the existing 'admissions'
table, matched by admission_id.

Deliberately does NOT drop or recreate the admissions table itself --
that table has foreign keys from weekly_assessments, ect_sessions, and
lab_workflow, plus several views built on top of it, all of which would
break if the table were dropped and rebuilt. A column-level update
leaves every relationship intact.

Run this ONCE after downloading admissions_cleaned.csv from Drive.

Usage (from the project root, with docker compose up -d running):
    docker compose exec api python -m app.scripts.load_cleaned_admissions
    firebender
"""

import pandas as pd
from sqlalchemy import text

from app.db import engine

CSV_PATH = "/app/data/admissions_cleaned.csv"
CHANGED_COLUMNS = [
    "length_of_stay_days",
    "family_support_score",
    "medication_adherence_score",
    "total_cost_ngn",
]
STAGING_TABLE = "admissions_cleaned_staging"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    staging_df = df[["admission_id"] + CHANGED_COLUMNS]

    # The staging table is brand new each run and has no dependents,
    # so it's safe to drop/recreate -- only the real admissions table
    # needed the more careful treatment below.
    staging_df.to_sql(STAGING_TABLE, engine, if_exists="replace", index=False)
    print(f"Staged {len(staging_df)} rows into {STAGING_TABLE}")

    set_clause = ", ".join(f"{col} = s.{col}" for col in CHANGED_COLUMNS)
    update_sql = text(
        f"""
        UPDATE admissions a
        SET {set_clause}
        FROM {STAGING_TABLE} s
        WHERE a.admission_id = s.admission_id
        """
    )

    with engine.begin() as conn:
        result = conn.execute(update_sql)
        print(f"Updated {result.rowcount} rows in admissions")
        conn.execute(text(f"DROP TABLE {STAGING_TABLE}"))
        print(f"Dropped temporary staging table {STAGING_TABLE}")

    with engine.connect() as conn:
        count = conn.exec_driver_sql("SELECT COUNT(*) FROM admissions").scalar()
        missing = conn.exec_driver_sql(
            """
            SELECT
              COUNT(*) FILTER (WHERE family_support_score IS NULL) AS missing_family,
              COUNT(*) FILTER (WHERE medication_adherence_score IS NULL) AS missing_med,
              COUNT(*) FILTER (WHERE total_cost_ngn IS NULL) AS missing_cost
            FROM admissions
            """
        ).fetchone()
    print(f"Row count in DB (unchanged, as expected): {count}")
    print(f"Remaining missing values (should all be 0): {missing}")


if __name__ == "__main__":
    main()
