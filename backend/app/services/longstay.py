"""
Service for querying the Long-Stay Residents register and survival metrics.
"""
import pandas as pd
from sqlalchemy import text
from app.db import engine

def get_longstay_residents() -> list[dict]:
    query = text("""
        SELECT 
            resident_id,
            unit_id,
            sex,
            age_years,
            diagnosis_id,
            year_of_entry,
            years_resident,
            funding_source,
            annual_grant_ngn,
            next_of_kin_contact,
            guardianship_status,
            mobility_band,
            nursing_hours_per_week,
            resettlement_assessed_flag,
            still_resident_flag
        FROM long_stay_residents
        ORDER BY year_of_entry ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    results = []
    for _, row in df.iterrows():
        rid = str(row['resident_id'])
        fs_raw = str(row['funding_source'] or '').lower()
        if 'federal' in fs_raw:
            fs = 'Federal Grant'
        elif 'state' in fs_raw:
            fs = 'State Subsidized'
        elif 'family' in fs_raw:
            fs = 'Family Funded'
        else:
            fs = 'Indigent Hospital Reserve'

        assessed = bool(row['resettlement_assessed_flag'])
        mob = str(row['mobility_band'] or '').lower()
        if mob == 'independent' and assessed:
            feasibility = 'Eligible for Community Step-down'
        elif mob in ('wheelchair', 'bed bound'):
            feasibility = 'Specialized Institutional'
        else:
            feasibility = 'High Support Needed'

        results.append({
            'id': rid,
            'residentCode': rid,
            'unit': str(row['unit_id']),
            'gender': str(row['sex']),
            'age': int(row['age_years']),
            'yearsResident': int(row['years_resident']),
            'admissionYear': int(row['year_of_entry']),
            'fundingSource': fs,
            'resettlementAssessedFlag': assessed,
            'resettlementFeasibility': feasibility,
            'primaryDiagnosis': str(row['diagnosis_id']),
        })
    return results
