"""
Descriptive baseline service for the hospital units (Overview / Baseline screen).
Profiles capacity, occupancy, stay, staffing, and locked status across all units.
"""
import pandas as pd
from sqlalchemy import text
from app.db import engine

CATEGORY_MAP = {
    'ACM': 'Acute',
    'ACF': 'Acute',
    'CAU': 'Specialist',
    'FOR': 'Specialist',
    'DRU': 'Rehabilitation',
    'LSR': 'Rehabilitation',
    'LTW': 'Long-Term',
    'GER': 'Specialist',
    'CTU': 'Outpatient',
    'DAY': 'Outpatient',
    'OPD': 'Outpatient',
}

def get_units_baseline() -> list[dict]:
    query = text("""
        SELECT 
            u.unit_id AS code,
            u.unit_name AS name,
            u.bed_capacity AS total_beds,
            COALESCE(c.avg_occ, u.bed_capacity * 0.9) AS occupied_beds,
            COALESCE(adm.avg_los, 25.0) AS avg_los,
            COALESCE(b.locked_beds > 0, false) AS is_locked,
            COALESCE(s.staff_count, 20) AS assigned_staff,
            COALESCE(cost.daily_bed_cost * 30 * u.bed_capacity, 12000000) AS monthly_cost
        FROM units u
        LEFT JOIN (
            SELECT unit_id, AVG(occupied_beds) AS avg_occ
            FROM daily_census
            WHERE census_date >= (SELECT MAX(census_date) - INTERVAL '6 months' FROM daily_census)
            GROUP BY unit_id
        ) c ON u.unit_id = c.unit_id
        LEFT JOIN (
            SELECT unit_id, AVG(length_of_stay_days) AS avg_los
            FROM admissions
            GROUP BY unit_id
        ) adm ON u.unit_id = adm.unit_id
        LEFT JOIN (
            SELECT unit_id, SUM(locked_flag) AS locked_beds
            FROM beds
            GROUP BY unit_id
        ) b ON u.unit_id = b.unit_id
        LEFT JOIN (
            SELECT home_unit_id AS unit_id, COUNT(*) AS staff_count
            FROM staff
            GROUP BY home_unit_id
        ) s ON u.unit_id = s.unit_id
        LEFT JOIN (
            SELECT unit_id, AVG(daily_cost_ngn) AS daily_bed_cost
            FROM beds
            GROUP BY unit_id
        ) cost ON u.unit_id = cost.unit_id
        ORDER BY u.unit_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    results = []
    for _, row in df.iterrows():
        code = str(row['code'])
        total_beds = int(row['total_beds'])
        occupied_beds = round(float(row['occupied_beds']), 1)
        occ_rate = round((occupied_beds / total_beds * 100) if total_beds > 0 else 0, 1)
        
        results.append({
            'code': code,
            'name': str(row['name']),
            'category': CATEGORY_MAP.get(code, 'Specialist'),
            'totalBeds': total_beds,
            'occupiedBeds': occupied_beds,
            'occupancyRate': occ_rate,
            'avgLengthOfStayDays': round(float(row['avg_los']), 1),
            'monthlyOperatingCostNgn': round(float(row['monthly_cost']), 0),
            'assignedStaff': int(row['assigned_staff']),
            'isLocked': bool(row['is_locked']),
        })
    return results
