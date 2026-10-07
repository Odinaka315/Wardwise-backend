from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import text
from app.db import engine

router = APIRouter(prefix="/api/v1/baseline", tags=["baseline"])

class BaselineSummary(BaseModel):
    occupancy_series: list[dict]
    avg_length_of_stay: float
    avg_cost_ngn: float
    attendance_rate: float
    safety_incidents: int

@router.get("/summary", response_model=BaselineSummary)
def get_baseline_summary(
    unit_id: Optional[str] = None,
    diagnosis_group: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    with engine.connect() as conn:
        # 1. Occupancy & Capacity (from daily_census)
        census_query = "SELECT census_date, SUM(bed_capacity) as nominal_capacity, SUM(occupied_beds) as occupied_beds FROM daily_census WHERE 1=1"
        census_params = {}
        if unit_id:
            census_query += " AND unit_id = :unit_id"
            census_params["unit_id"] = unit_id
        if start_date:
            census_query += " AND census_date >= :start_date"
            census_params["start_date"] = start_date
        if end_date:
            census_query += " AND census_date <= :end_date"
            census_params["end_date"] = end_date
        census_query += " GROUP BY census_date ORDER BY census_date"
        
        census_df = pd.read_sql(text(census_query), conn, params=census_params)
        # Calculate occupancy rate directly from aggregated beds
        census_df['occupancy_rate'] = (census_df['occupied_beds'] / census_df['nominal_capacity'].replace(0, 1)) * 100
        occupancy_series = census_df.assign(
            census_date=census_df['census_date'].astype(str)
        ).to_dict(orient="records")

        # 2. Stay & Cost (from admissions joined with diagnoses)
        adm_query = """
            SELECT a.length_of_stay_days, a.total_cost_ngn 
            FROM admissions a
            LEFT JOIN diagnoses d ON a.diagnosis_id = d.diagnosis_id
            WHERE 1=1
        """
        adm_params = {}
        if unit_id:
            adm_query += " AND a.unit_id = :unit_id"
            adm_params["unit_id"] = unit_id
        if diagnosis_group:
            adm_query += " AND d.diagnosis_group = :diagnosis_group"
            adm_params["diagnosis_group"] = diagnosis_group
        if start_date:
            adm_query += " AND a.admit_date >= :start_date"
            adm_params["start_date"] = start_date
        if end_date:
            adm_query += " AND a.admit_date <= :end_date"
            adm_params["end_date"] = end_date
            
        adm_df = pd.read_sql(text(adm_query), conn, params=adm_params)
        avg_los = adm_df['length_of_stay_days'].mean() if not adm_df.empty else 0.0
        avg_cost = adm_df['total_cost_ngn'].mean() if not adm_df.empty else 0.0

        # 3. Attendance (from outpatient_visits)
        # Note: outpatient_visits doesn't map cleanly to unit_id, filtering by date primarily
        op_query = "SELECT did_not_attend_flag FROM outpatient_visits WHERE 1=1"
        op_params = {}
        if start_date:
            op_query += " AND scheduled_date >= :start_date"
            op_params["start_date"] = start_date
        if end_date:
            op_query += " AND scheduled_date <= :end_date"
            op_params["end_date"] = end_date
            
        op_df = pd.read_sql(text(op_query), conn, params=op_params)
        attendance_rate = (1 - op_df['did_not_attend_flag'].mean()) * 100 if not op_df.empty else 0.0

        # 4. Safety Indicators (from ward_incidents)
        inc_query = "SELECT COUNT(*) as incident_count FROM ward_incidents WHERE 1=1"
        inc_params = {}
        if unit_id:
            inc_query += " AND unit_id = :unit_id"
            inc_params["unit_id"] = unit_id
        if start_date:
            inc_query += " AND incident_ts >= :start_date"
            inc_params["start_date"] = start_date
        if end_date:
            inc_query += " AND incident_ts <= :end_date"
            inc_params["end_date"] = end_date
            
        inc_df = pd.read_sql(text(inc_query), conn, params=inc_params)
        safety_incidents = int(inc_df['incident_count'].iloc[0]) if not inc_df.empty else 0

        return BaselineSummary(
            occupancy_series=occupancy_series,
            avg_length_of_stay=round(avg_los, 1),
            avg_cost_ngn=round(avg_cost, 2),
            attendance_rate=round(attendance_rate, 1),
            safety_incidents=safety_incidents
        )