# db_utils.py
"""Storing functions relating to database
cursor setup as well as queries to keep main app
clean of stray functions. """

import sqlite3
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv('secrets.env')
DB_PATH = os.getenv("DATABASE")

def dict_factory(cursor, row):
    """Convert database row objects to a dictionary"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_latest_activity(db_path = DB_PATH):
    """Retrieves latest Run or Ride activity ID as int"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""SELECT id FROM activities
                    WHERE type in ("Run", "Ride")
                    ORDER BY start_date DESC LIMIT 1""")
        row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as e:
        print(f"Failed to get id: {e}")
        return ""
    
# Get polyline on startup to avoid race conditions
def get_latest_polyline(activity_id_polyline, db_path = DB_PATH):
    """Retreives a polyline for the latest activity id"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT map_summary_polyline FROM activities WHERE id = {activity_id_polyline}")
        row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as e:
        print(f"Failed to get polyline: {e}")
        return ""
    
def get_acwr_data(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        WITH daily_load AS (
            SELECT 
                date(datetime(start_date)) as date,
                SUM(average_speed * moving_time / 1000.0) as daily_km
            FROM activities
            WHERE type = 'Run'
            GROUP BY date
        ),
        rolling_loads AS (
            SELECT
                date,
                daily_km,
                SUM(daily_km) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) / 7.0 AS acute_load,
                SUM(daily_km) OVER (ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) / 28.0 AS chronic_load
            FROM daily_load
        )
        SELECT
            date,
            daily_km,
            acute_load,
            chronic_load,
            ROUND(CAST(acute_load AS REAL) / NULLIF(chronic_load, 0), 2) AS acwr
        FROM rolling_loads
        WHERE chronic_load > 0
        ORDER BY date DESC
        LIMIT 90;
    """, conn)
    conn.close()
    return df

def get_hr_drift_data(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        WITH activity_durations AS (
        SELECT
            activity_id,
            MIN(CAST(time_data AS INTEGER)) AS min_time,
            MAX(CAST(time_data AS INTEGER)) AS max_time,
            (MAX(CAST(time_data AS INTEGER)) - MIN(CAST(time_data AS INTEGER))) AS total_duration
        FROM streams
        GROUP BY activity_id
    ),
    stream_segments AS (
        SELECT
            s.activity_id,
            CAST(s.time_data AS INTEGER) AS time_data,
            CAST(s.heartrate_data AS INTEGER) AS heartrate_data,
            ad.total_duration,
            CASE
                WHEN CAST(s.time_data AS INTEGER) - ad.min_time <= ad.total_duration * 0.33 THEN 'first'
                WHEN CAST(s.time_data AS INTEGER) - ad.min_time >= ad.total_duration * 0.66 THEN 'last'
                ELSE 'middle'
            END AS segment
        FROM streams s
        JOIN activity_durations ad ON s.activity_id = ad.activity_id
    ),
    segment_averages AS (
        SELECT
            activity_id,
            AVG(CASE WHEN segment = 'first' THEN heartrate_data ELSE NULL END) AS hr_first,
            AVG(CASE WHEN segment = 'last' THEN heartrate_data ELSE NULL END) AS hr_last
        FROM stream_segments
        GROUP BY activity_id
    )
    SELECT
        a.id AS activity_id,
        a.start_date,
        CASE
            WHEN sa.hr_first IS NULL OR sa.hr_last IS NULL THEN NULL  -- Handle NULL hr_first or hr_last
            WHEN sa.hr_first = 0 THEN NULL  -- Handle hr_first being 0
            ELSE ROUND((sa.hr_last - sa.hr_first) * 100.0 / sa.hr_first, 1)
        END AS hr_drift_pct
    FROM segment_averages sa
    JOIN activities a ON sa.activity_id = a.id
    WHERE a.moving_time >= 2700
    AND a.type = 'Run'
    ORDER BY a.start_date DESC
    LIMIT 90;
    """, conn)
    conn.close()
    return df

def get_cadence_stability_data(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
        a.id AS activity_id,
        a.average_speed * 3.6 AS avg_pace_kmh,
        COALESCE(s.avg_cadence, 0) AS avg_cadence,  -- Handle NULL
        COALESCE(s.cadence_stdev, 0) AS cadence_stdev,
        COALESCE(
            CASE  -- Calculate CV here
                WHEN s.avg_cadence > 0 THEN (s.cadence_stdev * 100.0 / s.avg_cadence)
                ELSE 0  -- Or NULL, depending on how you want to handle it
            END,
            0
        ) AS cadence_cv
    FROM activities a

    LEFT JOIN (  -- Use LEFT JOIN
        SELECT
            activity_id,
            AVG(CAST(cadence_data AS INTEGER)) AS avg_cadence,
            SUM(cadence_data) AS sum_cadence,
            COUNT(cadence_data) AS count_cadence,
            AVG(cadence_data * cadence_data) - AVG(cadence_data) * AVG(cadence_data) AS variance_cadence,
            SQRT(ABS(AVG(cadence_data * cadence_data) - AVG(cadence_data) * AVG(cadence_data))) AS cadence_stdev  -- Use ABS
        FROM streams
        WHERE cadence_data IS NOT NULL  -- Use IS NOT NULL
        GROUP BY activity_id
    ) s ON a.id = s.activity_id
    WHERE a.type = 'Run'
    ORDER BY a.start_date DESC
    LIMIT 90;
    """, conn)
    conn.close()
    return df
