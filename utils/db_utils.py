# db_utils.py
"""Storing functions relating to database
cursor setup as well as queries to keep main app
clean of stray functions. """

import sqlite3
from dotenv import load_dotenv
import os
import pandas as pd
import json

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
    """Ratio of 7-day to 28-day load; balance between acute and chronic load."""
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
    "Percentage HR increase during session; indicates aerobic efficiency."
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT id, start_date, moving_time FROM activities WHERE type = 'Run' LIMIT 90", conn)
        streams_df = pd.read_sql_query("SELECT activity_id, time_data, heartrate_data FROM streams", conn)
        conn.close()

        def calculate_hr_drift(row):
            if row['moving_time'] < 2700:
                return None
            if pd.isna(row['time_data']):
                return None
            time_data = json.loads(row['time_data']) if row['time_data'] else []
            heartrate_data = json.loads(row['heartrate_data']) if row['heartrate_data'] else []

            if not time_data or not heartrate_data or len(time_data) != len(heartrate_data):
                return None

            total_duration = time_data[-1] - time_data[0]
            if total_duration == 0:
                return None

            first_segment_hr = []
            last_segment_hr = []

            for i in range(len(time_data)):
                relative_time = time_data[i] - time_data[0]
                if relative_time <= total_duration * 0.33:
                    first_segment_hr.append(heartrate_data[i])
                elif relative_time >= total_duration * 0.66:
                    last_segment_hr.append(heartrate_data[i])

            if not first_segment_hr or not last_segment_hr:
                return None

            avg_hr_first = sum(first_segment_hr) / len(first_segment_hr)
            avg_hr_last = sum(last_segment_hr) / len(last_segment_hr)

            if avg_hr_first == 0:
                return None

            return round((avg_hr_last - avg_hr_first) * 100.0 / avg_hr_first, 1)

        merged_df = df.merge(streams_df, left_on='id', right_on='activity_id', how='left')
        df['hr_drift_pct'] = merged_df.apply(calculate_hr_drift, axis=1)
        df = df[['id', 'start_date', 'hr_drift_pct']].dropna()  # Keep relevant columns and drop NaNs
        return df.rename(columns={'id': 'activity_id'})

    except sqlite3.Error as e:
        print(f"Database error in get_hr_drift_data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in get_hr_drift_data: {e}")
        return pd.DataFrame()

def get_cadence_stability_data(db_path = DB_PATH):
    "Coefficient of variation of cadence relative to pace; running economy."
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT id, average_speed, start_date FROM activities WHERE type = 'Run' LIMIT 90", conn)
        streams_df = pd.read_sql_query("SELECT activity_id, cadence_data FROM streams", conn)
        conn.close()

        def calculate_cadence_stats(row):
            if pd.isna(row['cadence_data']):
                return pd.Series([0, 0])
            cadence_data = json.loads(row['cadence_data']) if row['cadence_data'] else []
            cadence_data = [c for c in cadence_data if c > 0]  # Filter out 0s

            if not cadence_data:
                return pd.Series([0, 0])  # Or pd.Series([None, None]) if you prefer NULL

            avg_cadence = sum(cadence_data) / len(cadence_data)
            variance = sum([(c - avg_cadence) ** 2 for c in cadence_data]) / len(cadence_data)
            stdev = variance**0.5

            return pd.Series([avg_cadence, stdev])

        merged_df = df.merge(streams_df, left_on='id', right_on='activity_id', how='left')
        df[['avg_cadence', 'cadence_stdev']] = merged_df.apply(calculate_cadence_stats, axis=1)
        df['avg_pace_kmh'] = df['average_speed'] * 3.6
        df['cadence_cv'] = df.apply(
            lambda row: (row['cadence_stdev'] / row['avg_cadence'] * 100 if row['avg_cadence'] > 0 else 0),
            axis=1
        )

        return df[['id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv']].rename(
            columns={'id': 'activity_id'}
        )

    except sqlite3.Error as e:
        print(f"Database error in get_cadence_stability_data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in get_cadence_stability_data: {e}")
        return pd.DataFrame()
