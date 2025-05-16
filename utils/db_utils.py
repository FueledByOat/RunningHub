# db_utils.py
"""Storing functions relating to database
cursor setup as well as queries to keep main app
clean of stray functions. """

import sqlite3
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import json
import datetime

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
    
def get_acwr_data(db_path=DB_PATH):
    """
    Calculate Acute:Chronic Workload Ratio (ACWR)
    
    ACWR represents the ratio of 7-day (acute) to 28-day (chronic) workload.
    Research suggests maintaining this ratio between 0.8-1.3 to minimize injury risk.
    """
    try:
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
    except sqlite3.Error as e:
        print(f"Database error in get_acwr_data: {e}")
        return pd.DataFrame(columns=['date', 'acute_load', 'chronic_load', 'acwr'])

def get_hr_drift_data(db_path=DB_PATH):
    """
    Calculate Heart Rate Drift
    
    HR drift measures the percentage increase in heart rate during a workout at constant intensity.
    It indicates cardiovascular efficiency and aerobic development.
    Lower values (<5%) suggest better aerobic fitness.
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Get activities and streams data
        activities_df = pd.read_sql_query("""
            SELECT id, start_date, moving_time, average_speed
            FROM activities 
            WHERE type = 'Run' AND moving_time >= 1200
            ORDER BY start_date DESC
            LIMIT 90
        """, conn)
        
        streams_df = pd.read_sql_query("""
            SELECT activity_id, time_data, heartrate_data, distance_data
            FROM streams
            WHERE activity_id IN (SELECT id FROM activities WHERE type = 'Run' AND moving_time >= 1200)
        """, conn)
        
        conn.close()
        
        def calculate_hr_drift(row):
            """
            Improved HR drift calculation that:
            1. Excludes the first 5 minutes (warm-up period)
            2. Compares first and second half of the remaining workout
            3. Filters out stops/pauses (using distance data)
            """
            if pd.isna(row['time_data']) or pd.isna(row['heartrate_data']):
                return None
                
            try:
                time_data = json.loads(row['time_data'])
                hr_data = json.loads(row['heartrate_data'])
                distance_data = json.loads(row['distance_data']) if not pd.isna(row['distance_data']) else None
                
                if not time_data or not hr_data or len(time_data) != len(hr_data):
                    return None
                
                # Filter out warm-up (first 5 minutes or 300 seconds)
                warm_up_cutoff = 300
                start_idx = 0
                
                for i, t in enumerate(time_data):
                    if t >= warm_up_cutoff:
                        start_idx = i
                        break
                
                # If workout is too short after removing warm-up, return None
                if start_idx >= len(time_data) - 10:
                    return None
                
                filtered_time = time_data[start_idx:]
                filtered_hr = hr_data[start_idx:]
                
                # Identify moving segments if distance data available
                if distance_data and len(distance_data) == len(time_data):
                    filtered_distance = distance_data[start_idx:]
                    moving_indices = []
                    
                    # Find segments where runner is moving (distance increasing)
                    for i in range(1, len(filtered_distance)):
                        if filtered_distance[i] > filtered_distance[i-1]:
                            moving_indices.append(i)
                    
                    if len(moving_indices) < 10:  # Not enough data points
                        return None
                        
                    # Filter to only moving segments
                    filtered_hr = [filtered_hr[i] for i in moving_indices]
                    filtered_time = [filtered_time[i] for i in moving_indices]
                
                # Split into first and second half
                midpoint = len(filtered_time) // 2
                first_half_hr = filtered_hr[:midpoint]
                second_half_hr = filtered_hr[midpoint:]
                
                if len(first_half_hr) < 5 or len(second_half_hr) < 5:
                    return None
                
                avg_hr_first = sum(first_half_hr) / len(first_half_hr)
                avg_hr_second = sum(second_half_hr) / len(second_half_hr)
                
                if avg_hr_first <= 0:
                    return None
                
                hr_drift_pct = (avg_hr_second - avg_hr_first) * 100.0 / avg_hr_first
                return round(hr_drift_pct, 1)
                
            except (json.JSONDecodeError, ValueError, TypeError, IndexError) as e:
                print(f"Error processing HR drift for activity {row['activity_id']}: {e}")
                return None
        
        # Merge datasets and calculate drift
        merged_df = activities_df.merge(streams_df, left_on='id', right_on='activity_id', how='left')
        merged_df['hr_drift_pct'] = merged_df.apply(calculate_hr_drift, axis=1)
        
        # Clean up result dataframe
        result_df = merged_df[['id', 'start_date', 'hr_drift_pct']].dropna()
        result_df = result_df.rename(columns={'id': 'activity_id'})
        
        return result_df
        
    except sqlite3.Error as e:
        print(f"Database error in get_hr_drift_data: {e}")
        return pd.DataFrame(columns=['activity_id', 'start_date', 'hr_drift_pct'])
    except Exception as e:
        print(f"Error in get_hr_drift_data: {e}")
        return pd.DataFrame(columns=['activity_id', 'start_date', 'hr_drift_pct'])

def get_cadence_stability_data(db_path=DB_PATH):
    """
    Calculate Cadence Stability (using coefficient of variation)
    
    Cadence stability measures the consistency of running cadence relative to pace.
    Lower values indicate better running economy and neuromuscular control.
    Values under 4% suggest excellent running form consistency.
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Get activities and streams data
        activities_df = pd.read_sql_query("""
            SELECT id, average_speed, start_date, moving_time
            FROM activities 
            WHERE type = 'Run' AND moving_time >= 600
            ORDER BY start_date DESC
            LIMIT 90
        """, conn)
        
        streams_df = pd.read_sql_query("""
            SELECT activity_id, cadence_data, velocity_smooth_data
            FROM streams
            WHERE activity_id IN (SELECT id FROM activities WHERE type = 'Run' AND moving_time >= 600)
        """, conn)
        
        conn.close()
        
        def calculate_cadence_stability(row):
            """
            Improved cadence stability calculation that:
            1. Filters out transition periods (acceleration/deceleration)
            2. Normalizes cadence relative to velocity
            3. Uses coefficient of variation as stability metric
            """
            if pd.isna(row['cadence_data']):
                return pd.Series([0, 0, 0])
                
            try:
                cadence_data = json.loads(row['cadence_data'])
                velocity_data = json.loads(row['velocity_smooth_data']) if not pd.isna(row['velocity_smooth_data']) else None
                
                # Filter out zero values
                cadence_data = [c for c in cadence_data if c > 0]
                
                if not cadence_data:
                    return pd.Series([0, 0, 0])
                
                # If velocity data available, analyze cadence only during stable pace sections
                if velocity_data and len(velocity_data) == len(cadence_data):
                    stable_indices = []
                    
                    # Calculate velocity rolling average
                    window_size = min(5, len(velocity_data) - 1)
                    rolling_velocity = []
                    
                    for i in range(len(velocity_data) - window_size + 1):
                        avg = sum(velocity_data[i:i+window_size]) / window_size
                        rolling_velocity.append(avg)
                    
                    # Pad rolling velocity to match original length
                    rolling_velocity = [rolling_velocity[0]] * (window_size - 1) + rolling_velocity
                    
                    # Find stable pace segments (velocity close to rolling average)
                    for i in range(len(velocity_data)):
                        if velocity_data[i] > 0 and abs(velocity_data[i] - rolling_velocity[i]) / rolling_velocity[i] < 0.1:
                            stable_indices.append(i)
                    
                    if len(stable_indices) < 10:  # Not enough stable data points
                        # Fallback to using all non-zero data
                        filtered_cadence = cadence_data
                    else:
                        # Use only stable sections
                        filtered_cadence = [cadence_data[i] for i in stable_indices]
                else:
                    filtered_cadence = cadence_data
                
                # Calculate statistics
                avg_cadence = sum(filtered_cadence) / len(filtered_cadence)
                variance = sum([(c - avg_cadence) ** 2 for c in filtered_cadence]) / len(filtered_cadence)
                stdev = variance**0.5
                cv = (stdev / avg_cadence * 100) if avg_cadence > 0 else 0
                
                return pd.Series([avg_cadence, stdev, cv])
                
            except (json.JSONDecodeError, ValueError, TypeError, IndexError) as e:
                print(f"Error processing cadence stability for activity {row['activity_id']}: {e}")
                return pd.Series([0, 0, 0])
        
        # Merge datasets and calculate cadence stability
        merged_df = activities_df.merge(streams_df, left_on='id', right_on='activity_id', how='left')
        merged_df[['avg_cadence', 'cadence_stdev', 'cadence_cv']] = merged_df.apply(calculate_cadence_stability, axis=1)
        merged_df['avg_pace_kmh'] = merged_df['average_speed'] * 3.6
        
        # Clean up result dataframe
        result_df = merged_df[['id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv']]
        result_df = result_df.rename(columns={'id': 'activity_id'})
        
        return result_df
        
    except sqlite3.Error as e:
        print(f"Database error in get_cadence_stability_data: {e}")
        return pd.DataFrame(columns=['activity_id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv'])
    except Exception as e:
        print(f"Error in get_cadence_stability_data: {e}")
        return pd.DataFrame(columns=['activity_id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv'])

def get_efficiency_index(db_path=DB_PATH):
    """
    Efficiency Factor (EF) Calculation and Usage

    Definition:
    Efficiency Factor (EF) = Speed (meters/minute) / Heart Rate (bpm)

    Purpose:
    EF measures cardiovascular efficiency by showing how much speed a runner generates per heartbeat.
    Higher values indicate better aerobic fitness and running economy.

    Key Metrics:
    1. Single-activity EF: Snapshot of efficiency for a specific run
    2. Rolling average EF (7/30/90 day): Shows trends in efficiency over time
    3. Efficiency Index: Normalized EF that accounts for different paces

    Best Practices:
    - Compare EF values from similar workouts (similar pace/terrain)
    - Monitor long-term trends in the 30 and 90-day averages
    - Use flat-terrain EF (commented implementation) for most accurate comparisons
    - Increases in EF over time indicate improving aerobic fitness
    - Sudden drops may indicate fatigue, illness, or overtraining

    Limitations:
    - Influenced by environmental factors (heat, humidity, elevation)
    - Not directly comparable between different types of terrain
    - Heart rate can be affected by factors other than fitness (medication, stress, etc.)

    Implementation Notes:
    The provided code calculates basic EF and rolling averages. The commented section shows how
    to implement terrain-specific (flat segments only) EF calculations for more precise comparisons.
    """

    conn = sqlite3.connect(db_path)
    
    # Retrieve activities with distance, time and heart rate data
    query = """
    SELECT 
        id,
        distance,         -- in meters
        moving_time,      -- in seconds
        average_heartrate,
        start_date,
        average_speed,    -- in m/s
        type
    FROM activities
    WHERE 
        type = 'Run' 
        AND average_heartrate IS NOT NULL 
        AND average_heartrate > 0
        AND distance IS NOT NULL
        AND distance > 0
        AND moving_time IS NOT NULL
        AND moving_time > 0
    ORDER BY start_date ASC;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert start_date to datetime
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Calculate speed in meters per minute
    df['speed_mpm'] = df['average_speed'] * 60  # convert m/s to m/min
    
    # Calculate Efficiency Factor
    df['efficiency_factor'] = df['speed_mpm'] / df['average_heartrate']
    
    # Calculate efficiency factor for flat segments only (advanced version)
    # This requires accessing the grade_smooth_data from the streams table

    conn = sqlite3.connect(db_path)
    flat_ef_values = []
    
    for idx, row in df.iterrows():
        activity_id = row['id']
        
        # Get grade data for this activity
        grade_query = "SELECT grade_smooth_data FROM streams WHERE activity_id = ?"
        grade_result = conn.execute(grade_query, (activity_id,)).fetchone()
        
        if grade_result and grade_result[0]:
            # Process grade data to find flat segments
            grade_data = json.loads(grade_result[0])
            
            # Get velocity and heartrate data
            velocity_query = "SELECT velocity_smooth_data FROM streams WHERE activity_id = ?"
            hr_query = "SELECT heartrate_data FROM streams WHERE activity_id = ?"
            
            velocity_result = conn.execute(velocity_query, (activity_id,)).fetchone()
            hr_result = conn.execute(hr_query, (activity_id,)).fetchone()
            
            if velocity_result and velocity_result[0] and hr_result and hr_result[0]:
                velocity_data = json.loads(velocity_result[0])
                hr_data = json.loads(hr_result[0])
                
                # Find flat segments (grade between -1% and 1%)
                flat_indices = [i for i, grade in enumerate(grade_data) if -1 <= grade <= 1]
                
                if flat_indices:
                    # Calculate EF for flat segments only
                    flat_velocity = [velocity_data[i] * 60 for i in flat_indices if i < len(velocity_data)]
                    flat_hr = [hr_data[i] for i in flat_indices if i < len(hr_data)]
                    
                    if flat_velocity and flat_hr:
                        avg_flat_velocity = sum(flat_velocity) / len(flat_velocity)
                        avg_flat_hr = sum(flat_hr) / len(flat_hr)
                        flat_ef = avg_flat_velocity / avg_flat_hr if avg_flat_hr > 0 else None
                        flat_ef_values.append(flat_ef)
                    else:
                        flat_ef_values.append(None)
                else:
                    flat_ef_values.append(None)
            else:
                flat_ef_values.append(None)
        else:
            flat_ef_values.append(None)
    
    df['flat_efficiency_factor'] = flat_ef_values
    conn.close()

    
    # Calculate rolling averages (7-day, 30-day, 90-day)
    df['ef_7day'] = df['efficiency_factor'].rolling(window=7, min_periods=3).mean()
    df['ef_30day'] = df['efficiency_factor'].rolling(window=30, min_periods=7).mean()
    df['ef_90day'] = df['efficiency_factor'].rolling(window=90, min_periods=14).mean()
    
    # Calculate Efficiency Index (normalized for pace)
    # This helps compare EF across different paces
    df['norm_pace'] = 1000 / df['average_speed']  # Pace in seconds per km
    pace_factor = df['norm_pace'] / 300  # Normalize against a 5:00/km pace (300 seconds)
    df['efficiency_index'] = df['efficiency_factor'] * pace_factor
    
    return df

def calculate_running_tss(moving_time, avg_hr=None, max_hr=None, threshold_hr=None):
    """
    Calculate a Training Stress Score (TSS) for running activities.
    
    This uses a simplified approach based on duration and heart rate data when available.
    If heart rate data is not available, it falls back to a duration-based estimate.
    
    Args:
        moving_time: Duration in seconds
        avg_hr: Average heart rate during activity
        max_hr: Maximum heart rate during activity
        threshold_hr: Threshold heart rate (lactate threshold)
        
    Returns:
        Estimated TSS value
    """
    # Convert moving time to hours
    hours = moving_time / 3600
    
    # If we have heart rate data and threshold heart rate
    if avg_hr and threshold_hr:
        # Calculate heart rate as percentage of threshold
        hr_ratio = avg_hr / threshold_hr
        # Calculate intensity factor based on heart rate ratio
        intensity = hr_ratio ** 1.05
        # TSS formula: duration (hours) * intensity^2 * 100
        tss = hours * intensity * intensity * 100
    else:
        # Simple duration-based estimate if no HR data
        # Assuming moderate intensity (IF of ~0.75-0.85)
        tss = hours * 0.8 * 0.8 * 100
    
    return tss

def get_ctl_atl_tsb_tss_data(db_path=DB_PATH, days_to_retrieve=180, athlete_threshold_hr=None):
    """
    Calculate CTL (Chronic Training Load), ATL (Acute Training Load), and TSB (Training Stress Balance)
    using proper time-based exponential decay.
    
    Args:
        db_path: Path to SQLite database
        days_to_retrieve: Number of days of history to include (recommended minimum 120 days)
        athlete_threshold_hr: Athlete's threshold heart rate for TSS calculations
        
    Returns:
        DataFrame with daily CTL, ATL, and TSB values
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Get activities with relevant metrics
        activities_df = pd.read_sql_query("""
        SELECT
            id,
            moving_time,
            weighted_average_watts,
            kilojoules,
            average_heartrate,
            max_heartrate,
            start_date
        FROM activities
        WHERE type = 'Run'
        AND start_date >= date('now', '-{} days')
        ORDER BY start_date ASC
        """.format(days_to_retrieve), conn)
        
        conn.close()
        
        # Convert start_date to datetime and set as index
        activities_df['start_date'] = pd.to_datetime(activities_df['start_date'])
        
        # Calculate TSS for each activity
        activities_df['tss'] = activities_df.apply(
            lambda row: calculate_running_tss(
                row['moving_time'], 
                row['average_heartrate'],
                row['max_heartrate'],
                athlete_threshold_hr
            ), 
            axis=1
        )
        
        # Get date range covering all activities plus buffer days
        if not activities_df.empty:
            start_date = activities_df['start_date'].min().date() - datetime.timedelta(days=42)
            end_date = activities_df['start_date'].max().date() + datetime.timedelta(days=7)
            
            # Create a continuous date range dataframe
            date_range = pd.date_range(start=start_date, end=end_date)
            daily_df = pd.DataFrame(index=date_range)
            daily_df.index.name = 'date'
            
            # Group activities by day and sum TSS
            activities_df['date'] = activities_df['start_date'].dt.date
            daily_tss = activities_df.groupby('date')['tss'].sum().reset_index()
            daily_tss['date'] = pd.to_datetime(daily_tss['date'])
            daily_tss = daily_tss.set_index('date')
            
            # Merge into daily dataframe
            daily_df = daily_df.join(daily_tss)
            daily_df['tss'].fillna(0, inplace=True)
            
            # Time constants (in days)
            ctl_time_constant = 42
            atl_time_constant = 7
            
            # Calculate CTL and ATL using proper exponential decay
            # CTL formula: Today's CTL = Yesterday's CTL + (Today's TSS - Yesterday's CTL) / CTL time constant
            # ATL formula: Today's ATL = Yesterday's ATL + (Today's TSS - Yesterday's ATL) / ATL time constant
            
            daily_df['CTL'] = 0
            daily_df['ATL'] = 0
            
            for i in range(1, len(daily_df)):
                yesterday = daily_df.index[i-1]
                today = daily_df.index[i]
                
                # Calculate CTL with proper exponential decay
                daily_df.at[today, 'CTL'] = (
                    daily_df.at[yesterday, 'CTL'] + 
                    (daily_df.at[today, 'tss'] - daily_df.at[yesterday, 'CTL']) / ctl_time_constant
                )
                
                # Calculate ATL with proper exponential decay
                daily_df.at[today, 'ATL'] = (
                    daily_df.at[yesterday, 'ATL'] + 
                    (daily_df.at[today, 'tss'] - daily_df.at[yesterday, 'ATL']) / atl_time_constant
                )
            
            # Calculate TSB (Form) = CTL - ATL
            daily_df['TSB'] = daily_df['CTL'] - daily_df['ATL']
            
            # Return only the most recent N days to keep the result set manageable
            return daily_df.tail(90).reset_index()
        else:
            return pd.DataFrame(columns=['date', 'tss', 'CTL', 'ATL', 'TSB'])
            
    except sqlite3.Error as e:
        print(f"Database error in get_ctl_atl_tsb_data: {e}")
        return pd.DataFrame(columns=['date', 'tss', 'CTL', 'ATL', 'TSB'])