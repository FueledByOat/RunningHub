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

# def get_efficiency_data(db_path=DB_PATH):
#     """
    
#     """
#     try:
#         conn = sqlite3.connect(db_path)
        
#         # Get streams data
#         streams_df = pd.read_sql_query("""
#             SELECT 
#                 s.time_data,
#                 s.distance_data,
#                 s.heartrate_data
#             FROM streams s
#             limit 90
#         """, conn)
        
#         conn.close()
#         efficiency
#         def calculate_cadence_stability(row):
#             """
#             Improved cadence stability calculation that:
#             1. Filters out transition periods (acceleration/deceleration)
#             2. Normalizes cadence relative to velocity
#             3. Uses coefficient of variation as stability metric
#             """
#             if pd.isna(row['cadence_data']):
#                 return pd.Series([0, 0, 0])
                
#             try:
#                 cadence_data = json.loads(row['cadence_data'])
#                 velocity_data = json.loads(row['velocity_smooth_data']) if not pd.isna(row['velocity_smooth_data']) else None
                
#                 # Filter out zero values
#                 cadence_data = [c for c in cadence_data if c > 0]
                
#                 if not cadence_data:
#                     return pd.Series([0, 0, 0])
                
#                 # If velocity data available, analyze cadence only during stable pace sections
#                 if velocity_data and len(velocity_data) == len(cadence_data):
#                     stable_indices = []
                    
#                     # Calculate velocity rolling average
#                     window_size = min(5, len(velocity_data) - 1)
#                     rolling_velocity = []
                    
#                     for i in range(len(velocity_data) - window_size + 1):
#                         avg = sum(velocity_data[i:i+window_size]) / window_size
#                         rolling_velocity.append(avg)
                    
#                     # Pad rolling velocity to match original length
#                     rolling_velocity = [rolling_velocity[0]] * (window_size - 1) + rolling_velocity
                    
#                     # Find stable pace segments (velocity close to rolling average)
#                     for i in range(len(velocity_data)):
#                         if velocity_data[i] > 0 and abs(velocity_data[i] - rolling_velocity[i]) / rolling_velocity[i] < 0.1:
#                             stable_indices.append(i)
                    
#                     if len(stable_indices) < 10:  # Not enough stable data points
#                         # Fallback to using all non-zero data
#                         filtered_cadence = cadence_data
#                     else:
#                         # Use only stable sections
#                         filtered_cadence = [cadence_data[i] for i in stable_indices]
#                 else:
#                     filtered_cadence = cadence_data
                
#                 # Calculate statistics
#                 avg_cadence = sum(filtered_cadence) / len(filtered_cadence)
#                 variance = sum([(c - avg_cadence) ** 2 for c in filtered_cadence]) / len(filtered_cadence)
#                 stdev = variance**0.5
#                 cv = (stdev / avg_cadence * 100) if avg_cadence > 0 else 0
                
#                 return pd.Series([avg_cadence, stdev, cv])
                
#             except (json.JSONDecodeError, ValueError, TypeError, IndexError) as e:
#                 print(f"Error processing cadence stability for activity {row['activity_id']}: {e}")
#                 return pd.Series([0, 0, 0])
        
#         # Merge datasets and calculate cadence stability
#         merged_df = activities_df.merge(streams_df, left_on='id', right_on='activity_id', how='left')
#         merged_df[['avg_cadence', 'cadence_stdev', 'cadence_cv']] = merged_df.apply(calculate_cadence_stability, axis=1)
#         merged_df['avg_pace_kmh'] = merged_df['average_speed'] * 3.6
        
#         # Clean up result dataframe
#         result_df = merged_df[['id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv']]
#         result_df = result_df.rename(columns={'id': 'activity_id'})
        
#         return result_df
        
#     except sqlite3.Error as e:
#         print(f"Database error in get_cadence_stability_data: {e}")
#         return pd.DataFrame(columns=['activity_id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv'])
#     except Exception as e:
#         print(f"Error in get_cadence_stability_data: {e}")
#         return pd.DataFrame(columns=['activity_id', 'start_date', 'avg_pace_kmh', 'avg_cadence', 'cadence_stdev', 'cadence_cv'])

def get_ctl_atl_tsb_data(db_path=DB_PATH):
    """
    Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.
    """
    try:
        conn = sqlite3.connect(db_path)
        activities_df  = pd.read_sql_query("""
        SELECT 
            id,
            moving_time,
            weighted_average_watts,
            kilojoules,
            average_heartrate,
            start_date
        FROM activities
        WHERE type = 'Run'
        ORDER BY start_date DESC
        LIMIT 90;
        """, conn)
        conn.close()
        activities_df['training_load'] = activities_df['kilojoules']

        # Compute CTL, ATL, TSB
        activities_df['CTL'] = activities_df['training_load'].ewm(span=28).mean()
        activities_df['ATL'] = activities_df['training_load'].ewm(span=7).mean()
        activities_df['TSB'] = activities_df['CTL'] - activities_df['ATL']
        return activities_df
    except sqlite3.Error as e:
        print(f"Database error in get_ctl_atl_tsb_data: {e}")
        return pd.DataFrame(columns=['id', 'CTL', 'ATL', 'TSB'])
