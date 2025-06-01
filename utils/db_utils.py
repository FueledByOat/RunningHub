# db_utils.py
"""Database utilities for activity data queries and connection management."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import datetime
from threading import Lock
from queue import Queue, Empty

import pandas as pd
import numpy as np

from config import Config
from utils import exception_utils

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.pool = Queue(maxsize=max_connections)
        self.lock = Lock()
        
    def _create_connection(self):
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = dict_factory
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool or create new one."""
        try:
            conn = self.pool.get_nowait()
        except Empty:
            conn = self._create_connection()
            
        try:
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database operation failed: {e}")
            conn.rollback()
            raise exception_utils.DatabaseError(f"Database error: {e}") from e
        finally:
            try:
                self.pool.put_nowait(conn)
            except:
                conn.close()


# Global connection pool
_connection_pool = None

def get_connection_pool(db_path: str) -> ConnectionPool:
    """Get or create global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        if not db_path:
            raise exception_utils.DatabaseError("Database path not configured")
        _connection_pool = ConnectionPool(db_path)
    return _connection_pool


@contextmanager
def get_db_connection(db_path: str):
    """Context manager for pooled database connections."""
    pool = get_connection_pool(db_path)
    with pool.get_connection() as conn:
        yield conn


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> Dict[str, Any]:
    """Convert database row objects to a dictionary.
    
    Args:
        cursor: Database cursor
        row: Database row
        
    Returns:
        Dictionary representation of the row
    """
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def dict_from_row(row) -> Dict:
    """Convert sqlite3.Row to dictionary"""
    return dict(row) if row else {}

def dicts_from_rows(rows) -> List[Dict]:
    """Convert list of sqlite3.Row to list of dictionaries"""
    return [dict(row) for row in rows] if rows else []

# -------------------------------------
# Activity Page SQL Logic
# -------------------------------------


def get_activity_details_by_id(conn, activity_id: int, activity_types: List[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve all activity details for given activity ID.
    
    Args:
        conn: Pooled context managed from db_utils, defined in base_service and 
        typically referenced in service by _get_connection
        activity_id: 
        activity_types: List of activity types to filter by (default: ["Run", "Ride"])
    Returns:
        Latest activity ID or None if not found
        
    Raises:
        DatabaseError: If database query fails
    """
    if activity_types is None:
        activity_types = ["Run"]
    
    # Using dynamic placeholders here as the number of parameters is variable
    placeholders = ",".join("?" * len(activity_types))
    conn.row_factory = sqlite3.Row
    query = f"""
        SELECT a.*, COALESCE(CONCAT(g.model_name, " ", g.nickname), a.gear_id) as gear_name
        FROM activities as a
        LEFT JOIN gear as g ON a.gear_id = g.gear_id
        WHERE a.id = ?
        AND a.gear_id IS NOT NULL AND a.gear_id != ''
        AND a.type IN ({placeholders})
        ORDER BY a.start_date DESC LIMIT 1
    """
    try:
        cur = conn.cursor()
        cur.execute(query, [activity_id] + activity_types,)
        result = cur.fetchone()
        
        if result:
            return dict(result)
        
        else:
            logger.warning(f"No activities found for activity ID: {activity_id}")
            return None    
                
    except exception_utils.DatabaseError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting activity details: {e}")
        raise exception_utils.DatabaseError(f"Failed to get activity details: {e}") from e

def get_latest_activity_id(conn, activity_types: List[str] = None) -> Optional[int]:
    """Retrieve latest activity ID by type.
    
    Args:
        conn: Pooled context managed from db_utils, defined in base_service and 
        typically referenced in service by _get_connection
        activity_types: List of activity types to filter by (default: ["Run", "Ride"])
        
    Returns:
        Latest activity ID or None if not found
        
    Raises:
        DatabaseError: If database query fails
    """
    if activity_types is None:
        activity_types = ["Run", "Ride"]
    
    # Using dynamic placeholders here as the number of parameters is variable
    placeholders = ",".join("?" * len(activity_types))
    query = f"""
        SELECT id FROM activities
        WHERE type IN ({placeholders})
        ORDER BY start_date DESC 
        LIMIT 1
    """
    
    try:
        cur = conn.cursor()
        cur.execute(query, activity_types)
        row = cur.fetchone()
        
        if row:
            return row['id']
        else:
            logger.warning(f"No activities found for types: {activity_types}")
            return None
                
    except exception_utils.DatabaseError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting latest activity: {e}")
        raise exception_utils.DatabaseError(f"Failed to get latest activity: {e}") from e

# -------------------------------------
# Statistics Page SQL Logic
# -------------------------------------

def get_total_activities_count(conn: sqlite3.Connection, start_date: str) -> int:
    """Get count of run activities since start_date."""
    result = conn.execute(
        "SELECT COUNT(*) as count FROM activities WHERE start_date >= ? AND type = 'Run'",
        (start_date,)
    ).fetchone()
    return result['count'] or 0

def get_total_elevation_count(conn: sqlite3.Connection, start_date: str) -> int:
    """Get sum of elevation for run activities since start_date."""
    result = conn.execute(
            "SELECT SUM(total_elevation_gain) as elevation_sum FROM activities WHERE start_date >= ? AND type = 'Run'",
            (start_date,)).fetchone()
    return result['elevation_sum'] or 0

def get_total_distance_count(conn: sqlite3.Connection, start_date: str) -> int:
    """Get sum of distance of run activities since start_date."""
    result = conn.execute(
            "SELECT SUM(distance) as total FROM activities WHERE start_date >= ? AND type = 'Run'",
            (start_date,)
        ).fetchone()
    return result['total'] or 0

def get_total_time(conn: sqlite3.Connection, start_date: str) -> int:
    """Get sum of run time for activities since start_date."""
    result = conn.execute(
            "SELECT SUM(moving_time) as total FROM activities WHERE start_date >= ? AND type = 'Run'",
            (start_date,)
        ).fetchone()
    return result['total'] or 0

def get_total_calories(conn: sqlite3.Connection, start_date: str) -> int:
    """Get sum of kilojoules burned for run activities since start_date."""
    result = conn.execute(
            "SELECT SUM(kilojoules) as total FROM activities WHERE start_date >= ? AND type = 'Run'",
            (start_date,)
        ).fetchone()
    return result['total'] or 0

def get_weekly_distances(conn: sqlite3.Connection, seven_days_ago: str) -> int:
    """Get distance data for the last 7 days."""
    result = conn.execute(
            """SELECT start_date_local, distance 
               FROM activities 
               WHERE start_date_local >= ? AND type = 'Run'
               ORDER BY start_date_local""",
            (seven_days_ago.strftime('%Y-%m-%d'),)
        ).fetchall()
    return result

def get_pace_trends(conn: sqlite3.Connection) -> int:
    """Get pace trends for the last 10 activities."""
    result = conn.execute(
            """SELECT start_date_local, distance, moving_time 
               FROM activities 
               WHERE type = 'Run' AND distance > 0 AND moving_time > 0
               ORDER BY start_date_local DESC 
               LIMIT 10"""
        ).fetchall()
    return result

def get_shoe_usage(conn: sqlite3.Connection, start_date: str) -> int:
    """Get shoe usage data."""
    result = conn.execute(
            """SELECT COALESCE(CONCAT(g.model_name, " ", g.nickname), a.gear_id) as gear_id, 
                      COUNT(*) as activities,
                      SUM(a.distance) as total_distance,
                      MAX(start_date_local) as last_used
               FROM activities as a
               LEFT JOIN gear as g ON a.gear_id = g.gear_id
               WHERE a.gear_id IS NOT NULL AND a.gear_id != ''
               GROUP BY a.gear_id
               HAVING MAX(start_date_local) >= ?
               ORDER BY last_used DESC""",
            (start_date,)
        ).fetchall()
    return result

def get_recent_activities(conn: sqlite3.Connection) -> int:
    """Get the most recent 5 activities data."""
    result = conn.execute(
            """SELECT id, name, distance, moving_time, start_date_local
               FROM activities
               WHERE type = 'Run'
               ORDER BY start_date_local DESC
               LIMIT 5"""
        ).fetchall()
    return result

# -------------------------------------
# Statistics Page SQL Logic END
# -------------------------------------

# -------------------------------------
# Trophy Page SQL Logic 
# -------------------------------------

def get_distance_record(conn: sqlite3.Connection, distance_name: str, 
                    min_distance: int, max_distance: int, units: str) -> Optional[Dict[str, Any]]:
    """Get personal record for a specific distance range."""
    result = conn.execute(
        """SELECT id, name, distance, moving_time, start_date_local
            FROM activities
            WHERE type = 'Run' AND distance BETWEEN ? AND ?
            ORDER BY moving_time ASC
            LIMIT 1""",
        (min_distance, max_distance)
    ).fetchone()
    return result or None

def get_longest_run(conn: sqlite3.Connection, units: str) -> Optional[Dict[str, Any]]:
    """Get longest run record."""
    result = conn.execute(
            """SELECT id, name, distance, moving_time, start_date_local
            FROM activities
            WHERE type = 'Run'
            ORDER BY distance DESC
            LIMIT 1"""
        ).fetchone()
    return result or None


# -------------------------------------
# Trophy Page SQL Logic END
# -------------------------------------

def get_activity_polyline(activity_id: int, db_path: str = Config.DB_PATH) -> Optional[str]:
    """Retrieve polyline for specified activity.
    
    Args:
        activity_id: Activity identifier
        db_path: Database file path
        
    Returns:
        Polyline string or None if not found
        
    Raises:
        DatabaseError: If database query fails
    """
    query = "SELECT map_summary_polyline FROM activities WHERE id = ?"
    
    try:
        with get_db_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (activity_id,))
            row = cur.fetchone()
            
            if row:
                return row['map_summary_polyline']
            else:
                logger.warning(f"No polyline found for activity {activity_id}")
                return None
                
    except exception_utils.DatabaseError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting polyline for activity {activity_id}: {e}")
        raise exception_utils.DatabaseError(f"Failed to get polyline: {e}") from e

def get_activities_by_type(activity_type: str, limit: int = 10, db_path: str = Config.DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve activities by type.
    
    Args:
        activity_type: Type of activity to retrieve
        limit: Maximum number of activities to return
        db_path: Database file path
        
    Returns:
        List of activity dictionaries
    """
    query = """
        SELECT * FROM activities 
        WHERE type = ? 
        ORDER BY start_date DESC 
        LIMIT ?
    """
    
    try:
        with get_db_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (activity_type, limit))
            return cur.fetchall()
            
    except Exception as e:
        logger.error(f"Failed to get activities of type {activity_type}: {e}")
        raise exception_utils.DatabaseError(f"Failed to get activities: {e}") from e
    
def get_acwr_data(db_path=Config.DB_PATH):
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

def get_hr_drift_data(db_path=Config.DB_PATH):
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

def get_cadence_stability_data(db_path=Config.DB_PATH):
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

def get_efficiency_index(db_path=Config.DB_PATH):
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
    df.set_index('start_date', inplace=True)
    df.sort_index(inplace=True)
    df['ef_7day'] = df['efficiency_factor'].rolling('7D', min_periods=3).mean()
    df['ef_30day'] = df['efficiency_factor'].rolling('30D', min_periods=7).mean()
    df['ef_90day'] = df['efficiency_factor'].rolling('90D', min_periods=14).mean()
    
    # Calculate Efficiency Index (normalized for pace)
    # This helps compare EF across different paces
    df['efficiency_index'] = (df['efficiency_factor'] - df['efficiency_factor'].mean()) / df['efficiency_factor'].std()

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
    if avg_hr is not None and threshold_hr:
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

def get_ctl_atl_tsb_tss_data(db_path=Config.DB_PATH, days_to_retrieve=180, athlete_threshold_hr=172):
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
            # Remove timezone and floor to day
            activities_df['date'] = activities_df['start_date'].dt.tz_localize(None).dt.floor('D')

            # Group by date
            daily_tss = activities_df.groupby('date')['tss'].sum()

            # Build full date range using Timestamp values
            start_date = activities_df['date'].min() - datetime.timedelta(days=42)
            end_date = activities_df['date'].max() + datetime.timedelta(days=7)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Create a daily DataFrame with datetime index
            daily_df = pd.DataFrame(index=date_range)
            daily_df.index.name = 'date'

            # Merge on aligned datetime indices
            daily_df = daily_df.merge(daily_tss, left_index=True, right_index=True, how='left')
            daily_df['tss'].fillna(0, inplace=True)

            # Time constants (in days)
            ctl_time_constant = 42
            atl_time_constant = 7
            
            # Calculate CTL and ATL using proper exponential decay
            # CTL formula: Today's CTL = Yesterday's CTL + (Today's TSS - Yesterday's CTL) / CTL time constant
            # ATL formula: Today's ATL = Yesterday's ATL + (Today's TSS - Yesterday's ATL) / ATL time constant

            daily_df['CTL'] = daily_tss.iloc[-1]
            daily_df['ATL'] = daily_tss.iloc[-1]
            
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
            
            today = pd.Timestamp.today().normalize()
            start_date = today - datetime.timedelta(days=90)

            # Filter from start_date up to and including today
            last_90_days = daily_df.loc[start_date:today]

            # Return only the most recent N days to keep the result set manageable
            return last_90_days.reset_index()
        else:
            return pd.DataFrame(columns=['date', 'tss', 'CTL', 'ATL', 'TSB'])
            
    except sqlite3.Error as e:
        print(f"Database error in get_ctl_atl_tsb_data: {e}")
        return pd.DataFrame(columns=['date', 'tss', 'CTL', 'ATL', 'TSB'])

def get_enhanced_training_shape_data(db_path: str = Config.DB_PATH) -> pd.DataFrame:
    """
    Calculate enhanced Training Shape metrics from running activity history.
    
    Combines fitness (CTL), fatigue (ATL), form (TSB), speed trends, and efficiency 
    into a composite training readiness score. Uses proper TSS calculation with 
    intensity factors and exponential weighted moving averages.
    
    Args:
        db_path (str): Path to SQLite database containing activities table
        
    Returns:
        pd.DataFrame: Weekly aggregated data with columns:
            - training_shape (float): Composite score 0-100
            - ctl (float): Chronic Training Load (fitness)  
            - atl (float): Acute Training Load (fatigue)
            - tsb (float): Training Stress Balance (form)
            - readiness (str): 'Fresh', 'Neutral', or 'Fatigued'
            - fitness_trend (float): 4-week fitness change
            - weekly_distance_km (float): Weekly volume
            - start_date (datetime): Week start date
            
    Notes:
        - Requires activities table with Run type entries
        - TSS calculated using heart rate or pace intensity factors
        - Uses 6-week CTL and 1-week ATL exponential averages
        - Adaptive normalization based on personal performance ranges
    """
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        id, distance, moving_time, average_heartrate, max_heartrate,
        average_speed, average_cadence, kilojoules, start_date
    FROM activities
    WHERE type = 'Run' AND distance IS NOT NULL AND moving_time IS NOT NULL
    ORDER BY start_date ASC;
    """
    
    activities_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if activities_df.empty:
        return pd.DataFrame()
    
    activities_df['start_date'] = pd.to_datetime(activities_df['start_date'])
    activities_df['date'] = activities_df['start_date'].dt.date
    
    # Enhanced TSS calculation
    activities_df['pace_per_km'] = activities_df['moving_time'] / (activities_df['distance'] / 1000)
    
    # Use heart rate zones if available, otherwise pace-based intensity
    if activities_df['average_heartrate'].notna().sum() > len(activities_df) * 0.5:
        # HR-based TSS (requires user's threshold HR - could be estimated)
        threshold_hr = activities_df['average_heartrate'].quantile(0.85)  # Rough estimate
        activities_df['intensity_factor'] = activities_df['average_heartrate'] / threshold_hr
    else:
        # Pace-based intensity using critical pace concept
        # Use best 5km equivalent pace as threshold
        best_5k_pace = activities_df[activities_df['distance'].between(4000, 6000)]['pace_per_km'].min()
        if pd.isna(best_5k_pace):
            best_5k_pace = activities_df['pace_per_km'].quantile(0.1)  # Fastest 10%
        
        activities_df['intensity_factor'] = best_5k_pace / activities_df['pace_per_km']
    
    # Calculate TSS with proper intensity factor
    activities_df['duration_hours'] = activities_df['moving_time'] / 3600
    activities_df['tss'] = activities_df['duration_hours'] * (activities_df['intensity_factor'] ** 2) * 100
    activities_df['tss'] = activities_df['tss'].clip(0, 500)  # Cap extreme values
    
    # Enhanced efficiency factor
    mask = (activities_df['average_heartrate'].notna() & 
            (activities_df['average_heartrate'] > 0) & 
            (activities_df['distance'] > 1000))  # Only for runs > 1km
    
    activities_df.loc[mask, 'ef'] = (
        (activities_df.loc[mask, 'distance'] / 1000) / 
        (activities_df.loc[mask, 'moving_time'] / 3600) / 
        activities_df.loc[mask, 'average_heartrate'] * 60
    )
    
    # Weekly aggregation
    activities_df['year_week'] = (activities_df['start_date'].dt.isocalendar().year.astype(str) + 
                                 '-' + activities_df['start_date'].dt.isocalendar().week.astype(str).str.zfill(2))
    
    weekly_df = activities_df.groupby('year_week').agg({
        'start_date': 'min',
        'distance': 'sum',
        'moving_time': 'sum',
        'tss': 'sum',
        'ef': 'mean',
        'intensity_factor': 'mean'
    }).reset_index()
    
    weekly_df = weekly_df.sort_values('start_date')
    
    # Performance Management Chart metrics
    weekly_df['ctl'] = weekly_df['tss'].ewm(span=42, adjust=False).mean()  # 6-week exponential
    weekly_df['atl'] = weekly_df['tss'].ewm(span=7, adjust=False).mean()   # 1-week exponential  
    weekly_df['tsb'] = weekly_df['ctl'] - weekly_df['atl']  # Training Stress Balance
    
    # Speed and efficiency trends
    weekly_df['weekly_distance_km'] = weekly_df['distance'] / 1000
    weekly_df['weekly_time_hours'] = weekly_df['moving_time'] / 3600
    weekly_df['weekly_speed_kmh'] = weekly_df['weekly_distance_km'] / weekly_df['weekly_time_hours']
    
    # Adaptive normalization based on personal ranges
    def adaptive_normalize(series, percentile_range=(5, 95)):
        min_val = series.quantile(percentile_range[0] / 100)
        max_val = series.quantile(percentile_range[1] / 100)
        return 100 * (series - min_val) / (max_val - min_val)
    
    # Component scores
    weekly_df['fitness_score'] = adaptive_normalize(weekly_df['ctl']).clip(0, 100)
    weekly_df['speed_score'] = adaptive_normalize(weekly_df['weekly_speed_kmh']).clip(0, 100)
    weekly_df['freshness_score'] = adaptive_normalize(weekly_df['tsb']).clip(0, 100)
    
    if weekly_df['ef'].notna().sum() > 5:  # Need minimum data points
        weekly_df['efficiency_score'] = adaptive_normalize(weekly_df['ef']).clip(0, 100)
        # Composite with all components
        weekly_df['training_shape'] = (
            weekly_df['fitness_score'] * 0.4 + 
            weekly_df['speed_score'] * 0.25 + 
            weekly_df['efficiency_score'] * 0.2 +
            weekly_df['freshness_score'] * 0.15
        )
    else:
        # Without efficiency
        weekly_df['training_shape'] = (
            weekly_df['fitness_score'] * 0.5 + 
            weekly_df['speed_score'] * 0.35 + 
            weekly_df['freshness_score'] * 0.15
        )
    
    # Add readiness indicator
    weekly_df['readiness'] = np.where(
        weekly_df['tsb'] > 5, 'Fresh',
        np.where(weekly_df['tsb'] < -10, 'Fatigued', 'Neutral')
    )
    
    # Add trend indicators
    window = min(4, len(weekly_df))
    weekly_df['fitness_trend'] = weekly_df['fitness_score'].diff(window).fillna(0)
    weekly_df['form_trend'] = weekly_df['training_shape'].diff(window).fillna(0)
    
    return weekly_df


def add_risk_indicators(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add injury risk and consistency indicators to weekly training data.
    
    Args:
        weekly_df (pd.DataFrame): Output from get_enhanced_training_shape_data()
        
    Returns:
        pd.DataFrame: Enhanced dataframe with additional columns:
            - volume_change_pct (float): Week-over-week distance change
            - injury_risk_flag (bool): True if >30% volume increase
            - load_ratio (float): ATL/CTL ratio for overreaching detection
            - consistency_streak (int): Consecutive weeks with training
    """
    df = weekly_df.copy()
    
    # Volume change indicators
    df['volume_change_pct'] = df['weekly_distance_km'].pct_change() * 100
    df['injury_risk_flag'] = df['volume_change_pct'] > 30
    
    # Load ratio for overreaching
    df['load_ratio'] = df['atl'] / df['ctl']
    
    # Consistency streak
    df['has_training'] = df['weekly_distance_km'] > 0
    df['consistency_streak'] = df['has_training'].cumsum() - df['has_training'].cumsum().where(~df['has_training']).ffill().fillna(0)
    
    return df


def get_dashboard_summary(weekly_df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for dashboard display.
    
    Args:
        weekly_df (pd.DataFrame): Weekly training data
        
    Returns:
        dict: Summary metrics for dashboard cards/indicators
    """
    if weekly_df.empty:
        return {}
    
    latest = weekly_df.iloc[-1]
    last_4_weeks = weekly_df.tail(4)
    
    return {
        'current_training_shape': round(latest['training_shape'], 1),
        'fitness_trend': 'Improving' if latest['fitness_trend'] > 2 else 'Declining' if latest['fitness_trend'] < -2 else 'Stable',
        'readiness': latest['readiness'],
        'avg_weekly_distance': round(last_4_weeks['weekly_distance_km'].mean(), 1),
        'injury_risk_weeks': (last_4_weeks['volume_change_pct'] > 30).sum(),
        'consistency_rate': round((weekly_df['weekly_distance_km'] > 0).mean() * 100, 1),
        'peak_fitness': round(weekly_df['training_shape'].max(), 1),
        'current_vs_peak': round((latest['training_shape'] / weekly_df['training_shape'].max()) * 100, 1)
    }

def get_cumulative_training_shape_data(db_path: str = Config.DB_PATH) -> pd.DataFrame:
    """
    Calculate cumulative Training Shape metrics representing lifetime fitness gains.
    
    Unlike current form (CTL/ATL), this tracks accumulated training adaptations
    with decay over time but no complete reset. Represents overall fitness capacity
    built through consistent training over months/years.
    
    Args:
        db_path (str): Path to SQLite database containing activities table
        
    Returns:
        pd.DataFrame: Weekly data with cumulative metrics:
            - cumulative_fitness (float): Lifetime fitness accumulation
            - training_consistency (float): Long-term adherence score
            - cumulative_volume (float): Total distance logged
            - fitness_bank (float): Composite cumulative score 0-100
            - experience_factor (float): Training maturity multiplier
    """
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        id, distance, moving_time, average_heartrate, max_heartrate,
        average_speed, start_date
    FROM activities
    WHERE type = 'Run' AND distance IS NOT NULL AND moving_time IS NOT NULL
    ORDER BY start_date ASC;
    """
    
    activities_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if activities_df.empty:
        return pd.DataFrame()
    
    activities_df['start_date'] = pd.to_datetime(activities_df['start_date'])
    
    # Calculate training value for each activity
    activities_df['pace_per_km'] = activities_df['moving_time'] / (activities_df['distance'] / 1000)
    activities_df['duration_hours'] = activities_df['moving_time'] / 3600
    
    # Estimate intensity factor
    if activities_df['average_heartrate'].notna().sum() > len(activities_df) * 0.3:
        threshold_hr = activities_df['average_heartrate'].quantile(0.85)
        activities_df['intensity_factor'] = activities_df['average_heartrate'] / threshold_hr
    else:
        best_pace = activities_df['pace_per_km'].quantile(0.1)
        activities_df['intensity_factor'] = best_pace / activities_df['pace_per_km']
    
    activities_df['intensity_factor'] = activities_df['intensity_factor'].clip(0.3, 2.0)
    
    # Training Impulse (TRIMP) - represents training stimulus
    activities_df['trimp'] = (activities_df['duration_hours'] * 
                             activities_df['intensity_factor'] * 
                             60)  # Scale to reasonable range
    
    # Weekly aggregation
    activities_df['year_week'] = (activities_df['start_date'].dt.isocalendar().year.astype(str) + 
                                 '-' + activities_df['start_date'].dt.isocalendar().week.astype(str).str.zfill(2))
    
    weekly_df = activities_df.groupby('year_week').agg({
        'start_date': 'min',
        'distance': 'sum',
        'moving_time': 'sum',
        'trimp': 'sum',
        'intensity_factor': 'mean'
    }).reset_index()
    
    weekly_df = weekly_df.sort_values('start_date').reset_index(drop=True)
    
    # Calculate weeks since start for experience factor
    start_date = weekly_df['start_date'].min()
    weekly_df['weeks_training'] = ((weekly_df['start_date'] - start_date).dt.days / 7).astype(int) + 1
    
    # Cumulative metrics with slow decay
    decay_rate = 0.998  # 2% annual decay (very slow)
    
    # Initialize cumulative fitness
    weekly_df['cumulative_fitness'] = 0.0
    weekly_df['cumulative_volume'] = weekly_df['distance'].cumsum() / 1000  # km
    
    # Calculate cumulative fitness with decay
    for i in range(len(weekly_df)):
        if i == 0:
            weekly_df.loc[i, 'cumulative_fitness'] = weekly_df.loc[i, 'trimp']
        else:
            weeks_gap = (weekly_df.loc[i, 'start_date'] - weekly_df.loc[i-1, 'start_date']).days / 7
            decay_factor = decay_rate ** weeks_gap
            
            previous_fitness = weekly_df.loc[i-1, 'cumulative_fitness'] * decay_factor
            current_stimulus = weekly_df.loc[i, 'trimp']
            
            weekly_df.loc[i, 'cumulative_fitness'] = previous_fitness + current_stimulus
    
    # Training consistency score (percentage of weeks with training)
    total_weeks = weekly_df['weeks_training'].max()
    active_weeks = len(weekly_df)
    weekly_df['training_consistency'] = (active_weeks / total_weeks) * 100
    
    # Experience factor - training maturity bonus
    # Plateaus after ~2 years of consistent training
    weekly_df['experience_factor'] = np.minimum(
        1.0 + (weekly_df['weeks_training'] / 104) * 0.5,  # 50% bonus over 2 years
        1.5
    )
    
    # Volume consistency - reward steady accumulation
    weekly_df['weekly_distance_km'] = weekly_df['distance'] / 1000
    weekly_df['volume_consistency'] = 100 - (weekly_df['weekly_distance_km'].rolling(
        window=min(12, len(weekly_df)), min_periods=1
    ).std() / weekly_df['weekly_distance_km'].rolling(
        window=min(12, len(weekly_df)), min_periods=1
    ).mean() * 100).fillna(0)
    weekly_df['volume_consistency'] = weekly_df['volume_consistency'].clip(0, 100)
    
    # Normalize components for composite score
    def safe_normalize(series, target_max=None):
        if series.max() == series.min():
            return pd.Series([50] * len(series), index=series.index)
        
        if target_max:
            return (series / target_max * 100).clip(0, 100)
        else:
            return ((series - series.min()) / (series.max() - series.min()) * 100)
    
    # Component scores
    weekly_df['fitness_score'] = safe_normalize(weekly_df['cumulative_fitness'])
    weekly_df['volume_score'] = safe_normalize(weekly_df['cumulative_volume'])
    weekly_df['consistency_score'] = weekly_df['training_consistency']
    weekly_df['experience_score'] = safe_normalize(weekly_df['experience_factor'])
    
    # Composite Fitness Bank score
    weekly_df['fitness_bank'] = (
        weekly_df['fitness_score'] * 0.4 +           # Accumulated training load
        weekly_df['volume_score'] * 0.25 +           # Total volume logged  
        weekly_df['consistency_score'] * 0.25 +      # Training adherence
        weekly_df['experience_score'] * 0.1          # Training maturity
    )
    
    # Add milestones
    volume_milestones = [100, 500, 1000, 2000, 5000, 10000]  # km
    weekly_df['volume_milestone'] = 0
    for milestone in volume_milestones:
        weekly_df.loc[weekly_df['cumulative_volume'] >= milestone, 'volume_milestone'] = milestone
    
    # Training age in years
    weekly_df['training_age_years'] = weekly_df['weeks_training'] / 52.0
    
    return weekly_df


def get_lifetime_achievements(weekly_df: pd.DataFrame) -> dict:
    """
    Extract lifetime training achievements from cumulative data.
    
    Args:
        weekly_df (pd.DataFrame): Output from get_cumulative_training_shape_data()
        
    Returns:
        dict: Achievement metrics for gamification/motivation
    """
    if weekly_df.empty:
        return {}
    
    latest = weekly_df.iloc[-1]
    
    # Calculate streaks and achievements
    max_fitness = weekly_df['fitness_bank'].max()
    current_fitness_pct = (latest['fitness_bank'] / max_fitness * 100) if max_fitness > 0 else 0
    
    return {
        'total_distance_km': round(latest['cumulative_volume'], 1),
        'training_age_years': round(latest['training_age_years'], 1),
        'fitness_bank_score': round(latest['fitness_bank'], 1),
        'peak_fitness_achieved': round(max_fitness, 1),
        'current_vs_peak_pct': round(current_fitness_pct, 1),
        'consistency_rate': round(latest['training_consistency'], 1),
        'volume_milestone': latest['volume_milestone'],
        'experience_level': 'Beginner' if latest['training_age_years'] < 0.5 else
                           'Developing' if latest['training_age_years'] < 1.5 else
                           'Experienced' if latest['training_age_years'] < 3 else 'Veteran',
        'total_weeks_trained': len(weekly_df)
    }

def add_exercise(conn, data):
        c = conn.cursor()
        c.execute('''
        INSERT INTO exercises (
            name, description, instructions, exercise_type, movement_pattern,
            primary_muscles, secondary_muscles, muscle_groups, unilateral,
            difficulty_rating, prerequisites, progressions, regressions,
            equipment_required, equipment_optional, setup_time, space_required,
            rep_range_min, rep_range_max, tempo, range_of_motion, compound_vs_isolation,
            injury_risk_level, contraindications, common_mistakes, safety_notes,
            image_url, video_url, gif_url, diagram_url,
            category, training_style, experience_level, goals,
            duration_minutes, popularity_score, alternatives, supersets_well_with
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            data.get('name'), data.get('description'), data.get('instructions'), data.get('exercise_type'), data.get('movement_pattern'),
            json.dumps(data.get('primary_muscles')), json.dumps(data.get('secondary_muscles')), data.get('muscle_groups'), data.get('unilateral'),
            data.get('difficulty_rating'), data.get('prerequisites'), json.dumps(data.get('progressions')), json.dumps(data.get('regressions')),
            json.dumps(data.get('equipment_required')), data.get('equipment_optional'), data.get('setup_time'), data.get('space_required'),
            data.get('rep_range_min'), data.get('rep_range_max'), data.get('tempo'), data.get('range_of_motion'), data.get('compound_vs_isolation'),
            data.get('injury_risk_level'), data.get('contraindications'), json.dumps(data.get('common_mistakes')), data.get('safety_notes'),
            data.get('image_url'), data.get('video_url'), data.get('gif_url'), data.get('diagram_url'),
            data.get('category'), json.dumps(data.get('training_style')), json.dumps(data.get('experience_level')), json.dumps(data.get('goals')),
            data.get('duration_minutes'), data.get('popularity_score'), json.dumps(data.get('alternatives')), json.dumps(data.get('supersets_well_with'))
        ])
        conn.commit()

# C team late add

# Exercise functions
def get_all_exercises(conn: sqlite3.Connection) -> List[Dict]:
    """Get all exercises from the database"""
    cursor = conn.execute("SELECT * FROM exercises ORDER BY name")
    result = cursor.fetchall()
    return dicts_from_rows(result)

def get_exercise_by_id(conn: sqlite3.Connection, exercise_id: int) -> Optional[Dict]:
    """Get a specific exercise by ID"""
    cursor = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,))
    result = cursor.fetchone()
    return dict_from_row(result)

# Routine functions
def get_all_routines(conn: sqlite3.Connection) -> List[Dict]:
    """Get all workout routines with exercise count"""
    cursor = conn.execute("""
            SELECT 
                wr.*,
                COUNT(re.id) as exercise_count
            FROM workout_routines wr
            LEFT JOIN routine_exercises re ON wr.id = re.routine_id
            GROUP BY wr.id
            ORDER BY wr.date_created DESC
        """)
    result = cursor.fetchall()
    return dicts_from_rows(result)

def get_routine_by_id(conn: sqlite3.Connection, routine_id: int) -> Optional[Dict]:
    """Get a specific routine by ID"""
    cursor = conn.execute("SELECT * FROM workout_routines WHERE id = ?", (routine_id,))
    result = cursor.fetchone()
    return dict_from_row(result)

def create_routine(conn: sqlite3.Connection, name: str) -> int:
    """Create a new workout routine and return its ID"""
    cursor = conn.execute(
        "INSERT INTO workout_routines (name, date_created) VALUES (?, ?)",
        (name, datetime.datetime.now().date())
    )
    conn.commit()
    return cursor.lastrowid

def add_exercise_to_routine(conn: sqlite3.Connection, routine_id: int, exercise_id: int, sets: int, 
                          reps: int, load_lbs: float, order_index: int, notes: str = '') -> int:
    """Add an exercise to a routine"""
    cursor = conn.execute("""
        INSERT INTO routine_exercises 
        (routine_id, exercise_id, sets, reps, load_lbs, order_index, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (routine_id, exercise_id, sets, reps, load_lbs, order_index, notes))
    conn.commit()
    return cursor.lastrowid

def get_routine_exercises(conn: sqlite3.Connection, routine_id: int) -> List[Dict]:
    """Get all exercises for a specific routine"""
    cursor = conn.execute("""
            SELECT 
                re.*,
                e.name,
                e.description,
                e.primary_muscles,
                e.secondary_muscles,
                e.equipment_required,
                e.instructions
            FROM routine_exercises re
            JOIN exercises e ON re.exercise_id = e.id
            WHERE re.routine_id = ?
            ORDER BY re.order_index
        """, (routine_id,))
    exercises = cursor.fetchall()
    return dicts_from_rows(exercises)

def delete_routine(conn: sqlite3.Connection, routine_id: int) -> bool:
    """Delete a routine and all associated exercises"""
    # Delete routine exercises first due to foreign key constraint
    conn.execute("DELETE FROM routine_exercises WHERE routine_id = ?", (routine_id,))
    # Delete the routine
    cursor = conn.execute("DELETE FROM workout_routines WHERE id = ?", (routine_id,))
    conn.commit()
    return cursor.rowcount > 0

# Workout performance functions
def save_workout_performance(conn: sqlite3.Connection, routine_id: int, exercise_id: int, workout_date: str,
                           planned_sets: int, actual_sets: int, planned_reps: int,
                           actual_reps: int, planned_load_lbs: float, actual_load_lbs: float,
                           notes: str = '', completion_status: str = 'completed') -> int:
    """Save workout performance data"""
    cursor = conn.execute("""
            INSERT INTO workout_performance 
            (routine_id, exercise_id, workout_date, planned_sets, actual_sets,
             planned_reps, actual_reps, planned_load_lbs, actual_load_lbs,
             notes, completion_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (routine_id, exercise_id, workout_date, planned_sets, actual_sets,
              planned_reps, actual_reps, planned_load_lbs, actual_load_lbs,
              notes, completion_status))
    conn.commit()
    return cursor.lastrowid

def get_workout_history(conn: sqlite3.Connection, routine_id: int) -> List[Dict]:
    """Get workout history for a specific routine"""
    cursor = conn.execute("""
            SELECT 
                wp.*,
                e.name as exercise_name,
                e.primary_muscles
            FROM workout_performance wp
            JOIN exercises e ON wp.exercise_id = e.id
            WHERE wp.routine_id = ?
            ORDER BY wp.workout_date DESC, wp.created_at DESC
        """, (routine_id,))
    history = cursor.fetchall()
    return dicts_from_rows(history)

def get_workout_performance_by_date(conn: sqlite3.Connection, routine_id: int, workout_date: str) -> List[Dict]:
    """Get workout performance for a specific routine and date"""
    cursor = conn.execute("""
            SELECT 
                wp.*,
                e.name as exercise_name,
                e.primary_muscles
            FROM workout_performance wp
            JOIN exercises e ON wp.exercise_id = e.id
            WHERE wp.routine_id = ? AND wp.workout_date = ?
            ORDER BY wp.created_at
        """, (routine_id, workout_date))
    performance = cursor.fetchall()
    return dicts_from_rows(performance)

def get_exercise_progress(conn: sqlite3.Connection, exercise_id: int, limit: int = 10) -> List[Dict]:
    """Get progress history for a specific exercise"""
    cursor = conn.execute("""
            SELECT 
                wp.*,
                wr.name as routine_name
            FROM workout_performance wp
            JOIN workout_routines wr ON wp.routine_id = wr.id
            WHERE wp.exercise_id = ?
            ORDER BY wp.workout_date DESC, wp.created_at DESC
            LIMIT ?
        """, (exercise_id, limit))
    progress = cursor.fetchall()
    return dicts_from_rows(progress)

def get_recent_workouts(conn: sqlite3.Connection, limit: int = 10) -> List[Dict]:
    """Get recent workout sessions"""
    cursor = conn.execute("""
            SELECT 
                wp.workout_date,
                wp.routine_id,
                wr.name as routine_name,
                COUNT(wp.id) as exercises_completed,
                AVG(CASE WHEN wp.completion_status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as completion_rate
            FROM workout_performance wp
            JOIN workout_routines wr ON wp.routine_id = wr.id
            GROUP BY wp.workout_date, wp.routine_id
            ORDER BY wp.workout_date DESC
            LIMIT ?
        """, (limit,))
    workouts = cursor.fetchall()
    return dicts_from_rows(workouts)

def get_workout_stats(conn: sqlite3.Connection, routine_id: int = None) -> Dict:
    """Get workout statistics"""
    cursor = conn.cursor()
    
    if routine_id:
        # Stats for specific routine
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT workout_date) as total_sessions,
                COUNT(*) as total_exercises,
                AVG(CASE WHEN completion_status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as avg_completion_rate,
                MAX(workout_date) as last_workout
            FROM workout_performance
            WHERE routine_id = ?
        """, (routine_id,))
    else:
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT workout_date || '-' || routine_id) as total_sessions,
                COUNT(*) as total_exercises,
                AVG(CASE WHEN completion_status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as avg_completion_rate,
                MAX(workout_date) as last_workout
            FROM workout_performance
        """)
    
    stats = cursor.fetchone()
    return dict_from_row(stats)

# Utility functions for database setup
def initialize_runstrong_database(conn: sqlite3.Connection) -> bool:
    """Initialize the database with all required tables"""
    cursor = conn.cursor()
    
    # Create exercises table

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            description TEXT,
            instructions TEXT,
            exercise_type TEXT,
            movement_pattern TEXT,
            primary_muscles TEXT,
            secondary_muscles TEXT,
            muscle_groups TEXT,
            unilateral BOOLEAN,
            difficulty_rating TEXT,
            prerequisites TEXT,
            progressions TEXT,
            regressions TEXT,
            equipment_required TEXT,
            equipment_optional TEXT,
            setup_time INTEGER,
            space_required TEXT,
            rep_range_min INTEGER,
            rep_range_max INTEGER,
            tempo TEXT,
            range_of_motion TEXT,
            compound_vs_isolation TEXT,
            injury_risk_level TEXT,
            contraindications TEXT,
            common_mistakes TEXT,
            safety_notes TEXT,
            image_url TEXT,
            video_url TEXT,
            gif_url TEXT,
            diagram_url TEXT,
            category TEXT,
            training_style TEXT,
            experience_level TEXT,
            goals TEXT,
            duration_minutes INTEGER,
            popularity_score INTEGER,
            alternatives TEXT,
            supersets_well_with TEXT
        )
    ''')
    
    # Create workout_routines table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workout_routines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date_created DATE
        )
    ''')
    
    # Create routine_exercises table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS routine_exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            routine_id INTEGER,
            exercise_id INTEGER,
            sets INTEGER,
            reps INTEGER,
            load_lbs FLOAT,
            order_index INTEGER,
            notes TEXT,
            FOREIGN KEY(routine_id) REFERENCES workout_routines(id),
            FOREIGN KEY(exercise_id) REFERENCES exercises(id)
        )
    ''')
    
    # Create workout_performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workout_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            routine_id INTEGER,
            exercise_id INTEGER,
            workout_date DATE,
            planned_sets INTEGER,
            actual_sets INTEGER,
            planned_reps INTEGER,
            actual_reps INTEGER,
            planned_load_lbs FLOAT,
            actual_load_lbs FLOAT,
            notes TEXT,
            completion_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(routine_id) REFERENCES workout_routines(id),
            FOREIGN KEY(exercise_id) REFERENCES exercises(id)
        )
    ''')
    
    conn.commit()
    return True

## Begin LLM section

def update_daily_training_metrics(conn: sqlite3.Connection, df, user_id=1):
    """
    Inserts or updates daily training metrics into SQLite.

    Args:
        df (DataFrame): Must have columns ['date', 'tss', 'CTL', 'ATL', 'TSB']
        user_id (int): The athlete identifier
        db_path (str): Path to SQLite database
    """

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            user_id INTEGER,
            total_tss FLOAT,
            ctl FLOAT,
            atl FLOAT,
            tsb FLOAT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO daily_training_metrics (
                date, user_id, total_tss, ctl, atl, tsb
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_tss = excluded.total_tss,
                ctl = excluded.ctl,
                atl = excluded.atl,
                tsb = excluded.tsb;
        """, (
            row['date'].date(),  # Ensure date only, no time
            user_id,
            row['tss'],
            row['CTL'],
            row['ATL'],
            row['TSB']
        ))

    conn.commit()
    conn.close()

def get_latest_daily_training_metrics(conn: sqlite3.Connection):
    """
    Retreives latest day's data from the daily_training_metrics table.

    Args:
        conn
    """
    c = conn.cursor()
    query = """
        SELECT date, total_tss, ctl, atl, tsb
        FROM daily_training_metrics
        ORDER BY date DESC
        LIMIT 1;
    """
    result = c.execute(query).fetchall()

    return result