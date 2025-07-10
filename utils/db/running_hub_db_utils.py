# running_hub_db_utils.py
"""Database utilities for RunningHub main pages like acticities, statistics, achievements."""

import logging
import sqlite3
from typing import Optional, Dict, Any, List

from config import Config
from utils import exception_utils
import utils.db.db_utils as db_utils

logger = logging.getLogger(__name__)

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
    query = f"""
    SELECT a.*, COALESCE(CONCAT(g.model_name, " ", g.nickname), a.gear_id) as gear_name, 
    w.location_name as location_name,
    w.temp_f as temp_f,
    w.condition_text as condition_text,
    w.condition_icon as condition_icon,
    w.wind_mph as wind_mph,
    w.feelslike_f as feelslike_f, 
    w.windchill_f as windchill_f,
    w.heatindex_f as heatindex_f,
    w.dewpoint_f as dewpoint_f,
    w.gust_mph as gust_mph,
    w.uv as uv,
    w.humidity as humidity,
    w.rcs as rcs
        FROM activities as a
        LEFT JOIN gear as g ON a.gear_id = g.gear_id
		LEFT JOIN weather as w ON a.id = w.activity_id
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

    query = f"""
        SELECT id FROM activities
        WHERE type IN ('Run')
        ORDER BY start_date DESC 
        LIMIT 1
    """
    
    try:
        cur = conn.cursor()
        cur.execute(query)
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
    


def get_activities_by_type(activity_type: str, limit: int = 10, db_path: str = Config.DB_PATH) -> list[dict]:
    """Retrieve activities by type. May not be used and can be purged. 
    
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
    

# -------------------------------------
# Statistics Page SQL Logic
# -------------------------------------

def get_summary_stats(conn: sqlite3.Connection, start_date: str) -> Dict[str, Any]:
    """Get all summary stats in one query."""
    query = """
        SELECT
            COUNT(*) as total_activities,
            SUM(total_elevation_gain) as total_elevation,
            SUM(distance) as total_distance_meters,
            SUM(moving_time) as total_seconds,
            SUM(kilojoules) as total_kilojoules
        FROM activities
        WHERE start_date >= ? AND type = 'Run'
    """
    result = conn.execute(query, (start_date,)).fetchone()
    return db_utils.dict_from_row(result) if result else {}

def get_weekly_distances(conn: sqlite3.Connection, seven_days_ago: str) -> list[dict]:
    """Get distance data for the last 7 days."""
    result = conn.execute(
            """SELECT start_date_local, distance 
               FROM activities 
               WHERE start_date_local >= ? AND type = 'Run'
               ORDER BY start_date_local""",
            (seven_days_ago.strftime('%Y-%m-%d'),)
        ).fetchall()
    return db_utils.dicts_from_rows(result) if result else {}

def get_pace_trends(conn: sqlite3.Connection) -> list[dict]:
    """Get pace trends for the last 10 activities."""
    result = conn.execute(
            """SELECT start_date_local, distance, moving_time 
               FROM activities 
               WHERE type = 'Run' AND distance > 0 AND moving_time > 0
               ORDER BY start_date_local DESC 
               LIMIT 10"""
        ).fetchall()
    return db_utils.dicts_from_rows(result) if result else {}

def get_shoe_usage(conn: sqlite3.Connection, start_date: str) -> list[dict]:
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
    return db_utils.dicts_from_rows(result) if result else {}

def get_recent_activities(conn: sqlite3.Connection) -> list[dict]:
    """Get the most recent 5 activities data."""
    result = conn.execute(
            """SELECT id, name, distance, moving_time, start_date_local
               FROM activities
               WHERE type = 'Run'
               ORDER BY start_date_local DESC
               LIMIT 5"""
        ).fetchall()
    return db_utils.dicts_from_rows(result) if result else {}

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
    return db_utils.dict_from_row(result) if result else {} or None

def get_longest_run(conn: sqlite3.Connection, units: str) -> Optional[Dict[str, Any]]:
    """Get longest run record."""
    result = conn.execute(
            """SELECT id, name, distance, moving_time, start_date_local
            FROM activities
            WHERE type = 'Run'
            ORDER BY distance DESC
            LIMIT 1"""
        ).fetchone()
    return db_utils.dict_from_row(result) if result else {} or None


# -------------------------------------
# Trophy Page SQL Logic END
# -------------------------------------