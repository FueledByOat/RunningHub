# strava_utils.py
"""
Utility functions for interacting with the Strava API and database operations.
Handles data extraction, transformation, and loading for Strava athlete data.
"""

import sqlite3
import requests
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

from utils import db_utils
from config import Config

# Configure module-level logger
logger = logging.getLogger(__name__)


def update_env_variable(key: str, value: str, env_file: str = "secrets.env") -> None:
    """
    Update or add an environment variable in the specified .env file.
    
    Args:
        key: Environment variable name
        value: Environment variable value
        env_file: Path to the .env file (default: "secrets.env")
    
    Raises:
        IOError: If file operations fail
    """
    try:
        lines = []
        updated = False
        
        # Read existing file if it exists
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Update existing variable
            for i, line in enumerate(lines):
                # Handle both "KEY=value" and "KEY = value" formats
                if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break
                    
        # Add new variable if not found
        if not updated:
            # Ensure we have a newline before adding new variable
            if lines and not lines[-1].endswith('\n'):
                lines.append('\n')
            lines.append(f"{key}={value}\n")
            
        # Write changes back to file
        with open(env_file, 'w', encoding='utf-8') as file:
            file.writelines(lines)
            
        logger.debug(f"Updated environment variable '{key}' in {env_file}")
        
    except IOError as e:
        logger.error(f"Failed to update environment variable '{key}': {e}")
        raise


def latest_activity_import_date(db_path: str) -> Optional[int]:
    """
    Get the timestamp of the most recently imported activity.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Unix timestamp of the latest import, or None if no activities exist
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(import_date) FROM activities")
            result = cursor.fetchone()[0]
            
            if result:
                # Parse the ISO format datetime and convert to Unix timestamp
                dt = datetime.fromisoformat(result.replace('Z', '+00:00'))
                timestamp = int(dt.timestamp())
                logger.debug(f"Latest activity import timestamp: {timestamp}")
                return timestamp
                
            logger.info("No previous activity imports found")
            return None
            
    except sqlite3.Error as e:
        logger.error(f"Database error getting latest import date: {e}")
        raise
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing import date: {e}")
        return None


def get_activities(access_token: str, page: int = 1, per_page: int = 30, 
                  **optional_parameters) -> List[Dict[str, Any]]:
    """
    Retrieve activities from the Strava API.
    
    Args:
        access_token: Valid Strava access token
        page: Page number for pagination (default: 1)
        per_page: Number of activities per page (default: 30, max: 200)
        **optional_parameters: Additional query parameters (before, after timestamps)
        
    Returns:
        List of activity dictionaries from Strava API
        
    Raises:
        requests.HTTPError: If API request fails
        requests.RequestException: For other request-related errors
    """
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Build query parameters
    params = {
        "page": page, 
        "per_page": min(per_page, 200)  # Strava API limit is 200
    }
    params.update(optional_parameters)
    
    try:
        logger.debug(f"Fetching activities with params: {params}")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        activities = response.json()
        logger.info(f"Successfully fetched {len(activities)} activities")
        return activities
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout when fetching activities")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching activities: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching activities: {e}")
        raise


def get_streams(access_token: str, activity_id: int, 
               keys: str = "time,distance,latlng,altitude,velocity_smooth,heartrate,cadence,watts,temp,moving,grade_smooth") -> Dict[str, Any]:
    """
    Retrieve stream data for a specific activity from the Strava API.
    
    Args:
        access_token: Valid Strava access token
        activity_id: Strava activity ID
        keys: Comma-separated string of stream types to retrieve
        
    Returns:
        Dictionary containing stream data organized by type
        
    Raises:
        requests.HTTPError: If API request fails (404 if no streams available)
        requests.RequestException: For other request-related errors
    """
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"keys": keys, "key_by_type": True}
    
    try:
        logger.debug(f"Fetching streams for activity {activity_id}")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        streams = response.json()
        logger.debug(f"Successfully fetched streams for activity {activity_id}")
        return streams
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout when fetching streams for activity {activity_id}")
        raise
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            logger.warning(f"No stream data available for activity {activity_id}")
        else:
            logger.error(f"HTTP error fetching streams for activity {activity_id}: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching streams for activity {activity_id}: {e}")
        raise


def get_gear(access_token: str, gear_id: str) -> Dict[str, Any]:
    """
    Retrieve gear information from the Strava API.
    
    Args:
        access_token: Valid Strava access token
        gear_id: Strava gear ID
        
    Returns:
        Dictionary containing gear information
        
    Raises:
        requests.HTTPError: If API request fails
        requests.RequestException: For other request-related errors
    """
    url = f"https://www.strava.com/api/v3/gear/{gear_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        logger.debug(f"Fetching gear information for gear {gear_id}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        gear_data = response.json()
        logger.debug(f"Successfully fetched gear data for {gear_id}")
        return gear_data
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout when fetching gear {gear_id}")
        raise
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            logger.warning(f"Gear {gear_id} not found")
        else:
            logger.error(f"HTTP error fetching gear {gear_id}: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching gear {gear_id}: {e}")
        raise


def insert_activities_batch(activity_list: List[Dict[str, Any]], db_path: str) -> int:
    """
    Efficiently insert multiple activity records into the database using batch operations.
    
    Args:
        activity_list: List of activity dictionaries from Strava API
        db_path: Path to the SQLite database
        
    Returns:
        Number of activities successfully inserted
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    if not activity_list:
        logger.info("No activities to insert")
        return 0
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Prepare data for batch insert
            data = []
            current_time = datetime.now().isoformat()
            
            for activity in activity_list:
                # Safely extract nested values with proper defaults
                athlete_data = activity.get("athlete", {})
                map_data = activity.get("map", {})
                
                data.append({
                    "id": activity.get("id"),
                    "resource_state": activity.get("resource_state"),
                    "athlete_id": athlete_data.get("id"),
                    "athlete_resource_state": athlete_data.get("resource_state"),
                    "name": activity.get("name"),
                    "distance": activity.get("distance"),
                    "moving_time": activity.get("moving_time"),
                    "elapsed_time": activity.get("elapsed_time"),
                    "total_elevation_gain": activity.get("total_elevation_gain"),
                    "type": activity.get("type"),
                    "sport_type": activity.get("sport_type"),
                    "workout_type": activity.get("workout_type"),
                    "start_date": activity.get("start_date"),
                    "start_date_local": activity.get("start_date_local"),
                    "timezone": activity.get("timezone"),
                    "utc_offset": activity.get("utc_offset"),
                    "location_city": activity.get("location_city"),
                    "location_state": activity.get("location_state"),
                    "location_country": activity.get("location_country"),
                    "achievement_count": activity.get("achievement_count"),
                    "kudos_count": activity.get("kudos_count"),
                    "comment_count": activity.get("comment_count"),
                    "athlete_count": activity.get("athlete_count"),
                    "photo_count": activity.get("photo_count"),
                    "map_id": map_data.get("id"),
                    "map_summary_polyline": map_data.get("summary_polyline"),
                    "map_resource_state": map_data.get("resource_state"),
                    "trainer": activity.get("trainer"),
                    "commute": activity.get("commute"),
                    "manual": activity.get("manual"),
                    "private": activity.get("private"),
                    "visibility": activity.get("visibility"),
                    "flagged": activity.get("flagged"),
                    "gear_id": activity.get("gear_id"),
                    "start_latlng": json.dumps(activity.get("start_latlng")) if activity.get("start_latlng") else None,
                    "end_latlng": json.dumps(activity.get("end_latlng")) if activity.get("end_latlng") else None,
                    "average_speed": activity.get("average_speed"),
                    "max_speed": activity.get("max_speed"),
                    "average_cadence": activity.get("average_cadence"),
                    "average_watts": activity.get("average_watts"),
                    "max_watts": activity.get("max_watts"),
                    "weighted_average_watts": activity.get("weighted_average_watts"),
                    "device_watts": activity.get("device_watts"),
                    "kilojoules": activity.get("kilojoules"),
                    "has_heartrate": activity.get("has_heartrate"),
                    "average_heartrate": activity.get("average_heartrate"),
                    "max_heartrate": activity.get("max_heartrate"),
                    "heartrate_opt_out": activity.get("heartrate_opt_out"),
                    "display_hide_heartrate_option": activity.get("display_hide_heartrate_option"),
                    "elev_high": activity.get("elev_high"),
                    "elev_low": activity.get("elev_low"),
                    "upload_id": activity.get("upload_id"),
                    "upload_id_str": activity.get("upload_id_str"),
                    "external_id": activity.get("external_id"),
                    "from_accepted_tag": activity.get("from_accepted_tag"),
                    "pr_count": activity.get("pr_count"),
                    "total_photo_count": activity.get("total_photo_count"),
                    "has_kudoed": activity.get("has_kudoed"),
                    "import_date": current_time
                })

            # Execute batch insert with proper error handling
            cursor.executemany('''
                INSERT OR IGNORE INTO activities VALUES (
                    :id, :resource_state, :athlete_id, :athlete_resource_state,
                    :name, :distance, :moving_time, :elapsed_time, :total_elevation_gain,
                    :type, :sport_type, :workout_type, :start_date, :start_date_local,
                    :timezone, :utc_offset, :location_city, :location_state, :location_country,
                    :achievement_count, :kudos_count, :comment_count, :athlete_count, :photo_count,
                    :map_id, :map_summary_polyline, :map_resource_state,
                    :trainer, :commute, :manual, :private, :visibility, :flagged, :gear_id,
                    :start_latlng, :end_latlng,
                    :average_speed, :max_speed, :average_cadence, :average_watts,
                    :max_watts, :weighted_average_watts, :device_watts, :kilojoules,
                    :has_heartrate, :average_heartrate, :max_heartrate,
                    :heartrate_opt_out, :display_hide_heartrate_option,
                    :elev_high, :elev_low,
                    :upload_id, :upload_id_str, :external_id, :from_accepted_tag,
                    :pr_count, :total_photo_count, :has_kudoed, :import_date
                )
            ''', data)
            
            rows_affected = cursor.rowcount
            conn.commit()
            
            logger.info(f"Successfully inserted {rows_affected} activities into database")
            return rows_affected
            
    except sqlite3.Error as e:
        logger.error(f"Database error during batch activity insert: {e}")
        raise



def insert_stream_data(activity_id: int, stream_dict: Dict[str, Any], db_path: str) -> bool:
    """
    Insert or replace stream data for a specific activity.
    
    Args:
        activity_id: Strava activity ID
        stream_dict: Dictionary containing stream data with keys like 'time', 'distance', etc.
        db_path: Path to the SQLite database
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    if not stream_dict:
        logger.warning(f"No stream data provided for activity {activity_id}")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Build dynamic column mappings
            columns = ["activity_id"]
            placeholders = ["?"]
            values = [activity_id]

            # Process each stream type
            for key, stream_data in stream_dict.items():
                if not isinstance(stream_data, dict):
                    logger.warning(f"Invalid stream data format for key '{key}' in activity {activity_id}")
                    continue

                # Add columns for each stream attribute
                columns.extend([
                    f"{key}_data",
                    f"{key}_series_type", 
                    f"{key}_original_size",
                    f"{key}_resolution"
                ])
                placeholders.extend(["?"] * 4)
                
                # Safely extract stream attributes
                values.extend([
                    json.dumps(stream_data.get("data")) if stream_data.get("data") else None,
                    stream_data.get("series_type"),
                    stream_data.get("original_size"),
                    stream_data.get("resolution")
                ])

            # Construct and execute insert statement
            sql = f"""
                INSERT OR REPLACE INTO streams ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            cursor.execute(sql, values)
            conn.commit()
            
            logger.debug(f"Successfully inserted stream data for activity {activity_id}")
            return True
            
    except sqlite3.Error as e:
        logger.error(f"Database error inserting stream data for activity {activity_id}: {e}")
        return False


def insert_single_gear(gear_data: Dict[str, Any], db_path: str) -> bool:
    """
    Insert or replace a single gear record in the database.
    
    Args:
        gear_data: Dictionary containing gear information from Strava API
        db_path: Path to the SQLite database
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    if not gear_data or not gear_data.get("id"):
        logger.warning("Invalid gear data provided")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO gear VALUES (
                    :gear_id, :is_primary, :nickname, :resource_state, :retired, :distance,
                    :brand_name, :model_name, :frame_type, :description, :weight
                )
            ''', {
                "gear_id": gear_data.get("id"),
                "is_primary": gear_data.get("primary"),  # Note: API returns 'primary', DB expects 'is_primary'
                "nickname": gear_data.get("nickname"),
                "resource_state": gear_data.get("resource_state"),
                "retired": gear_data.get("retired"),
                "distance": gear_data.get("distance"),
                "brand_name": gear_data.get("brand_name"),
                "model_name": gear_data.get("model_name"),
                "frame_type": gear_data.get("frame_type"),
                "description": gear_data.get("description"),
                "weight": gear_data.get("weight"),
                "import_date": datetime.now().isoformat()
            })
            
            conn.commit()
            logger.debug(f"Successfully inserted gear data for gear {gear_data.get('id')}")
            return True
            
    except sqlite3.Error as e:
        logger.error(f"Database error inserting gear {gear_data.get('id', 'unknown')}: {e}")
        return False


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> Dict[str, str]:
    """
    Refresh the Strava access token using the refresh token.
    
    Args:
        client_id: Strava application client ID
        client_secret: Strava application client secret
        refresh_token: Current refresh token
        
    Returns:
        Dictionary containing new access_token and refresh_token
        
    Raises:
        requests.HTTPError: If token refresh fails
        requests.RequestException: For other request-related errors
    """
    url = "https://www.strava.com/oauth/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    
    try:
        logger.info("Attempting to refresh access token")
        response = requests.post(url, data=data, timeout=30)
        response.raise_for_status()
        
        token_info = response.json()
        
        # Validate response contains required fields
        if not all(key in token_info for key in ["access_token", "refresh_token"]):
            raise ValueError("Invalid token refresh response from Strava API")
        
        # Update environment variables with new tokens
        update_env_variable("REFRESH_TOKEN", token_info["refresh_token"])
        update_env_variable("ACCESS_TOKEN", token_info["access_token"])
        
        # Reload environment variables
        load_dotenv(dotenv_path="secrets.env", override=True)
        
        logger.info("Successfully refreshed access token")
        return {
            "access_token": token_info["access_token"],
            "refresh_token": token_info["refresh_token"]
        }
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout during token refresh")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during token refresh: {e}")
        if hasattr(e.response, 'json'):
            try:
                error_details = e.response.json()
                logger.error(f"Token refresh error details: {error_details}")
            except json.JSONDecodeError:
                pass
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during token refresh: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid response during token refresh: {e}")
        raise


def create_database_tables(db_path: str) -> None:
    """
    Create necessary database tables if they don't exist.
    
    Args:
        db_path: Path to the SQLite database
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create activities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY,
                    resource_state INTEGER,
                    athlete_id INTEGER,
                    athlete_resource_state INTEGER,
                    name TEXT,
                    distance REAL,
                    moving_time INTEGER,
                    elapsed_time INTEGER,
                    total_elevation_gain REAL,
                    type TEXT,
                    sport_type TEXT,
                    workout_type INTEGER,
                    start_date TEXT,
                    start_date_local TEXT,
                    timezone TEXT,
                    utc_offset REAL,
                    location_city TEXT,
                    location_state TEXT,
                    location_country TEXT,
                    achievement_count INTEGER,
                    kudos_count INTEGER,
                    comment_count INTEGER,
                    athlete_count INTEGER,
                    photo_count INTEGER,
                    map_id TEXT,
                    map_summary_polyline TEXT,
                    map_resource_state INTEGER,
                    trainer BOOLEAN,
                    commute BOOLEAN,
                    manual BOOLEAN,
                    private BOOLEAN,
                    visibility TEXT,
                    flagged BOOLEAN,
                    gear_id TEXT,
                    start_latlng TEXT,
                    end_latlng TEXT,
                    average_speed REAL,
                    max_speed REAL,
                    average_cadence REAL,
                    average_watts REAL,
                    max_watts REAL,
                    weighted_average_watts REAL,
                    device_watts BOOLEAN,
                    kilojoules REAL,
                    has_heartrate BOOLEAN,
                    average_heartrate REAL,
                    max_heartrate REAL,
                    heartrate_opt_out BOOLEAN,
                    display_hide_heartrate_option BOOLEAN,
                    elev_high REAL,
                    elev_low REAL,
                    upload_id INTEGER,
                    upload_id_str TEXT,
                    external_id TEXT,
                    from_accepted_tag BOOLEAN,
                    pr_count INTEGER,
                    total_photo_count INTEGER,
                    has_kudoed BOOLEAN,
                    import_date TEXT
                )
            ''')
            
            # Create gear table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gear (
                    gear_id TEXT PRIMARY KEY,
                    is_primary BOOLEAN,
                    nickname TEXT,
                    resource_state INTEGER,
                    retired BOOLEAN,
                    distance REAL,
                    brand_name TEXT,
                    model_name TEXT,
                    frame_type INTEGER,
                    description TEXT,
                    weight REAL,
                    import_date TEXT
                )
            ''')
            
            # Note: Streams table structure would depend on your specific needs
            # This is a simplified version - you may want to create separate tables
            # for different stream types for better normalization
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS streams (
                    activity_id INTEGER PRIMARY KEY,
                    time_data TEXT,
                    time_series_type TEXT,
                    time_original_size INTEGER,
                    time_resolution TEXT,
                    distance_data TEXT,
                    distance_series_type TEXT,
                    distance_original_size INTEGER,
                    distance_resolution TEXT,
                    latlng_data TEXT,
                    latlng_series_type TEXT,
                    latlng_original_size INTEGER,
                    latlng_resolution TEXT,
                    altitude_data TEXT,
                    altitude_series_type TEXT,
                    altitude_original_size INTEGER,
                    altitude_resolution TEXT,
                    velocity_smooth_data TEXT,
                    velocity_smooth_series_type TEXT,
                    velocity_smooth_original_size INTEGER,
                    velocity_smooth_resolution TEXT,
                    heartrate_data TEXT,
                    heartrate_series_type TEXT,
                    heartrate_original_size INTEGER,
                    heartrate_resolution TEXT,
                    cadence_data TEXT,
                    cadence_series_type TEXT,
                    cadence_original_size INTEGER,
                    cadence_resolution TEXT,
                    watts_data TEXT,
                    watts_series_type TEXT,
                    watts_original_size INTEGER,
                    watts_resolution TEXT,
                    temp_data TEXT,
                    temp_series_type TEXT,
                    temp_original_size INTEGER,
                    temp_resolution TEXT,
                    moving_data TEXT,
                    moving_series_type TEXT,
                    moving_original_size INTEGER,
                    moving_resolution TEXT,
                    grade_smooth_data TEXT,
                    grade_smooth_series_type TEXT,
                    grade_smooth_original_size INTEGER,
                    grade_smooth_resolution TEXT,
                    FOREIGN KEY (activity_id) REFERENCES activities (id)
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activities_start_date ON activities(start_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activities_import_date ON activities(import_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activities_type ON activities(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gear_retired ON gear(retired)')
            
            conn.commit()
            logger.info("Database tables created/verified successfully")
            
    except sqlite3.Error as e:
        logger.error(f"Database error creating tables: {e}")
        raise


def get_database_stats(db_path: str) -> Dict[str, int]:
    """
    Get basic statistics about the database contents.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary containing count statistics for each table
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count activities
            cursor.execute("SELECT COUNT(*) FROM activities")
            stats['activities'] = cursor.fetchone()[0]
            
            # Count gear items
            cursor.execute("SELECT COUNT(*) FROM gear")
            stats['gear'] = cursor.fetchone()[0]
            
            # Count stream records
            cursor.execute("SELECT COUNT(*) FROM streams")
            stats['streams'] = cursor.fetchone()[0]
            
            # Get date range of activities
            cursor.execute("SELECT MIN(start_date), MAX(start_date) FROM activities")
            date_range = cursor.fetchone()
            stats['date_range'] = {
                'earliest': date_range[0],
                'latest': date_range[1]
            }
            
            logger.info(f"Database stats: {stats}")
            return stats
            
    except sqlite3.Error as e:
        logger.error(f"Database error getting stats: {e}")
        raise

def update_daily_dashboard_metrics() -> None:
    """
    After any potential activity import, call to db to calculate
    metrics like ctl, atl, tsb, tss, etc.
    """

    df = db_utils.get_ctl_atl_tsb_tss_data().tail(1)
    try:
        with sqlite3.connect(Config.DB_PATH) as conn:
            db_utils.update_daily_training_metrics(conn=conn, df=df)
            logger.info(f"Daily stats updated with: {df}")
    except sqlite3.Error as e:
        logger.error(f"Database error updating daily metrics: {e}")
        raise