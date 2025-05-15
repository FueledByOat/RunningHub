# strava_utils.py
"""Storing functions relating to pulling
data from the strava api via airflow
and writing it to the db"""

import sqlite3
import requests
import json
from datetime import datetime, date
import os
from dotenv import load_dotenv

def update_env_variable(key, value, env_file="secrets.env"):
    """Update or add an environment variable in the .env file"""
    lines = []
    updated = False
    # Read existing file
    try:
        with open(env_file, 'r') as file:
            lines = file.readlines()

        # Update existing variable
        for i, line in enumerate(lines):

            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break
                
        # Add new variable if not found
        if not updated:
            lines.append(f"\n{key}={value}\n")
            
        # Write changes back
        with open(env_file, 'w') as file:
            file.writelines(lines)
    except FileNotFoundError:
        # Create file if it doesn't exist
        with open(env_file, 'w') as file:
            file.write(f"{key}={value}\n")

def latest_activity_import_date(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""SELECT max(import_date) FROM activities""")
    record = c.fetchone()[0]
    conn.close()
    if record:
        dt = datetime.strptime(record, "%Y-%m-%dT%H:%M:%S.%f")
        return int(dt.timestamp())
    return None

def get_activities(access_token, page=1, per_page=30, **optional_parameters):
    """Retrieves activities from the Strava API.
    Optional parameters should be provided at the end of the call like so:
    before = epoch_timestamp, after = epoch_timestamp
    """
    url = f"https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"page": page, "per_page": per_page}
    params.update(optional_parameters)
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def get_streams(access_token, activity_id, keys="time,distance,latlng,altitude,velocity_smooth,heartrate,cadence,watts,temp,moving,grade_smooth"):
    """Retrieves stream data from the Strava API."""
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"keys": keys, "key_by_type": True}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def get_gear(access_token, gear_id):
    url = f"https://www.strava.com/api/v3/gear/{gear_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def insert_activities_batch(activity_list, db_path):
    """Efficiently insert multiple activity records into the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    data = []
    for activity in activity_list:
        data.append({
            "id": activity["id"],
            "resource_state": activity.get("resource_state"),
            "athlete_id": activity.get("athlete", {}).get("id"),
            "athlete_resource_state": activity.get("athlete", {}).get("resource_state"),
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
            "map_id": activity.get("map", {}).get("id"),
            "map_summary_polyline": activity.get("map", {}).get("summary_polyline"),
            "map_resource_state": activity.get("map", {}).get("resource_state"),
            "trainer": activity.get("trainer"),
            "commute": activity.get("commute"),
            "manual": activity.get("manual"),
            "private": activity.get("private"),
            "visibility": activity.get("visibility"),
            "flagged": activity.get("flagged"),
            "gear_id": activity.get("gear_id"),
            "start_latlng": json.dumps(activity.get("start_latlng")),
            "end_latlng": json.dumps(activity.get("end_latlng")),
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
            "import_date": datetime.now().isoformat()
        })

    try:
        c.executemany('''
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
    except sqlite3.Error as e:
        print("Error inserting batch:", e)

    conn.commit()
    conn.close()


def insert_stream_data(activity_id, stream_dict, db_path):
    """
    Inserts or replaces a row in the streams table for a given activity_id.
    stream_dict should have keys like 'time', 'distance', etc., with each value a dict containing:
    {
        'data': [...],
        'series_type': '...',
        'original_size': ...,
        'resolution': '...'
    }
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create the column mappings dynamically
    base_columns = []
    placeholders = []
    values = []

    # Always include activity_id
    base_columns.append("activity_id")
    placeholders.append("?")
    values.append(activity_id)

    for key, val in stream_dict.items():
        if not isinstance(val, dict):
            continue  # skip malformed

        base_columns.extend([
            f"{key}_data",
            f"{key}_series_type",
            f"{key}_original_size",
            f"{key}_resolution"
        ])
        placeholders.extend(["?"] * 4)

        values.extend([
            json.dumps(val.get("data")),
            val.get("series_type"),
            val.get("original_size"),
            val.get("resolution")
        ])

    sql = f"""
        INSERT INTO streams ({', '.join(base_columns)})
        VALUES ({', '.join(placeholders)})
    """
    

    try:
        c.execute(sql, values)
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        print(f"Activity {activity_id} already exists in the 'streams' table. Skipping insert.")

def insert_single_gear(gear, db_path):
    """Insert activity records, skipping those with duplicate 'id'."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute('''
        INSERT OR REPLACE INTO gear VALUES (
            :gear_id, :is_primary, :nickname, 
            :resource_state, :retired, :distance,
            :brand_name, :model_name,
            :frame_type, :description, :weight
                  )
        ''', {
            "gear_id": gear.get("id"),
            "is_primary": gear.get("primary"),
            "nickname" : gear.get("nickname"),
            "resource_state": gear.get("resource_state"),
            "retired" : gear.get("retired"),
            "distance": gear.get("distance"),
            "brand_name": gear.get("brand_name"),
            "model_name": gear.get("model_name"),
            "frame_type": gear.get("frame_type"),
            "description": gear.get("description"),
            "weight" : gear.get("weight"),
            "import_date": datetime.now().isoformat()
        })
    except sqlite3.IntegrityError:
        print(f"Skipping duplicate activity with id {gear['id']}")

    conn.commit()
    conn.close()


def refresh_access_token(client_id, client_secret, refresh_token):
    """API call using local refresh token to get new access token.
    Needs to pull client_id, client_secret, and refresh_token from environment variables.
    Needs to write new refresh token to environment variables and pass on the received access
    token to the following api functions
    """
    url = "https://www.strava.com/oauth/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    # print('making call')
    response = requests.post(url, data=data, timeout = 10)
    # print('response generated')
    # print(response)
    response.raise_for_status()
    token_info = response.json()
    update_env_variable("REFRESH_TOKEN", token_info["refresh_token"])
    # print("Refresh Token Saved")
    load_dotenv(dotenv_path="secrets.env", override=True)
    update_env_variable("ACCESS_TOKEN", token_info["access_token"])
    # print("Access Token Saved")
    load_dotenv(dotenv_path="secrets.env", override=True)
