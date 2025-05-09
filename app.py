from flask import Flask, render_template
import sqlite3
from dash_app import create_dash_app
import pandas as pd

app = Flask(__name__)

def format_pace(distance_miles, total_seconds):
        """
        Calculates and formats the average pace per mile.

        Args:
            distance_miles (float): Total distance run in miles.
            total_seconds (int): Total time taken in seconds.

        Returns:
            str: Formatted pace in minutes and seconds per mile (MM:SS).
                Returns "Invalid input" if inputs are invalid.
        """
        if not isinstance(distance_miles, (int, float)) or not isinstance(total_seconds, int) or distance_miles <= 0 or total_seconds < 0:
            return "Invalid input"

        seconds_per_mile = total_seconds / distance_miles
        minutes = int(seconds_per_mile // 60)
        seconds = int(seconds_per_mile % 60)
        return f"{minutes:02}:{seconds:02}"

def format_time(total_seconds):
        """
        Calculates and formats the average pace per mile.

        Args:
            distance_miles (float): Total distance run in miles.
            total_seconds (int): Total time taken in seconds.

        Returns:
            str: Formatted pace in minutes and seconds per mile (MM:SS).
                Returns "Invalid input" if inputs are invalid.
        """

        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02}:{seconds:02}"

# Get polyline on startup to avoid race conditions
def get_latest_activity():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id FROM activities ORDER BY start_date DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as e:
        print(f"Failed to get id: {e}")
        return ""

# Get polyline on startup to avoid race conditions
def get_latest_polyline(activity_id_polyline):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(f"SELECT map_summary_polyline FROM activities WHERE id = {activity_id_polyline}")
        row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as e:
        print(f"Failed to get polyline: {e}")
        return ""

DB_PATH = 'strava_data.db'
activity_id = get_latest_activity()
activity_id = 14393650080

# Initialize Dash app
create_dash_app(app, get_latest_polyline(activity_id), activity_id, DB_PATH)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/activity/")
def activity():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM activities ORDER BY start_date DESC LIMIT 1")
    # for testing
    cur.execute(f"SELECT * FROM activities WHERE id = {activity_id}")
    activity = dict(cur.fetchone())
    # print(activity.distance)
    
    activity['distance'] = round(activity['distance'] / 1609, 2)
    activity['average_pace'] = format_pace(activity['distance'], activity['moving_time'])
    activity['moving_time'] = format_time(activity['moving_time'])
    activity['moving_time_minutes'] = activity['moving_time']
    activity['average_cadence'] = int(round(activity['average_cadence'] * 2, 0))
    
    activity['average_speed'] = round(activity['average_speed'], 1)
    activity['max_speed'] = round(activity['max_speed'], 1)
    activity['max_heartrate'] = round(activity['max_heartrate'])
    activity['average_heartrate'] = round(activity['average_heartrate'])
    activity['kilojoules'] = round(activity['kilojoules'])    

    
    

    return render_template("activity.html", activity=activity)

if __name__ == '__main__':
    app.run(debug=True, port=5555)