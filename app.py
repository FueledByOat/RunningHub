from flask import Flask, render_template
import sqlite3
from dash_app import create_dash_app

app = Flask(__name__)

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
activity_id = 2058321970

# Initialize Dash app
create_dash_app(app, get_latest_polyline(activity_id), activity_id, DB_PATH)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/activity")
def single_activity():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM activities ORDER BY start_date DESC LIMIT 1")
    activity = cur.fetchone()
    conn.close()

    return render_template("single_activity.html", activity=activity)

if __name__ == '__main__':
    app.run(debug=True, port=5555)