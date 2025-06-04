# dash_apps/density/utils.py
"""
Utility functions for the density Dash application.
"""
import sqlite3
import polyline

def decode_polyline(p):
    """Decode a polyline string to coordinates."""
    try:
        return polyline.decode(p)
    except Exception:
        return []

def get_polylines_since(start_date, db_path):
    """Get all activity polylines since a given start date."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT map_summary_polyline FROM activities WHERE start_date >= ?", (start_date,))
    rows = cur.fetchall()
    conn.close()
    return [decode_polyline(row[0]) for row in rows if row[0]]

def get_latest_starting_coords(db_path):
    """Get the starting coordinates of the most recent activity."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""SELECT start_latlng FROM activities
                WHERE start_latlng is not null
                and type in ("Run", "Bike")
                order by start_date desc
                limit 1""")
    try:
        rows = cur.fetchall()[0][0].strip("[]")
        lat, lon = map(float, rows.split(","))
    except:
        lat, lon = (35.6764, 139.6500)  # Default to Tokyo
    conn.close()
    return (lat, lon)