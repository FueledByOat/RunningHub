# dash_utils.py
"""Storing functions relating to 
the preparation and rendering of the 
dash app."""

import polyline
import json
import sqlite3
from dash import html, dcc
import numpy as np
import dash_leaflet as dl
import utils.db_utils as db_utils


def decode_polyline(polyline_str):
    try:
        return polyline.decode(polyline_str)  # List of (lat, lon)
    except Exception as e:
        print(f"Error decoding polyline: {e}")
        return []
    
def get_streams_data(activity_id, db_path = db_utils.DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT distance_data, heartrate_data, altitude_data, watts_data FROM streams WHERE activity_id = ?", (activity_id,))
    row = cur.fetchone()
    conn.close()

    if row:
        try:
            distance = json.loads(row[0]) if row[0] else []
            heartrate = json.loads(row[1]) if row[1] else []
            altitude = json.loads(row[2]) if row[2] else []
            power = json.loads(row[3]) if row[3] else []

            # What if the data returns None
            distance = [0 if i == None else i for i in distance]
            heartrate = [0 if i == None else i for i in heartrate]
            altitude = [0 if i == None else i for i in altitude]
            power = [0 if i == None else i for i in power]

            return distance, heartrate, altitude, power
        except Exception as e:
            print("Failed to load stream data:", e)
    return [], [], [], []

def interpolate_to_common_x(x_ref, y_raw, x_raw):
    """Interpolates y_raw over x_ref, assuming x_raw and y_raw are aligned."""
    if len(x_raw) < 2 or len(x_ref) < 2:
        return [None] * len(x_ref)
    return np.interp(x_ref, x_raw, y_raw).tolist()

