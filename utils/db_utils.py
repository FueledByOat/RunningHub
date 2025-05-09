# db_utils.py
"""Storing functions relating to database
cursor setup as well as queries to keep main app
clean of stray functions. """

import sqlite3
from dotenv import load_dotenv
import os

load_dotenv('secrets.env')
DB_PATH = os.getenv("DATABASE")

def dict_factory(cursor, row):
    """Convert database row objects to a dictionary"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_latest_activity(db_path = DB_PATH):
    """Retrieves latest Run or Bike activity ID as int"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""SELECT id FROM activities
                    WHERE type in ("Run", "Bike")
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