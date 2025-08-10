# runstrong_db_utils.py

"""Database utilities RunStrong."""

import json
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
import datetime
from datetime import datetime, timedelta

from utils.db import db_utils

def get_all_exercises(conn: sqlite3.Connection) -> list:
    """Get all exercises."""
    cursor = conn.execute("select * from exercises")
    return [dict(row) for row in cursor.fetchall()] 

def get_all_exercises_with_load(conn: sqlite3.Connection) -> list:
    """Get all exercises with load data."""
    query = """SELECT
            e.name as exercise_name,
            e.type as exercise_type,
            SUM(emm.load_factor) as total_load
        FROM exercises e
        LEFT JOIN exercise_muscle_map emm ON e.id = emm.exercise_id
        GROUP BY e.id, e.name
        ORDER BY total_load DESC;"""
    cursor = conn.execute(query)
    return [dict(row) for row in cursor.fetchall()] 

def get_exercise_details_by_id(conn: sqlite3.Connection, exercise_id: int) -> Optional[Dict]:
    """Get a single exercise and its associated muscles by its ID."""
    query = """
        SELECT
            e.id,
            e.name,
            e.description,
            GROUP_CONCAT(mg.name || ' (' || emm.load_factor || ')', '; ') as muscles
        FROM exercises e
        LEFT JOIN exercise_muscle_map emm ON e.id = emm.exercise_id
        LEFT JOIN muscle_groups mg ON emm.muscle_group_id = mg.id
        WHERE e.id = ?
        GROUP BY e.id, e.name, e.description;
    """
    cursor = conn.execute(query, (exercise_id,))
    return db_utils.dict_from_row(cursor.fetchone())

def get_all_workout_sessions_with_details(conn: sqlite3.Connection) -> List[Dict]:
    """Get all workout sessions with a summary of exercises performed."""
    query = """
 SELECT
    ws.id,
    ws.session_date,
    ws.notes,
    COUNT(w_set.id) AS total_sets,
    (
        SELECT GROUP_CONCAT(sub_q.name, ', ')
        FROM (
            SELECT DISTINCT e.name
            FROM workout_sets w_set
            JOIN exercises e ON w_set.exercise_id = e.id
            WHERE w_set.session_id = ws.id
        ) as sub_q
    ) AS exercises_performed
FROM
    workout_sessions ws
LEFT JOIN
    workout_sets w_set ON ws.id = w_set.session_id
GROUP BY
    ws.id
ORDER BY
    ws.session_date DESC;

    """
    cursor = conn.execute(query)
    return db_utils.dicts_from_rows(cursor.fetchall())

def get_fatigue_summary(conn: sqlite3.Connection) -> List[Dict]:
    """Get all fatigue summary data."""
    cursor = conn.execute("SELECT entity_type, entity_name, fatigue_score FROM fatigue_summary ORDER BY fatigue_score DESC;")
    return db_utils.dicts_from_rows(cursor.fetchall())

def get_active_goals(conn: sqlite3.Connection) -> List[Dict]:
    """Get all active user goals with associated exercise names."""
    query = """
        SELECT
            ug.id,
            e.name as exercise_name,
            ug.goal_description,
			ug.target_value_lbs,
            min(ws.weight_lbs) as start_value_lbs,
            max(ws.weight_lbs) as current_value_lbs

        FROM user_goals ug
        JOIN exercises e ON ug.exercise_id = e.id
        JOIN workout_sets ws on e.id = ws.exercise_id
        WHERE ug.is_active = 1
		GROUP BY 1,2,3,4
    """
    cursor = conn.execute(query)
    return db_utils.dicts_from_rows(cursor.fetchall())

def create_workout_session(conn: sqlite3.Connection, session_date: str, notes: str) -> int:
    """Creates a new workout session and returns the new session ID."""
    query = "INSERT INTO workout_sessions (session_date, notes) VALUES (?, ?);"
    cursor = conn.execute(query, (session_date, notes))
    conn.commit()
    return cursor.lastrowid

def create_workout_set(conn: sqlite3.Connection, session_id: int, exercise_id: int, set_number: int, weight: float, reps: int, rpe: float):
    """Logs a single set for a given workout session."""
    query = """
        INSERT INTO workout_sets (session_id, exercise_id, set_number, weight_lbs, reps, rpe)
        VALUES (?, ?, ?, ?, ?, ?);
    """
    conn.execute(query, (session_id, exercise_id, set_number, weight, reps, rpe))

def get_daily_muscle_workload(conn: sqlite3.Connection, days_history: int = 28) -> List[Dict]:
    """
    Calculates the total workload for each muscle for each day in the specified history.
    Workload = weight * reps * RPE * load_factor
    """
    start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d %H:%M:%S')
    
    query = """
        SELECT
            date(ws.session_date) as workout_date,
            mg.id as muscle_group_id,
            mg.name as muscle_group_name,
            mg.body_part,
            SUM(w_set.weight_lbs * w_set.reps * COALESCE(w_set.rpe, 5) * emm.load_factor) as daily_workload
        FROM workout_sets w_set
        JOIN workout_sessions ws ON w_set.session_id = ws.id
        JOIN exercise_muscle_map emm ON w_set.exercise_id = emm.exercise_id
        JOIN muscle_groups mg ON emm.muscle_group_id = mg.id
        WHERE ws.session_date >= ?
        GROUP BY workout_date, muscle_group_id
        ORDER BY workout_date, muscle_group_name;
    """
    cursor = conn.execute(query, (start_date,))
    return db_utils.dicts_from_rows(cursor.fetchall())

def get_exercise_max_weights(conn: sqlite3.Connection) -> List[Dict]:
    """Get exercises with their maximum weights from completed workout sets."""
    query = """
    SELECT 
        e.name as exercise_name,
        e.type as exercise_type,
        MAX(ws.weight_lbs) as max_weight_lbs,
        COUNT(ws.id) as total_sets,
        MAX(wo_sesh.session_date) as last_max_date
    FROM exercises e
    INNER JOIN workout_sets ws ON e.id = ws.exercise_id
    INNER JOIN workout_sessions wo_sesh on ws.session_id = wo_sesh.id
    WHERE ws.weight_lbs IS NOT NULL 
        AND ws.weight_lbs > 0
    GROUP BY e.id, e.name, e.type
    ORDER BY max_weight_lbs DESC;
    """
    cursor = conn.execute(query)
    return [dict(row) for row in cursor.fetchall()]

def get_exercise_max_for_goals(conn: sqlite3.Connection, exercise_id: int) -> float:
    """Get maximum weight for a specific exercise for goals tracking."""
    query = """
    SELECT MAX(weight_lbs) as max_weight
    FROM workout_sets
    WHERE exercise_id = ? AND weight_lbs IS NOT NULL AND weight_lbs > 0
    """
    cursor = conn.execute(query, (exercise_id,))
    result = cursor.fetchone()
    return result['max_weight'] if result and result['max_weight'] else 0.0