# runstrong_db_utils.py
"""Database utilities RunStrong."""

import json
import math
import logging
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
import datetime
from datetime import timedelta

from utils.db import db_utils

logger = logging.getLogger(__name__)

def update_routine_name(conn: sqlite3.Connection, routine_id: int, name: str):
    """Update routine name."""
    conn.execute("UPDATE workout_routines SET name = ? WHERE id = ?", (name, routine_id))

def clear_routine_exercises(conn: sqlite3.Connection, routine_id: int):
    """Remove all exercises from a routine."""
    conn.execute("DELETE FROM routine_exercises WHERE routine_id = ?", (routine_id,))

def get_exercise_max_loads(conn: sqlite3.Connection) -> dict:
    """Get max load per exercise from performance history."""
    cursor = conn.execute("""
        SELECT exercise_id, MAX(actual_load_lbs) as max_load
        FROM workout_performance 
        WHERE actual_load_lbs IS NOT NULL AND actual_load_lbs > 0
        GROUP BY exercise_id
    """)
    results = cursor.fetchall()
    return {row['exercise_id']: row['max_load'] for row in results}

# def delete_routine(conn: sqlite3.Connection, routine_id: int):
#     """Delete a workout routine."""
#     conn.execute("DELETE FROM workout_routines WHERE id = ?", (routine_id,))

# def get_routine_by_id(conn: sqlite3.Connection, routine_id: int) -> dict:
#     """Get a specific routine by ID."""
#     cursor = conn.execute("SELECT * FROM workout_routines WHERE id = ?", (routine_id,))
#     result = cursor.fetchone()
#     return dict(result) if result else {}

def get_routine_name_datecreated(conn: sqlite3.Connection) -> list:
    """Get all workout routines."""
    cursor = conn.execute("""
        SELECT id, name, date_created 
        FROM workout_routines 
        ORDER BY name
    """)
    return [dict(row) for row in cursor.fetchall()]

# Exercise functions
def get_all_exercises(conn: sqlite3.Connection) -> List[Dict]:
    """Get all exercises from the database"""
    cursor = conn.execute("SELECT * FROM exercises ORDER BY name")
    result = cursor.fetchall()
    return db_utils.dicts_from_rows(result)

def get_exercise_by_id(conn: sqlite3.Connection, exercise_id: int) -> Optional[Dict]:
    """Get a specific exercise by ID"""
    cursor = conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,))
    result = cursor.fetchone()
    return db_utils.dict_from_row(result)

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
    return db_utils.dicts_from_rows(result)

def get_routine_by_id(conn: sqlite3.Connection, routine_id: int) -> Optional[Dict]:
    """Get a specific routine by ID"""
    cursor = conn.execute("SELECT * FROM workout_routines WHERE id = ?", (routine_id,))
    result = cursor.fetchone()
    return db_utils.dict_from_row(result)

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
    return db_utils.dicts_from_rows(exercises)

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

def save_workout_performance_bulk(conn: sqlite3.Connection, routine_id: int, workout_date: str, exercises: List[Dict]) -> bool:
    """
    Saves a list of workout performance records in a single database transaction.
    Returns True on success.
    """
    cursor = conn.cursor()
    try:
        for exercise_data in exercises:
            cursor.execute("""
                INSERT INTO workout_performance 
                (routine_id, exercise_id, workout_date, planned_sets, actual_sets,
                 planned_reps, actual_reps, planned_load_lbs, actual_load_lbs,
                 notes, completion_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                routine_id, 
                exercise_data.get('exercise_id'), 
                workout_date,
                exercise_data.get('planned_sets'), 
                exercise_data.get('actual_sets'),
                exercise_data.get('planned_reps'), 
                exercise_data.get('actual_reps'),
                exercise_data.get('planned_load_lbs'), 
                exercise_data.get('actual_load_lbs'),
                exercise_data.get('notes'), 
                exercise_data.get('completion_status', 'completed')
            ))
        
        # Commit all inserts at once
        conn.commit()
        logger.info(f"Successfully saved {len(exercises)} performance records for routine {routine_id} on {workout_date}.")
        return True

    except sqlite3.Error as e:
        # If any insert fails, roll back all of them
        conn.rollback()
        logger.error(f"Database error during bulk save for routine {routine_id}. Rolling back. Error: {e}")
        # Re-raise the exception to be caught by the service layer
        raise e

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
    return db_utils.dicts_from_rows(history)

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
    return db_utils.dicts_from_rows(performance)

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
    return db_utils.dicts_from_rows(progress)

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
    return db_utils.dicts_from_rows(workouts)

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
    return db_utils.dict_from_row(stats)

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

        # Create weekly_training_summary table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weekly_training_summary (
    id INTEGER PRIMARY KEY,
    week_start_date DATE,
    total_volume FLOAT,
    total_sessions INTEGER,
    muscle_group_distribution TEXT, -- JSON
    avg_completion_rate FLOAT,
    training_stress_score FLOAT
)
    ''')
    

        # Create muscle_group_fatigue table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS muscle_group_fatigue (
    id INTEGER PRIMARY KEY,
    muscle_group TEXT UNIQUE,
    last_trained_date DATE,
    volume_7day FLOAT,
    volume_14day FLOAT,
    recovery_score FLOAT, -- calculated metric
    updated_at TIMESTAMP
)
    ''')
    

        # Create exercise_progression table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercise_progression (
    id INTEGER PRIMARY KEY,
    exercise_id INTEGER,
    current_1rm_estimate FLOAT,
    volume_trend_30day FLOAT,
    last_pr_date DATE,
    progression_rate FLOAT, -- % improvement per week
    stall_indicator BOOLEAN,
    FOREIGN KEY(exercise_id) REFERENCES exercises(id)
)
    ''')
    
    
    conn.commit()
    return True

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

# Runstrong Dashboard / Fatigue Tracking 

def calculate_training_volume(sets: int, reps: int, load: float) -> float:
    return sets * reps * load

def get_muscle_groups_from_exercise(exercise_data: dict) -> List[str]:
    """Extract muscle groups from exercise data, handling both list and string formats"""
    muscles = []
    
    # Handle primary_muscles
    if 'primary_muscles' in exercise_data and exercise_data['primary_muscles']:
        if isinstance(exercise_data['primary_muscles'], list):
            muscles.extend(exercise_data['primary_muscles'])
        else:
            muscles.extend([m.strip() for m in str(exercise_data['primary_muscles']).split(',')])
    
    # Handle secondary_muscles
    if 'secondary_muscles' in exercise_data and exercise_data['secondary_muscles']:
        if isinstance(exercise_data['secondary_muscles'], list):
            muscles.extend(exercise_data['secondary_muscles'])
        else:
            muscles.extend([m.strip() for m in str(exercise_data['secondary_muscles']).split(',')])
    
    # Handle muscle_groups
    if 'muscle_groups' in exercise_data and exercise_data['muscle_groups']:
        if isinstance(exercise_data['muscle_groups'], list):
            muscles.extend(exercise_data['muscle_groups'])
        else:
            muscles.append(exercise_data['muscle_groups'])
    
    return list(set(muscles))

def calculate_recovery_score(days_since_last: int, volume_7day: float, volume_14day: float) -> int:
    """
    Calculates a recovery score based on time since last workout and training load ratio.
    This version uses a tiered, rule-based system for clearer, more stable results.
    The final score is rounded to the nearest integer.
    """
    # Ensure inputs are valid numbers to prevent errors
    days_since_last = int(days_since_last or 0)
    volume_7day = float(volume_7day or 0.0)
    volume_14day = float(volume_14day or 0.0)

    # 1. Time Component: Recovery from being recently trained (Max 70 points)
    # Every day of rest adds 10 points to recovery, capping at 7 days.
    time_factor = min(days_since_last * 10, 70)

    # 2. Volume Component: Calculate Acute:Chronic Workload Ratio (ACWR)
    # This ratio compares the last 7 days of training load to the last 14.
    if volume_14day == 0:
        volume_ratio = 0
    else:
        # To avoid division by zero if there's volume in the last 14 days but not the last 7
        acute_load = volume_7day / 7
        chronic_load = volume_14day / 14
        if chronic_load == 0:
            volume_ratio = 2.0 # Assign a high ratio if there's acute load but no chronic load
        else:
            volume_ratio = acute_load / chronic_load

    # 3. Tiered Penalty System (Replaces the complex math.tanh function)
    volume_penalty = 0
    if volume_ratio > 1.5:  # Danger Zone: Sharp increase in load
        volume_penalty = 40
    elif volume_ratio > 1.2: # Caution Zone: Significant increase
        volume_penalty = 20
    elif volume_ratio > 1.0: # Slight increase
        volume_penalty = 5
    # If ratio is < 1.0, it means load is stable or decreasing, so no penalty.

    # 4. Final Score Calculation
    # A base score of 30 represents a baseline level of 'ready'.
    # This is increased by rest (time_factor) and decreased by recent hard training (volume_penalty).
    recovery_score = (30 + time_factor) - volume_penalty

    # Clamp the score between 0 and 100 and round to the nearest whole number.
    final_score = round(max(0, min(100, recovery_score)))
    
    return final_score

def get_muscle_group_mapping() -> Dict[str, str]:
    """
    Maps standardized muscle names to their broad anatomical group.
    This is the second step after normalization.
    """
    return {
        # --- Upper Body ---
        'Back': 'Upper body',
        'Biceps': 'Upper body',
        'Chest': 'Upper body',
        'Forearms': 'Upper body',
        'Shoulders': 'Upper body',
        'Triceps': 'Upper body',
        'Upper Body': 'Upper body', # Passthrough

        # --- Lower Body ---
        'Ankles': 'Lower body',
        'Calves': 'Lower body',
        'Glutes': 'Lower body',
        'Hamstrings': 'Lower body',
        'Hips': 'Lower body',
        'Quadriceps': 'Lower body',
        'Lower Body': 'Lower body', # Passthrough

        # --- Core ---
        'Core': 'Core'
    }

def normalize_muscle_name(muscle: str) -> str:
    """
    (FINAL CORRECTED VERSION) Normalizes a comprehensive list of muscle names
    into a single, standardized name without using aggressive suffix removal.
    """
    if not isinstance(muscle, str):
        return ''
    # Standardize by making lowercase and removing extra spaces
    normalized = muscle.lower().strip()
    
    # --- Comprehensive mapping with both singular and plural variations ---
    muscle_map = {
        # Back
        'back': 'Back', 'lats': 'Back', 'lat': 'Back', 'latissimus dorsi': 'Back', 
        'rhomboids': 'Back', 'rhomboid': 'Back', 'traps': 'Back', 'trapezius': 'Back', 
        'upper back': 'Back', 'middle back': 'Back', 'lower back': 'Back', 
        'erector spinae': 'Back', 'levator scapulae': 'Back', 'lower trapezius': 'Back', 
        'middle trapezius': 'Back', 'upper trapezius': 'Back',
        
        # Chest
        'chest': 'Chest', 'pecs': 'Chest', 'pec': 'Chest', 'pectorals': 'Chest', 
        'pectoral': 'Chest', 'serratus anterior': 'Chest',

        # Shoulders
        'shoulders': 'Shoulders', 'shoulder': 'Shoulders', 'delts': 'Shoulders', 
        'delt': 'Shoulders', 'deltoid': 'Shoulders', 'deltoids': 'Shoulders',
        'anterior deltoid': 'Shoulders', 'lateral deltoid': 'Shoulders',
        'posterior deltoid': 'Shoulders', 'rear deltoid': 'Shoulders',
        'anterior deltoids': 'Shoulders', 'rear deltoids': 'Shoulders',
        
        # Arms
        'biceps': 'Biceps', 'bicep': 'Biceps', 'brachialis': 'Biceps', 
        'triceps': 'Triceps', 'tricep': 'Triceps', 
        'forearms': 'Forearms', 'forearm': 'Forearms', 'wrists': 'Forearms', 
        'wrist': 'Forearms', 'brachioradialis': 'Forearms',

        # Quads
        'quads': 'Quadriceps', 'quad': 'Quadriceps', 'quadriceps': 'Quadriceps', 
        'quadricep': 'Quadriceps',

        # Hamstrings
        'hams': 'Hamstrings', 'ham': 'Hamstrings', 'hamstrings': 'Hamstrings', 
        'hamstring': 'Hamstrings',

        # Glutes
        'glutes': 'Glutes', 'glute': 'Glutes', 'gluteus maximus': 'Glutes',
        'gluteus medius': 'Glutes', 'gluteus minimus': 'Glutes',
        'glute medius': 'Glutes', 'glute minimus': 'Glutes',
        'glute maximus': 'Glutes', 

        # Calves / Lower Leg
        'calves': 'Calves', 'calf': 'Calves', 'gastrocnemius': 'Calves',
        'soleus': 'Calves', 'achilles tendon': 'Calves', 'ankles': 'Ankles', 'ankle': 'Ankles',

        # Hips
        'hips': 'Hips', 'hip': 'Hips', 'hip flexors': 'Hips', 'hip flexor': 'Hips',
        'hip abductors': 'Hips', 'hip abductor': 'Hips', 'hip adductors': 'Hips',
        'hip adductor': 'Hips', 'hip stabilizers': 'Hips', 'hip stabilizer': 'Hips',

        # Core
        'core': 'Core', 'abs': 'Core', 'ab': 'Core', 'abdominals': 'Core', 
        'abdominal': 'Core', 'obliques': 'Core', 'oblique': 'Core',
        'transverse abdominis': 'Core', 'deep abdominals': 'Core',
        'lower abdominals': 'Core', 'rectus abdominis': 'Core',

        # Handle broad categories found in your data
        'upper body': 'Upper Body',
        'lower body': 'Lower Body',

        # Ignore overly generic terms by mapping them to an empty string
        'full body': '',
        'stabilizers': '',
        'stabilizer': '',
    }
    
    # Use the map, otherwise, return the original name, cleaned and title-cased
    return muscle_map.get(normalized, muscle.strip().title())

def update_muscle_group_fatigue(conn: sqlite3.Connection):
    """(DEBUGGING VERSION) Updates muscle group fatigue."""
    logger.info("--- STARTING FATIGUE UPDATE (DEBUG MODE) ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name, primary_muscles, secondary_muscles, muscle_groups FROM exercises")
    results = cursor.fetchall()
    
    unique_normalized_muscles = set()
    print("\n[DEBUG] Stage 1: Parsing and Normalizing Muscle Names from 'exercises' table...")
    for row in results:
        for key in ['primary_muscles', 'secondary_muscles', 'muscle_groups']:
            field = row[key]
            if not field: continue
            
            muscles_to_process = []
            try: # Assume JSON first
                loaded_muscles = json.loads(field)
                if isinstance(loaded_muscles, list):
                    muscles_to_process = loaded_muscles
                elif isinstance(loaded_muscles, str):
                     muscles_to_process = [m.strip() for m in loaded_muscles.split(',')]
            except (json.JSONDecodeError, TypeError): # Fallback to comma-separated string
                muscles_to_process = str(field).split(',')

            for muscle_name in muscles_to_process:
                if muscle_name and muscle_name.strip():
                    normalized = normalize_muscle_name(muscle_name)
                    print(f"  - Original: '{muscle_name.strip()}' -> Normalized: '{normalized}'")
                    unique_normalized_muscles.add(normalized)
    
    print(f"\n[DEBUG] Stage 2: Final unique muscle groups to be processed: {sorted(list(unique_normalized_muscles))}\n")

    today = datetime.datetime.now().date()
    seven_days_ago = today - timedelta(days=7)
    fourteen_days_ago = today - timedelta(days=14)

    for muscle_group in unique_normalized_muscles:
        if not muscle_group or not muscle_group.strip():
            continue

        # This query should be the corrected version from our previous discussion
        cursor.execute("""
            SELECT
                MAX(wp.workout_date) as last_trained,
                COALESCE(SUM(CASE WHEN wp.workout_date >= ? THEN
                    (COALESCE(wp.actual_sets, 0) * COALESCE(wp.actual_reps, 0) * COALESCE(wp.actual_load_lbs, 0))
                    ELSE 0 END), 0) as volume_7day,
                COALESCE(SUM(COALESCE(wp.actual_sets, 0) * COALESCE(wp.actual_reps, 0) * COALESCE(wp.actual_load_lbs, 0)), 0) as volume_14day
            FROM workout_performance wp
            JOIN exercises e ON wp.exercise_id = e.id
            WHERE (e.primary_muscles LIKE ? OR e.secondary_muscles LIKE ? OR e.muscle_groups LIKE ?)
            AND wp.workout_date >= ?
        """, (str(seven_days_ago), f'%{muscle_group}%', f'%{muscle_group}%', f'%{muscle_group}%', str(fourteen_days_ago)))
        
        result = cursor.fetchone()
        
        last_trained_str = result['last_trained'] if result else None
        if last_trained_str:
            last_trained_date = datetime.datetime.strptime(last_trained_str, '%Y-%m-%d').date()
            days_since = (today - last_trained_date).days
        else:
            days_since = 99 # A high number for untrained muscles
        
        volume_7day = result['volume_7day'] if result else 0
        volume_14day = result['volume_14day'] if result else 0
        
        recovery_score = calculate_recovery_score(days_since, volume_7day, volume_14day)
        
        cursor.execute("""
            INSERT OR REPLACE INTO muscle_group_fatigue
            (muscle_group, last_trained_date, volume_7day, volume_14day, recovery_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (muscle_group, last_trained_str, volume_7day, volume_14day, recovery_score, datetime.datetime.now()))
    
    conn.commit()
    logger.info("--- FINISHED FATIGUE UPDATE (DEBUG MODE) ---")


def get_daily_training_data(conn: sqlite3.Connection, muscle_group_filter: str = None, days: int = 7) -> List[Dict]:
    """Get daily training intensity with optional muscle group filtering"""
    cursor = conn.cursor()
    
    today = datetime.datetime.now().date()
    days_ago = today - timedelta(days=days)
    
    base_query = """
        SELECT 
            wp.workout_date,
            SUM(COALESCE(wp.actual_sets, 0) * COALESCE(wp.actual_reps, 0) * COALESCE(wp.actual_load_lbs, 0)) as daily_volume
        FROM workout_performance wp
    """
    
    # Add muscle group filtering if specified
    if muscle_group_filter and muscle_group_filter.lower() != 'all':
        muscle_mapping = get_muscle_group_mapping()
        filtered_muscles = [muscle for muscle, group in muscle_mapping.items() 
                          if group.lower() == muscle_group_filter.lower()]
        
        if filtered_muscles:
            placeholders = ','.join(['?' for _ in filtered_muscles])
            base_query += f"""
                JOIN exercises e ON wp.exercise_id = e.id
                WHERE wp.workout_date >= ? AND (
                    e.primary_muscles IN ({placeholders}) OR 
                    e.secondary_muscles IN ({placeholders})
                )
            """
            params = [days_ago] + filtered_muscles + filtered_muscles
        else:
            base_query += " WHERE wp.workout_date >= ?"
            params = [days_ago]
    else:
        base_query += " WHERE wp.workout_date >= ?"
        params = [days_ago]
    
    base_query += " GROUP BY wp.workout_date ORDER BY wp.workout_date"
    
    cursor.execute(base_query, params)
    training_data = {row['workout_date']: row['daily_volume'] for row in cursor.fetchall()}
    
    # Create daily entries for all days
    daily_training = []
    for i in range(days):
        date = today - timedelta(days=days-1-i)
        day_name = date.strftime('%a')
        volume = training_data.get(date.strftime('%Y-%m-%d'), 0)
        
        # Convert volume to intensity percentage (0-100)
        max_volume = 5000  # Adjust this threshold
        intensity = min(100, (volume / max_volume) * 100) if volume > 0 else 0
        
        daily_training.append({
            'day': day_name,
            'intensity': round(intensity),
            'hasTraining': volume > 0,
            'volume': volume
        })
    
    return daily_training

def get_fatigue_trend_data(conn: sqlite3.Connection, muscle_group_filter: str = None, days: int = 7) -> List[Dict]:
    """Get fatigue trend with optional muscle group filtering"""
    daily_training = get_daily_training_data(conn, muscle_group_filter, days)
    
    fatigue_trend = []
    cumulative_fatigue = 30  # Starting baseline
    
    for day_data in daily_training:
        if day_data['hasTraining']:
            fatigue_increase = day_data['intensity'] * 0.3
            cumulative_fatigue = min(100, cumulative_fatigue + fatigue_increase)
        else:
            cumulative_fatigue = max(0, cumulative_fatigue - 10)
        
        fatigue_trend.append({
            'day': day_data['day'],
            'fatigue': round(cumulative_fatigue)
        })
    
    return fatigue_trend

def get_fatigue_dashboard_data(conn: sqlite3.Connection, muscle_group_filter: str = None) -> dict:
    """
    (FINAL ROBUST VERSION)
    Gets all data required for the fatigue dashboard, applying filters at the
    database level and using case-insensitive mapping for robust categorization.
    """
    logger.info(f"Fetching dashboard data with filter: '{muscle_group_filter}'")
    cursor = conn.cursor()

    # Get the canonical mapping of standard muscle names to broad groups
    muscle_to_broad_group_map = get_muscle_group_mapping()
    
    # --- Database Query with Filtering ---
    query = "SELECT muscle_group, last_trained_date, volume_7day, volume_14day, recovery_score FROM muscle_group_fatigue"
    params = ()

    # If a filter is active, build the WHERE clause to select only the relevant muscles
    if muscle_group_filter and muscle_group_filter.lower() != 'all':
        # Find all standard muscle names that belong to the desired broad group
        filtered_muscles = [
            std_name for std_name, broad_group in muscle_to_broad_group_map.items() 
            if broad_group.lower() == muscle_group_filter.lower()
        ]
        
        # If we found any muscles for that group, add them to the query
        if filtered_muscles:
            placeholders = ','.join(['?' for _ in filtered_muscles])
            query += f" WHERE muscle_group IN ({placeholders})"
            params = tuple(filtered_muscles)
        else:
            # If the filter is valid but has no muscles (e.g., "Core" if no core exercises exist)
            # ensure the query returns no results.
            query += " WHERE 1=0"

    query += " ORDER BY recovery_score ASC"
    cursor.execute(query, params)
    
    db_results = cursor.fetchall()

    # --- Process Results with Case-Insensitive Mapping ---
    
    # Create a case-insensitive lookup dictionary for maximum robustness
    case_insensitive_lookup = {k.lower(): v for k, v in muscle_to_broad_group_map.items()}
    
    muscle_fatigue = []
    for row in db_results:
        stored_muscle_group = row['muscle_group']
        
        # Look up the broad group using the case-insensitive dictionary
        broad_group = case_insensitive_lookup.get(stored_muscle_group.lower(), 'Other')
        
        fatigue_level = round(max(0, min(100, 100 - row['recovery_score'])))

        muscle_fatigue.append({
            'muscle_group': stored_muscle_group,
            'last_trained': row['last_trained_date'],
            'volume_7day': row['volume_7day'],
            'volume_14day': row['volume_14day'],
            'recovery_score': row['recovery_score'],
            'fatigue_level': fatigue_level,
            'broad_group': broad_group,
        })

    # --- Assemble Final Payload ---

    # Calculate overall fatigue based on the (now filtered) list of muscles
    if muscle_fatigue:
        overall_fatigue = sum(m['fatigue_level'] for m in muscle_fatigue) / len(muscle_fatigue)
    else:
        overall_fatigue = 0 # Default to 0 if no data matches the filter

    # These functions also need to respect the filter
    daily_training = get_daily_training_data(conn, muscle_group_filter)
    fatigue_trend = get_fatigue_trend_data(conn, muscle_group_filter)

    # Note: The 'recommendation' and 'least_used_muscles' are added in the
    # service layer, which is the correct place for that logic. This function
    # correctly returns all the necessary source data.
    
    return {
        'overall_fatigue': round(overall_fatigue),
        'muscle_fatigue': muscle_fatigue,
        'daily_training': daily_training,
        'fatigue_trend': fatigue_trend,
        'available_filters': ['All', 'Upper body', 'Lower body', 'Core'],
    }


def run_daily_update(conn: sqlite3.Connection):
    """Main function to run daily fatigue updates"""
    logger.info("Starting daily fatigue tracking update...")
    try:
        
        # Update muscle group fatigue
        update_muscle_group_fatigue(conn)
        
        # Update weekly summary (simplified version)
        today = datetime.datetime.now()
        week_start = today - timedelta(days=today.weekday())
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO weekly_training_summary
            (week_start_date, total_volume, total_sessions, training_stress_score)
            VALUES (?, ?, ?, ?)
        """, (week_start.date(), 5000.0, 3, 75.0))
        
        conn.commit()
        logger.info("Fatigue tracking update complete.")
        
        # Return dashboard data
        return get_fatigue_dashboard_data(conn)
        
    except Exception as e:
        logger.error(f"Error in run_daily_update: {e}")
        import traceback
        logger.error(traceback.format_exc())

def get_or_create_freestyle_routine(conn: sqlite3.Connection) -> int:
    """
    Finds the 'Freestyle Workouts' routine or creates it if it doesn't exist.
    Returns the ID of the routine.
    """
    FREESTYLE_ROUTINE_NAME = "Freestyle Workouts"
    cursor = conn.cursor()
    
    # Check if the routine exists
    cursor.execute("SELECT id FROM workout_routines WHERE name = ?", (FREESTYLE_ROUTINE_NAME,))
    result = cursor.fetchone()
    
    if result:
        # Return existing routine's ID
        return result['id']
    else:
        # Create it if it doesn't exist
        cursor.execute(
            "INSERT INTO workout_routines (name, date_created) VALUES (?, ?)",
            (FREESTYLE_ROUTINE_NAME, datetime.datetime.now().date())
        )
        conn.commit()
        logger.info(f"Created special routine: '{FREESTYLE_ROUTINE_NAME}'")
        return cursor.lastrowid
    
def rebuild_fatigue_table(conn: sqlite3.Connection):
    """
    Drops, recreates, and repopulates the muscle_group_fatigue table from scratch.
    """
    logger.warning("Rebuilding the muscle_group_fatigue table...")
    cursor = conn.cursor()
    
    # 1. Drop the existing table
    cursor.execute("DROP TABLE IF EXISTS muscle_group_fatigue")
    
    # 2. Re-create the table from its schema definition
    cursor.execute('''
        CREATE TABLE muscle_group_fatigue (
            id INTEGER PRIMARY KEY,
            muscle_group TEXT UNIQUE,
            last_trained_date DATE,
            volume_7day FLOAT,
            volume_14day FLOAT,
            recovery_score FLOAT,
            updated_at TIMESTAMP
        )
    ''')
    
    # 3. Repopulate it with fresh data by calling the update function
    #    (Ensure you have applied the bug fix to this function first!)
    update_muscle_group_fatigue(conn)
    
    conn.commit()
    logger.info("Successfully rebuilt and repopulated the muscle_group_fatigue table.")
