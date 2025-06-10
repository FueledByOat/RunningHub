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
    muscle_group TEXT,
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

def calculate_recovery_score(days_since_last: int, volume_7day: float, volume_14day: float) -> float:
    """Calculate recovery score based on time and volume"""
    # Time component: more days = better recovery (max 70 points)
    time_factor = min(days_since_last * 15, 70)
    
    # Volume component: higher recent volume = more fatigue
    volume_ratio = (volume_7day / 7) / (volume_14day / 14) # Switching to ACWR
    volume_penalty = 40 * math.tanh(volume_ratio - 1) # switching to expo smoothin
    volume_penalty = min(max(volume_penalty, 0), 50)
    
    # Base recovery + time factor - volume penalty
    recovery_score = max(0, min(100, time_factor + 30 - volume_penalty))
    return recovery_score

def get_muscle_group_mapping() -> Dict[str, str]:
    """Map individual muscles to broader muscle groups"""
    return {
        # Upper Body
        'Chest': 'Upper body',
        'Triceps': 'Upper body',
        'Anterior deltoids': 'Upper body',
        'Posterior deltoids': 'Upper body',
        'Lateral deltoids': 'Upper body',
        'Deltoids': 'Upper body',
        'Shoulders': 'Upper body',
        'Biceps': 'Upper body',
        'Lats': 'Upper body',
        'Latissimus dorsi': 'Upper body',
        'Rhomboids': 'Upper body',
        'Traps': 'Upper body',
        'Trapezius': 'Upper body',
        'Upper back': 'Upper body',
        'Middle back': 'Upper body',
        'Lower back': 'Upper body',
        'Erector spinae': 'Upper body',
        'Forearms': 'Upper body',
        'Wrists': 'Upper body',
        
        # Lower Body
        'Quadriceps': 'Lower body',
        'Quads': 'Lower body',
        'Hamstrings': 'Lower body',
        'Glutes': 'Lower body',
        'Gluteus maximus': 'Lower body',
        'Gluteus medius': 'Lower body',
        'Calves': 'Lower body',
        'Gastrocnemius': 'Lower body',
        'Soleus': 'Lower body',
        'Hip flexors': 'Lower body',
        'Hip abductors': 'Lower body',
        'Hip adductors': 'Lower body',
        'Tibialis anterior': 'Lower body',
        
        # Core (can be considered separate or part of upper)
        'Core': 'Core',
        'Abs': 'Core',
        'Abdominals': 'Core',
        'Obliques': 'Core',
        'Transverse abdominis': 'Core'
    }

def normalize_muscle_name(muscle: str) -> str:
    """Normalize muscle names for consistent matching"""
    muscle_map = {
        'quads': 'Quadriceps',
        'quadriceps': 'Quadriceps',
        'glutes': 'Glutes',
        'glute': 'Glutes',
        'hams': 'Hamstrings',
        'hamstrings': 'Hamstrings',
        'calves': 'Calves',
        'calf': 'Calves',
        'core': 'Core',
        'abs': 'Core',
        'chest': 'Chest',
        'pecs': 'Chest',
        'back': 'Back',
        'lats': 'Back',
        'shoulders': 'Shoulders',
        'delts': 'Shoulders',
        'arms': 'Arms',
        'biceps': 'Arms',
        'triceps': 'Arms'
    }
    
    normalized = muscle.lower().strip()
    return muscle_map.get(normalized, muscle.title())

def update_muscle_group_fatigue(conn: sqlite3.Connection):
    """Updated function to handle both database and direct exercise data"""
    logger.info("Updating muscle group fatigue...")
    cursor = conn.cursor()
    
    # Set row factory to return Row objects that support both index and name access
    cursor.row_factory = sqlite3.Row
    
    # Get muscle groups from exercises table
    cursor.execute("""
        SELECT primary_muscles, secondary_muscles, muscle_groups
        FROM exercises
        WHERE primary_muscles IS NOT NULL 
           OR secondary_muscles IS NOT NULL 
           OR muscle_groups IS NOT NULL
    """)
    
    results = cursor.fetchall()
    muscle_groups = set()
    
    # Parse muscle groups from database
    for row in results:
        for muscle_field in row:
            if muscle_field:
                try:
                    # Try parsing as JSON first
                    parsed = json.loads(muscle_field)
                    if isinstance(parsed, list):
                        muscle_groups.update([normalize_muscle_name(m) for m in parsed])
                    else:
                        muscle_groups.add(normalize_muscle_name(str(parsed)))
                except (json.JSONDecodeError, TypeError):
                    # Fallback to string parsing
                    if isinstance(muscle_field, str):
                        muscles = [normalize_muscle_name(m.strip()) for m in muscle_field.split(',') if m.strip()]
                        muscle_groups.update(muscles)
    
    # If no data in database, use default muscle groups
    if not muscle_groups:
        muscle_groups = {'Quadriceps', 'Glutes', 'Hamstrings', 'Calves', 'Core', 'Chest', 'Back', 'Shoulders', 'Arms'}
        logger.info("Using default muscle groups as no data found in database")
    
    muscle_groups = list(muscle_groups)
    logger.info(f"Processing muscle groups: {muscle_groups}")
    
    today = datetime.datetime.now().date()
    seven_days_ago = today - timedelta(days=7)
    fourteen_days_ago = today - timedelta(days=14)
    
    for muscle_group in muscle_groups:
        logger.info(f"Processing muscle group: {muscle_group}")
        
        # More flexible query that handles different data formats
        cursor.execute("""
            SELECT
                MAX(wp.workout_date) as last_trained,
                COALESCE(SUM(CASE WHEN wp.workout_date >= ? THEN
                    (COALESCE(wp.actual_sets, 0) * COALESCE(wp.actual_reps, 0) * COALESCE(wp.actual_load_lbs, 0))
                    ELSE 0 END), 0) as volume_7day,
                COALESCE(SUM(CASE WHEN wp.workout_date >= ? THEN
                    (COALESCE(wp.actual_sets, 0) * COALESCE(wp.actual_reps, 0) * COALESCE(wp.actual_load_lbs, 0))
                    ELSE 0 END), 0) as volume_14day
            FROM workout_performance wp
            JOIN exercises e ON wp.exercise_id = e.id
            WHERE (e.primary_muscles LIKE ? OR e.secondary_muscles LIKE ? OR e.muscle_groups LIKE ?)
            AND wp.workout_date >= ?
        """, (seven_days_ago, fourteen_days_ago,
            f'%{muscle_group}%', f'%{muscle_group}%', f'%{muscle_group}%', fourteen_days_ago))
        
        result = cursor.fetchone()
        
        # Fix: Use column names instead of indices
        if result and result['last_trained']:
            last_trained = result['last_trained']
            volume_7day = result['volume_7day'] or 0
            volume_14day = result['volume_14day'] or 0
            
            # Calculate days since last training
            if isinstance(last_trained, str):
                last_trained = datetime.datetime.strptime(last_trained, '%Y-%m-%d').date()
            days_since = (today - last_trained).days
        else:
            # No training data found - assume longer recovery time
            last_trained = fourteen_days_ago
            volume_7day = 0
            volume_14day = 0
            days_since = 14
        
        # Calculate recovery score
        recovery_score = calculate_recovery_score(days_since, volume_7day, volume_14day)
        
        # Insert/update muscle group fatigue
        cursor.execute("""
            INSERT OR REPLACE INTO muscle_group_fatigue
            (muscle_group, last_trained_date, volume_7day, volume_14day,
            recovery_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (muscle_group, last_trained, volume_7day, volume_14day,
            recovery_score, datetime.datetime.now()))
    
    conn.commit()
    logger.info("Muscle group fatigue update completed")


def get_daily_training_data(conn: sqlite3.Connection, muscle_group_filter: str = None, days: int = 7) -> List[Dict]:
    """Get daily training intensity with optional muscle group filtering"""
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row
    
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

def get_fatigue_dashboard_data(conn: sqlite3.Connection, muscle_group_filter: str = None) -> Dict:
    logger.info(f"Fetching fatigue dashboard data with filter: {muscle_group_filter}")
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row

    muscle_mapping = get_muscle_group_mapping()

    base_query = """
        SELECT muscle_group, last_trained_date, volume_7day, volume_14day, recovery_score
        FROM muscle_group_fatigue
    """

    if muscle_group_filter and muscle_group_filter.lower() in ['upper body', 'lower body', 'core']:
        filtered_muscles = [muscle for muscle, group in muscle_mapping.items()
                            if group.lower() == muscle_group_filter.lower()]
        if filtered_muscles:
            placeholders = ','.join(['?' for _ in filtered_muscles])
            query = base_query + f" WHERE muscle_group IN ({placeholders})"
            cursor.execute(query + " ORDER BY recovery_score ASC", filtered_muscles)
        else:
            cursor.execute(base_query + " ORDER BY recovery_score ASC")
    else:
        cursor.execute(base_query + " ORDER BY recovery_score ASC")

    muscle_fatigue = []
    for row in cursor.fetchall():
        last_trained = row['last_trained_date']
        fatigue_level = max(0, min(100, 100 - row['recovery_score']))

        muscle_fatigue.append({
            'muscle_group': row['muscle_group'],
            'last_trained': last_trained,
            'volume_7day': row['volume_7day'],
            'volume_14day': row['volume_14day'],
            'recovery_score': row['recovery_score'],
            'fatigue_level': fatigue_level,
            'broad_group': muscle_mapping.get(row['muscle_group'], 'Other'),
        })

    cursor.execute("""
        SELECT training_stress_score
        FROM weekly_training_summary
        ORDER BY week_start_date DESC
        LIMIT 7
    """)
    weekly_stress = [row['training_stress_score'] for row in cursor.fetchall()] or [60, 80, 45, 90, 30, 75, 85]

    overall_fatigue = (sum(m['fatigue_level'] for m in muscle_fatigue) / len(muscle_fatigue)) if muscle_fatigue else 50

    daily_training = get_daily_training_data(conn, muscle_group_filter)
    fatigue_trend = get_fatigue_trend_data(conn, muscle_group_filter)

    return {
        'overall_fatigue': round(overall_fatigue),
        'muscle_fatigue': muscle_fatigue,
        'weekly_stress': weekly_stress,
        'daily_training': daily_training,
        'fatigue_trend': fatigue_trend,
        'active_filter': muscle_group_filter,
        'available_filters': ['All', 'Upper body', 'Lower body', 'Core']
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