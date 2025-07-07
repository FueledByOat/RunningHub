# runnervision_db_utils.py

"""Database utilities Runnervision."""

import pandas as pd
import sqlite3
from typing import Optional, Tuple
import uuid
from datetime import datetime
import json

def create_analysis_session(
    conn: sqlite3.Connection, 
    user_id: str, 
    video_duration_seconds: float,
    total_frames: int,
    fps: float,
    analysis_type: str,
    analysis_date: Optional[datetime] = None
) -> str:
    """Creates a new analysis session and returns the session UUID."""
    session_id = str(uuid.uuid4())
    if analysis_date is None:
        analysis_date = datetime.now().isoformat()
    else:
        analysis_date = analysis_date.isoformat()
    
    query = """
        INSERT INTO runnervision_analysis_sessions 
        (session_id, user_id, analysis_date, video_duration_seconds, total_frames, fps, analysis_type)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (
        session_id, user_id, analysis_date, 
        video_duration_seconds, total_frames, fps, analysis_type
    ))
    conn.commit()
    return session_id

def insert_rear_metrics_raw(conn: sqlite3.Connection, session_id: str, df: pd.DataFrame) -> int:
    """Inserts rear metrics DataFrame into database. Returns number of rows inserted."""
    df_clean = df.copy()
    
    # Handle boolean conversions - SQLite uses 0/1 for boolean
    bool_columns = [
        'left_foot_crossover', 'right_foot_crossover', 'left_knee_valgus', 
        'left_knee_varus', 'right_knee_valgus', 'right_knee_varus',
        'left_wrist_crossover', 'right_wrist_crossover', 'stance_phase_detected'
    ]
    for col in bool_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({
                'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 
                1: 1, 0: 0, '1': 1, '0': 0
            })
    
    # Add session_id column
    df_clean['session_id'] = session_id
    
    # Get database column names from schema
    db_columns = [
        'session_id', 'timestamp', 'frame_number',
        'left_foot_crossover', 'right_foot_crossover',
        'left_distance_from_midline', 'right_distance_from_midline',
        'step_width', 'symmetry',
        'hip_drop_value', 'hip_drop_direction', 'hip_drop_severity',
        'pelvic_tilt_angle', 'pelvic_tilt_elevated_side', 'pelvic_tilt_severity',
        'pelvic_tilt_normalized',
        'left_knee_valgus', 'left_knee_varus', 'right_knee_valgus', 'right_knee_varus',
        'left_knee_normalized_deviation', 'right_knee_normalized_deviation',
        'knee_severity_left', 'knee_severity_right',
        'left_ankle_inversion_value', 'right_ankle_inversion_value',
        'left_ankle_normalized_value', 'right_ankle_normalized_value',
        'left_ankle_pattern', 'right_ankle_pattern',
        'left_ankle_severity', 'right_ankle_severity',
        'left_ankle_angle', 'right_ankle_angle',
        'vertical_elbow_diff', 'normalized_vertical_diff',
        'left_elbow_angle', 'right_elbow_angle',
        'normalized_shoulder_diff', 'normalized_shoulder_width',
        'arm_height_symmetry', 'elbow_angle_left', 'elbow_angle_right',
        'left_wrist_crossover', 'right_wrist_crossover', 'shoulder_rotation',
        'stance_phase_detected', 'stance_foot', 'stance_confidence'
    ]
    
    # Select only columns that exist in both DataFrame and database
    available_columns = [col for col in db_columns if col in df_clean.columns]
    df_insert = df_clean[available_columns]
    
    # Replace NaN with None for proper NULL handling
    df_insert = df_insert.where(pd.notnull(df_insert), None)
    
    # Use pandas to_sql for efficient insertion
    df_insert.to_sql('rear_metrics_raw', conn, if_exists='append', index=False)
    
    return len(df_insert)

def insert_side_metrics_raw(conn: sqlite3.Connection, session_id: str, df: pd.DataFrame) -> int:
    """Inserts side metrics DataFrame into database. Returns number of rows inserted."""
    df_clean = df.copy()
    
    # Handle boolean conversions - SQLite uses 0/1 for boolean
    bool_columns = [
        'foot_landing_is_under_center_of_mass', 'trunk_angle_is_optimal',
        'stance_phase_detected', 'stance_phase_detected_velocity'
    ]
    for col in bool_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({
                'TRUE': 1, 'FALSE': 0, True: 1, False: 0,
                1: 1, 0: 0, '1': 1, '0': 0
            })
    
    # Handle JSON columns (arm_swing_recommendations)
    if 'arm_swing_recommendations' in df_clean.columns:
        df_clean['arm_swing_recommendations'] = df_clean['arm_swing_recommendations'].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
        )
    
    # Add session_id
    df_clean['session_id'] = session_id
    
    # Get database column names from schema
    db_columns = [
        'session_id', 'timestamp', 'frame_number',
        'strike_pattern', 'strike_confidence', 'vertical_difference',
        'strike_foot_angle', 'strike_ankle_angle', 'strike_landing_stiffness',
        'foot_landing_position_category', 'foot_landing_distance_from_center_in_cm',
        'foot_landing_is_under_center_of_mass',
        'trunk_angle_degrees', 'trunk_angle_is_optimal', 'trunk_angle_assessment',
        'trunk_angle_assessment_detail', 'trunk_angle_confidence',
        'upper_arm_angle', 'elbow_angle', 'hand_position',
        'arm_swing_amplitude', 'arm_swing_overall_assessment', 'arm_swing_recommendations',
        'knee_angle_left', 'knee_angle_right',
        'stride_instantaneous_estimate_cm', 'stride_length_cm', 'normalized_stride_length',
        'stride_frequency', 'stride_assessment', 'stride_confidence',
        'stance_phase_detected', 'stance_foot', 'stance_confidence',
        'stance_phase_detected_velocity', 'stance_foot_velocity',
        'avg_contact_time_ms', 'ground_contact_efficiency_rating',
        'ground_contact_cadence_spm', 'vertical_oscillation_cm',
        'vertical_oscillation_efficiency_rating'
    ]
    
    # Select only available columns
    available_columns = [col for col in db_columns if col in df_clean.columns]
    df_insert = df_clean[available_columns]
    
    # Replace NaN with None
    df_insert = df_insert.where(pd.notnull(df_insert), None)
    
    # Use pandas to_sql for efficient insertion
    df_insert.to_sql('side_metrics_raw', conn, if_exists='append', index=False)
    
    return len(df_insert)

def calculate_rear_summary(conn: sqlite3.Connection, session_id: str) -> bool:
    """Calculate rear metrics summary statistics."""
    try:
        query = """
        INSERT OR REPLACE INTO rear_metrics_summary (
            session_id,
            left_foot_crossover_pct,
            right_foot_crossover_pct,
            avg_step_width,
            step_width_std,
            avg_symmetry,
            avg_hip_drop,
            hip_drop_std,
            hip_drop_severity_mode,
            avg_pelvic_tilt,
            pelvic_tilt_severity_mode,
            knee_valgus_left_pct,
            knee_valgus_right_pct,
            avg_knee_deviation_left,
            avg_knee_deviation_right,
            knee_severity_left_mode,
            knee_severity_right_mode,
            avg_ankle_angle_left,
            avg_ankle_angle_right,
            ankle_pattern_left_mode,
            ankle_pattern_right_mode,
            ankle_severity_left_mode,
            ankle_severity_right_mode,
            avg_elbow_angle_left,
            avg_elbow_angle_right,
            avg_arm_symmetry,
            wrist_crossover_left_pct,
            wrist_crossover_right_pct,
            shoulder_rotation_mode,
            stance_detection_reliability,
            avg_confidence
        )
       WITH base AS (
    SELECT 
        *,
        (SELECT AVG(step_width) FROM rear_metrics_raw WHERE session_id = ?) AS avg_step_width,
        (SELECT AVG(hip_drop_value) FROM rear_metrics_raw WHERE session_id = ?) AS avg_hip_drop
    FROM rear_metrics_raw
    WHERE session_id = ?
)
SELECT 
    ? as session_id,
    AVG(CAST(left_foot_crossover AS REAL)) * 100 as left_foot_crossover_pct,
    AVG(CAST(right_foot_crossover AS REAL)) * 100 as right_foot_crossover_pct,
    AVG(step_width) as avg_step_width,
    CASE 
        WHEN COUNT(step_width) > 1 THEN 
            SQRT(SUM((step_width - base.avg_step_width) * (step_width - base.avg_step_width)) / (COUNT(step_width) - 1))
        ELSE 0 
    END as step_width_std,
    AVG(symmetry) as avg_symmetry,
    AVG(hip_drop_value) as avg_hip_drop,
    CASE 
        WHEN COUNT(hip_drop_value) > 1 THEN 
            SQRT(SUM((hip_drop_value - base.avg_hip_drop) * (hip_drop_value - base.avg_hip_drop)) / (COUNT(hip_drop_value) - 1))
        ELSE 0 
    END as hip_drop_std,
    (SELECT hip_drop_severity FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY hip_drop_severity ORDER BY COUNT(*) DESC LIMIT 1) as hip_drop_severity_mode,
    AVG(pelvic_tilt_angle) as avg_pelvic_tilt,
    (SELECT pelvic_tilt_severity FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY pelvic_tilt_severity ORDER BY COUNT(*) DESC LIMIT 1) as pelvic_tilt_severity_mode,
    AVG(CAST(left_knee_valgus AS REAL)) * 100 as knee_valgus_left_pct,
    AVG(CAST(right_knee_valgus AS REAL)) * 100 as knee_valgus_right_pct,
    AVG(left_knee_normalized_deviation) as avg_knee_deviation_left,
    AVG(right_knee_normalized_deviation) as avg_knee_deviation_right,
    (SELECT knee_severity_left FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY knee_severity_left ORDER BY COUNT(*) DESC LIMIT 1) as knee_severity_left_mode,
    (SELECT knee_severity_right FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY knee_severity_right ORDER BY COUNT(*) DESC LIMIT 1) as knee_severity_right_mode,
    AVG(left_ankle_angle) as avg_ankle_angle_left,
    AVG(right_ankle_angle) as avg_ankle_angle_right,
    (SELECT left_ankle_pattern FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY left_ankle_pattern ORDER BY COUNT(*) DESC LIMIT 1) as ankle_pattern_left_mode,
    (SELECT right_ankle_pattern FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY right_ankle_pattern ORDER BY COUNT(*) DESC LIMIT 1) as ankle_pattern_right_mode,
    (SELECT left_ankle_severity FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY left_ankle_severity ORDER BY COUNT(*) DESC LIMIT 1) as ankle_severity_left_mode,
    (SELECT right_ankle_severity FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY right_ankle_severity ORDER BY COUNT(*) DESC LIMIT 1) as ankle_severity_right_mode,
    AVG(elbow_angle_left) as avg_elbow_angle_left,
    AVG(elbow_angle_right) as avg_elbow_angle_right,
    AVG(arm_height_symmetry) as avg_arm_symmetry,
    AVG(CAST(left_wrist_crossover AS REAL)) * 100 as wrist_crossover_left_pct,
    AVG(CAST(right_wrist_crossover AS REAL)) * 100 as wrist_crossover_right_pct,
    (SELECT shoulder_rotation FROM rear_metrics_raw WHERE session_id = ? 
     GROUP BY shoulder_rotation ORDER BY COUNT(*) DESC LIMIT 1) as shoulder_rotation_mode,
    AVG(CAST(stance_phase_detected AS REAL)) as stance_detection_reliability,
    AVG(stance_confidence) as avg_confidence
FROM base;
        """
        
        cursor = conn.cursor()
        cursor.execute(query, [session_id] * 13)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error calculating rear summary: {e}")
        return False

def calculate_side_summary(conn: sqlite3.Connection, session_id: str) -> bool:
    """Calculate side metrics summary statistics."""
    try:
        query = """
        INSERT OR REPLACE INTO side_metrics_summary (
            session_id,
            strike_pattern_mode,
            avg_strike_confidence,
            avg_foot_landing_distance,
            foot_under_com_pct,
            avg_trunk_angle,
            trunk_optimal_pct,
            trunk_assessment_mode,
            avg_trunk_confidence,
            avg_upper_arm_angle,
            avg_elbow_angle,
            hand_position_mode,
            arm_swing_assessment_mode,
            avg_stride_length,
            stride_length_std,
            avg_normalized_stride,
            avg_stride_frequency,
            stride_assessment_mode,
            avg_contact_time,
            avg_cadence,
            avg_vertical_oscillation,
            ground_contact_efficiency_mode,
            vertical_oscillation_efficiency_mode,
            stance_detection_reliability,
            avg_confidence
        )
        WITH base AS (
    SELECT *, 
           (SELECT AVG(stride_length_cm) 
            FROM side_metrics_raw 
            WHERE session_id = ?) AS avg_stride_length_cm
    FROM side_metrics_raw
    WHERE session_id = ?
)
SELECT 
    ? AS session_id,
    (SELECT strike_pattern FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY strike_pattern ORDER BY COUNT(*) DESC LIMIT 1) as strike_pattern_mode,
    AVG(strike_confidence) as avg_strike_confidence,
    AVG(foot_landing_distance_from_center_in_cm) as avg_foot_landing_distance,
    AVG(CAST(foot_landing_is_under_center_of_mass AS REAL)) * 100 as foot_under_com_pct,
    AVG(trunk_angle_degrees) as avg_trunk_angle,
    AVG(CAST(trunk_angle_is_optimal AS REAL)) * 100 as trunk_optimal_pct,
    (SELECT trunk_angle_assessment FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY trunk_angle_assessment ORDER BY COUNT(*) DESC LIMIT 1) as trunk_assessment_mode,
    AVG(trunk_angle_confidence) as avg_trunk_confidence,
    AVG(upper_arm_angle) as avg_upper_arm_angle,
    AVG(elbow_angle) as avg_elbow_angle,
    (SELECT hand_position FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY hand_position ORDER BY COUNT(*) DESC LIMIT 1) as hand_position_mode,
    (SELECT arm_swing_overall_assessment FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY arm_swing_overall_assessment ORDER BY COUNT(*) DESC LIMIT 1) as arm_swing_assessment_mode,
    AVG(stride_length_cm) as avg_stride_length,
    CASE 
        WHEN COUNT(stride_length_cm) > 1 THEN 
            SQRT(SUM((stride_length_cm - avg_stride_length_cm) * (stride_length_cm - avg_stride_length_cm)) / (COUNT(stride_length_cm) - 1))
        ELSE 0 
    END as stride_length_std,
    AVG(normalized_stride_length) as avg_normalized_stride,
    AVG(stride_frequency) as avg_stride_frequency,
    (SELECT stride_assessment FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY stride_assessment ORDER BY COUNT(*) DESC LIMIT 1) as stride_assessment_mode,
    AVG(avg_contact_time_ms) as avg_contact_time,
    AVG(ground_contact_cadence_spm) as avg_cadence,
    AVG(vertical_oscillation_cm) as avg_vertical_oscillation,
    (SELECT ground_contact_efficiency_rating FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY ground_contact_efficiency_rating ORDER BY COUNT(*) DESC LIMIT 1) as ground_contact_efficiency_mode,
    (SELECT vertical_oscillation_efficiency_rating FROM side_metrics_raw WHERE session_id = ? 
     GROUP BY vertical_oscillation_efficiency_rating ORDER BY COUNT(*) DESC LIMIT 1) as vertical_oscillation_efficiency_mode,
    AVG(CAST(stance_phase_detected AS REAL)) as stance_detection_reliability,
    AVG(stance_confidence) as avg_confidence
FROM base
        """
        
        cursor = conn.cursor()
        cursor.execute(query, [session_id] * 10)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error calculating side summary: {e}")
        return False

def generate_summaries(conn: sqlite3.Connection, session_id: str) -> Tuple[bool, bool]:
    """
    Generates summary tables for both rear and side metrics.
    Returns tuple of (rear_success, side_success).
    """
    rear_success = calculate_rear_summary(conn, session_id)
    side_success = calculate_side_summary(conn, session_id)
    
    return rear_success, side_success