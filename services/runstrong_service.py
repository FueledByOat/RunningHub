# services/runstrong_service.py

"""
Service layer for running strength data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import math

from services.base_service import BaseService
from utils import exception_utils
from utils.db import runstrong_db_utils

class RunStrongService(BaseService):
    """Service for RunStrong strength training operations."""

    def get_exercises(self) -> List[Dict]:
        """Get all available exercises."""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_all_exercises(conn)
        except Exception as e:
            self.logger.error(f"Error getting all exercises: {e}")
            raise exception_utils.DatabaseError(f"Failed to get all exercises: {e}")
        
    def get_exercises_with_load(self) -> List[Dict]:
        """Get all available exercises with load values."""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_all_exercises_with_load(conn)
        except Exception as e:
            self.logger.error(f"Error getting all exercises with load data: {e}")
            raise exception_utils.DatabaseError(f"Failed to get all exercises with load data: {e}")        

    def get_exercise_details(self, exercise_id: int) -> Dict:
        """Get details for a specific exercise."""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_exercise_details_by_id(conn, exercise_id)
        except Exception as e:
            self.logger.error(f"Error getting details for exercise {exercise_id}: {e}")
            raise exception_utils.DatabaseError("Failed to get exercise details.")

    def get_workout_journal(self) -> List[Dict]:
        """Get all workout sessions for the journal."""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_all_workout_sessions_with_details(conn)
        except Exception as e:
            self.logger.error(f"Error getting workout journal: {e}")
            raise exception_utils.DatabaseError("Failed to get workout journal.")

    def get_fatigue_data(self) -> Dict[str, List]:
        """Get and structure fatigue data for the dashboard."""
        try:
            with self._get_connection() as conn:
                fatigue_list = runstrong_db_utils.get_fatigue_summary(conn)
            # Structure the data for easy rendering
            structured_fatigue = {
                'overall': [],
                'body_part': [],
                'muscle_group': []
            }
            for item in fatigue_list:
                if item['entity_type'] in structured_fatigue:
                    structured_fatigue[item['entity_type']].append(item)
            return structured_fatigue
        except Exception as e:
            self.logger.error(f"Error getting fatigue data: {e}")
            raise exception_utils.DatabaseError("Failed to get fatigue data.")

    def get_goals_with_progress(self) -> List[Dict]:
        """Get active goals and calculate their progress percentage."""
        try:
            with self._get_connection() as conn:
                goals = runstrong_db_utils.get_active_goals(conn)
            
            for goal in goals:
                start = goal['start_value_lbs']
                current = goal['current_value_lbs']
                target = goal['target_value_lbs']
                if target > start:
                    progress = ((current - start) / (target - start)) * 100
                    goal['progress'] = round(max(0, min(progress, 100))) # Clamp between 0-100
                else:
                    goal['progress'] = 0
            return goals
        except Exception as e:
            self.logger.error(f"Error getting goals: {e}")
            raise exception_utils.DatabaseError("Failed to get goals.")
        
    def log_new_workout(self, workout_data: Dict) -> int:
        """
        Logs a new workout session and all of its sets in a single transaction.
        Expects workout_data to have 'session_date', 'notes', and a 'sets' list.
        """
        try:
            with self._get_connection() as conn:
                # Create the session and get its ID
                session_id = runstrong_db_utils.create_workout_session(
                    conn,
                    workout_data.get('session_date'),
                    workout_data.get('notes')
                )

                # Loop through and log each set
                for i, workout_set in enumerate(workout_data.get('sets', [])):
                    runstrong_db_utils.create_workout_set(
                        conn,
                        session_id,
                        workout_set.get('exercise_id'),
                        i + 1,  # Set number
                        workout_set.get('weight'),
                        workout_set.get('reps'),
                        workout_set.get('rpe')
                    )
                
                conn.commit()
                return session_id
                
        except Exception as e:
            self.logger.error(f"Error logging new workout: {e}")
            # The context manager will handle the rollback on error
            raise exception_utils.DatabaseError("Failed to log new workout.")
        
    def _get_fatigue_interpretation(self, score: float) -> str:
        """Provides a human-readable interpretation of a given fatigue score."""
        if score < 20:
            return "Rested & Ready: Muscles are fully recovered. Excellent time for a high-intensity session."
        elif score < 40:
            return "Low Fatigue: Primed for performance. Optimal state for productive training."
        elif score < 60:
            return "Productive Fatigue: Training stimulus is being applied. Continue as planned, but monitor recovery."
        elif score < 80:
            return "High Fatigue: Accumulated stress is significant. Consider a lighter day or active recovery."
        else:
            return "Very High Fatigue: Pushing limits. Recovery is critical. Prioritize rest or very light activity to avoid overtraining."

    def _acwr_to_fatigue_score(self, acwr: float) -> float:
        """Converts an ACWR value to a 0-100 fatigue score. (No changes from before)"""
        # ... (This function remains the same as the previous version)
        if acwr < 0.8: return acwr * 25
        elif acwr <= 1.3: return 20 + (acwr - 0.8) * 100
        elif acwr <= 1.5: return 70 + (acwr - 1.3) * 75
        else: return min(100, 85 + (acwr - 1.5) * 50)

    def get_fatigue_dashboard_data(self) -> Dict:
        """
        Calculates a comprehensive, time-decayed fatigue analysis for the dashboard.
        
        This on-the-fly calculation includes:
        - Exponentially decayed workload for higher accuracy.
        - Fatigue scores for individual muscles, body groups, and overall.
        - Human-readable interpretations of all scores.
        - Data structured for categorical filtering (Overall, Lower, Upper, Core).
        """
        with self._get_connection() as conn:
            # Get raw daily workload for the last 28 days
            workload_data = runstrong_db_utils.get_daily_muscle_workload(conn, days_history=28)

        # --- 1. Calculate Time-Decayed Fatigue Score for Every Muscle ---
        today = datetime.now().date()
        DECAY_RATE = 0.96  # Daily workload retention rate (4% daily decay)
        
        workload_by_muscle = defaultdict(lambda: defaultdict(float))
        all_muscle_info = {}
        for row in workload_data:
            muscle_id = row['muscle_group_id']
            workload_by_muscle[muscle_id][datetime.strptime(row['workout_date'], '%Y-%m-%d').date()] = row['daily_workload']
            if muscle_id not in all_muscle_info:
                all_muscle_info[muscle_id] = {'name': row['muscle_group_name'], 'body_part': row['body_part']}

        muscle_scores = []
        for muscle_id, daily_loads in workload_by_muscle.items():
            acute_load, chronic_load = 0.0, 0.0
            
            for day, load in daily_loads.items():
                days_ago = (today - day).days
                decayed_load = load * math.pow(DECAY_RATE, days_ago)
                
                if days_ago < 28: chronic_load += decayed_load
                if days_ago < 7: acute_load += decayed_load
            
            # Use decayed totals instead of simple averages for ACWR
            acwr = (acute_load / 7.0) / (chronic_load / 28.0) if chronic_load > 0 else (acute_load / 7.0)
            fatigue_score = self._acwr_to_fatigue_score(acwr)
            
            muscle_scores.append({
                'name': all_muscle_info[muscle_id]['name'],
                'body_part': all_muscle_info[muscle_id]['body_part'],
                'score': fatigue_score
            })

        # --- 2. Structure Data for Each Category (Overall, Lower, Upper, Core) ---
        categories = {
            "overall": muscle_scores,
            "lower_body": [m for m in muscle_scores if m['body_part'] == 'Lower Body'],
            "upper_body": [m for m in muscle_scores if m['body_part'] == 'Upper Body'],
            "core": [m for m in muscle_scores if m['body_part'] == 'Core']
        }
        
        final_data = {}
        for cat_key, cat_muscles in categories.items():
            # Calculate Summary Score
            total_score = sum(m['score'] for m in cat_muscles)
            summary_score = total_score / len(cat_muscles) if cat_muscles else 0
            
            # Calculate Top/Bottom 5
            cat_muscles.sort(key=lambda x: x['score'], reverse=True)
            top_5 = cat_muscles[:5]
            least_5 = sorted(cat_muscles, key=lambda x: x['score'])[:5]
            
            # Calculate 7-Day Workload for the category
            cat_workload_summary = defaultdict(float)
            for i in range(7): cat_workload_summary[(today - timedelta(days=i)).strftime('%A, %b %d')] = 0
            
            for row in workload_data:
                day = datetime.strptime(row['workout_date'], '%Y-%m-%d').date()
                if (today - day).days < 7:
                    if cat_key == 'overall' or row['body_part'].replace(' ', '_').lower() == cat_key:
                        cat_workload_summary[day.strftime('%A, %b %d')] += row['daily_workload']

            seven_day_workload = [{'day': d, 'workload': w} for d, w in cat_workload_summary.items()]
            seven_day_workload.reverse()

            final_data[cat_key] = {
                "summary_score": summary_score,
                "interpretation": self._get_fatigue_interpretation(summary_score),
                "top_5_fatigued": top_5,
                "least_5_fatigued": least_5,
                "seven_day_workload": seven_day_workload
            }
            
        return final_data