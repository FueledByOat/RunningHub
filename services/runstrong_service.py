# runstrong_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""

import datetime
from typing import Dict, List, Any, Tuple, Optional

from services.base_service import BaseService
from utils import format_utils, exception_utils
from config import Config
from utils.db import db_utils
from utils.db import runstrong_db_utils

class RunStrongService(BaseService):
    """Service for RunStrong strength training operations."""

     
    def add_exercise(self, data: dict) -> None:
        """Add single exercise to db."""
        try:
            with self._get_connection() as conn:
                runstrong_db_utils.add_exercise(conn, data)
            self.logger.info(f"Added exercise routine: {data['name']} to database")
        except Exception as e:
            self.logger.error(f"Error getting exercises: {e}")
            raise exception_utils.DatabaseError(f"Failed to get exercises: {e}")
    
    def save_routine(self, routine_name: str, routine_exercises: List[Dict[str, Any]]) -> None:
        """Save a workout routine."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Save routine
                cursor.execute(
                    "INSERT INTO workout_routines (name) VALUES (?)", 
                    (routine_name,)
                )
                routine_id = cursor.lastrowid
                
                # Save routine exercises
                for item in routine_exercises:
                    cursor.execute("""
                        INSERT INTO routine_exercises (routine_id, exercise_id, order_index)
                        VALUES (?, ?, ?)
                    """, (routine_id, item['id'], item['order']))
                
                conn.commit()
                self.logger.info(f"Saved routine: {routine_name} with {len(routine_exercises)} exercises")
                
        except Exception as e:
            self.logger.error(f"Error saving routine {routine_name}: {e}")
            raise exception_utils.DatabaseError(f"Failed to save routine: {e}")
    
    def get_routines(self) -> List[Tuple[int, str]]:
        """Get all workout routines."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM workout_routines")
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting routines: {e}")
            raise exception_utils.DatabaseError(f"Failed to get routines: {e}")


    def get_exercises(self) -> List[Tuple[int, str]]:
        """Get all available exercises."""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_all_exercises(conn)
            return result
        except Exception as e:
            self.logger.error(f"Error getting all exercises: {e}")
            return []

    def get_exercise_by_id(self, exercise_id: int) -> Optional[Tuple[int, str]]:
        """Get exercise by ID."""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_exercise_by_id(conn, exercise_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting exercise {exercise_id}: {e}")
            return None

    def get_all_routines(self) -> List[Tuple[int, str]]:
        """Get all available routines."""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_all_routines(conn)
            return result
        except Exception as e:
            self.logger.error(f"Error getting all routines: {e}")
            return []

    def get_routine_by_id(self, routine_id: int) -> Optional[Tuple[int, str]]:
        """Get routine by ID."""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_routine_by_id(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting routine {routine_id}: {e}")
            return None

    def create_routine(self, name: str) -> Optional[int]:
        """Create new routine and return routine ID."""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.create_routine(conn, name)
            return result
        except Exception as e:
            self.logger.error(f"Error creating routine {name}: {e}")
            return None

    def add_exercise_to_routine(self, routine_id: int, exercise_id: int, sets: int, 
                        reps: int, load_lbs: float, order_index: int, notes: str = '') -> bool:
        """Add exercise to routine."""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.add_exercise_to_routine(conn, routine_id, exercise_id, sets, 
                        reps, load_lbs, order_index, notes)
            return result
        except Exception as e:
            self.logger.error(f"Error adding exercise to routine {routine_id}: {e}")
            return False

    def get_routine_exercises(self, routine_id: int) -> List[Dict]:
        """Get all exercises for a specific routine"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_routine_exercises(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting exercises for routine {routine_id}: {e}")
            return []

    def delete_routine(self, routine_id: int) -> bool:
        """Delete a routine and all associated exercises"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.delete_routine(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error deleting routine {routine_id}: {e}")
            return False

    def save_workout_performance_bulk(self, routine_id: int, workout_date: str, exercises: List[Dict]):
        """
        Service layer method to save a complete workout performance log.
        Handles the database connection and calls the bulk DB utility.
        """
        try:
            with self._get_connection() as conn:
                runstrong_db_utils.save_workout_performance_bulk(
                    conn, routine_id, workout_date, exercises
                )
        except Exception as e:
            # The DB layer will log the specific DB error. This logs the service-level failure.
            self.logger.error(f"Failed to save bulk workout performance for routine {routine_id}: {e}")
            # Propagate a generic error to the route layer
            raise exception_utils.DatabaseError(f"Failed to save workout: {e}")

    def get_workout_history(self, routine_id: int) -> List[Dict]:
        """Get workout history for a specific routine"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_workout_history(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting history for routine {routine_id}: {e}")
            return []

    def get_workout_performance_by_date(self, routine_id: int, workout_date: str) -> List[Dict]:
        """Get workout performance for a specific routine and date"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_workout_performance_by_date(conn, routine_id, workout_date)
            return result
        except Exception as e:
            self.logger.error(f"Error getting history for routine {routine_id} on {workout_date}: {e}")
            return []

    def get_exercise_progress(self, exercise_id: int, limit: int = 10) -> List[Dict]:
        """Get progress history for a specific exercise"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_exercise_progress(conn, exercise_id, limit)
            return result
        except Exception as e:
            self.logger.error(f"Error getting progress history for exercise {exercise_id}: {e}")
            return []

    def get_recent_workouts(self, limit: int = 10) -> List[Dict]:
        """Get recent workout sessions"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_recent_workouts(conn, limit)
            return result
        except Exception as e:
            self.logger.error(f"Error getting recent workout data: {e}")
            return []

    def get_workout_stats(self, routine_id: int = None) -> Dict:
        """Get workout statistics"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.get_workout_stats(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting workout stats for routine {routine_id}: {e}")
            return {}

    def initialize_runstrong_database(self) -> bool:
        """Initialize the database with all required tables"""
        try:
            with self._get_connection() as conn:
                result = runstrong_db_utils.initialize_runstrong_database(conn)
                self.logger.info("Runstrong Database Initialized!")
            return result
        except Exception as e:
            self.logger.error(f"Error Initializing Runstrong Database: {e}")
            return False
        
    def update_routine_name(self, routine_id: int, name: str):
        """Update routine name."""
        try:
            with self._get_connection() as conn:
                runstrong_db_utils.update_routine_name(conn, routine_id, name)
        except Exception as e:
            self.logger.error(f"Error updating routine name for ID {routine_id}: {e}")

    def clear_routine_exercises(self, routine_id: int):
        """Remove all exercises from a routine."""
        try:
            with self._get_connection() as conn:
                runstrong_db_utils.clear_routine_exercises(conn, routine_id)
        except Exception as e:
            self.logger.error(f"Error clearing exercises for routine {routine_id}: {e}")

    def get_exercise_max_loads(self) -> dict:
        """Get maximum load for each exercise from workout performance history."""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_exercise_max_loads(conn)
        except Exception as e:
            self.logger.error(f"Error fetching max exercise loads: {e}")
            return {}

    def get_routine_name_datecreated(self) -> list:
        """Get all workout routines."""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_routine_name_datecreated(conn)
        except Exception as e:
            self.logger.error(f"Error fetching all routines: {e}")
            return []
        
    def run_daily_update(self):
        try:
            with self._get_connection() as conn:
                runstrong_db_utils.run_daily_update(conn)
        except Exception as e:
            self.logger.error(f"Error in run_daily_update: {e}")

    def get_fatigue_dashboard_data(self, muscle_group_filter: str = None) -> Dict:
        """Get fatigue dashboard data with optional muscle group filtering"""
        try:
            with self._get_connection() as conn:
                return runstrong_db_utils.get_fatigue_dashboard_data(conn, muscle_group_filter)
        except Exception as e:
            self.logger.error(f"Error in get_fatigue_dashboard_data: {e}")
            return runstrong_db_utils.get_fallback_dashboard_data(muscle_group_filter)

    def update_weekly_summary(self, week_start: datetime.datetime):
        try:
            with self._get_connection() as conn:
                runstrong_db_utils.update_weekly_training_summary(conn, week_start)
        except Exception as e:
            self.logger.error(f"Error in update_weekly_summary: {e}")

    def get_recommendation(self, overall_fatigue):
        if overall_fatigue < 40:
            return "You're well recovered. Consider a hard or high-volume workout today."
        elif overall_fatigue < 70:
            return "You're moderately fatigued. A light workout or recovery session is ideal."
        else:
            return "High fatigue detected. Rest or active recovery is strongly recommended."

    def get_least_used_muscle_groups(self, muscle_fatigue, days_threshold=5):
        try:
            return sorted(
                (m for m in muscle_fatigue if m['fatigue_level'] < 40 and m.get('last_trained')),
                key=lambda m: m['last_trained']
            )[:3]
        except Exception as e:
            self.logger.warning(f"Unable to compute least used muscle groups: {e}")
            return []

    def save_freestyle_workout(self, workout_date: str, exercises: List[Dict]):
        """
        Saves an ad-hoc (freestyle) workout session.
        It finds/creates a special routine and logs the exercises under it.
        """
        try:
            with self._get_connection() as conn:
                # Get the ID for the special freestyle routine
                freestyle_routine_id = runstrong_db_utils.get_or_create_freestyle_routine(conn)
                
                # Use the existing bulk save function with the special ID
                runstrong_db_utils.save_workout_performance_bulk(
                    conn, freestyle_routine_id, workout_date, exercises
                )
                
        except Exception as e:
            self.logger.error(f"Failed to save freestyle workout: {e}")
            raise exception_utils.DatabaseError(f"Failed to save freestyle workout: {e}")