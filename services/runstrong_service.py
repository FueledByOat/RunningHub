# runstrong_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""


from typing import Dict, List, Any, Tuple, Optional

from services.base_service import BaseService
from utils import db_utils, format_utils, exception_utils
from config import Config

class RunStrongService(BaseService):
    """Service for RunStrong strength training operations."""

    # def get_exercises(self) -> List[Tuple[int, str]]:
    #     """Get all available exercises."""
    #     try:
    #         with self._get_connection() as conn:
    #             cursor = conn.cursor()
    #             return cursor.execute("SELECT * FROM exercises").fetchall()
    #     except Exception as e:
    #         self.logger.error(f"Error getting exercises: {e}")
    #         raise exception_utils.DatabaseError(f"Failed to get exercises: {e}")
        
    def add_exercise(self, data: dict) -> None:
        """Add single exercise to db."""
        try:
            with self._get_connection() as conn:
                db_utils.add_exercise(conn, data)
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
    
    # def get_routine_exercises(self, routine_id: int) -> List[Tuple[int, str]]:
    #     """Get exercises for a specific routine."""
    #     try:
    #         with self._get_connection() as conn:
    #             cursor = conn.cursor()
    #             cursor.execute("""
    #                 SELECT e.id, e.name
    #                 FROM routine_exercises re
    #                 JOIN exercises e ON e.id = re.exercise_id
    #                 WHERE re.routine_id = ?
    #                 ORDER BY re.order_index ASC
    #             """, (routine_id,))
    #             return cursor.fetchall()
    #     except Exception as e:
    #         self.logger.error(f"Error getting routine exercises for {routine_id}: {e}")
    #         raise exception_utils.DatabaseError(f"Failed to get routine exercises: {e}")
    # Late Night C team section

    def get_exercises(self) -> List[Tuple[int, str]]:
        """Get all available exercises."""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_all_exercises(conn)
            return result
        except Exception as e:
            self.logger.error(f"Error getting all exercises: {e}")
            return []

    def get_exercise_by_id(self, exercise_id: int) -> Optional[Tuple[int, str]]:
        """Get exercise by ID."""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_exercise_by_id(conn, exercise_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting exercise {exercise_id}: {e}")
            return None

    def get_all_routines(self) -> List[Tuple[int, str]]:
        """Get all available routines."""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_all_routines(conn)
            return result
        except Exception as e:
            self.logger.error(f"Error getting all routines: {e}")
            return []

    def get_routine_by_id(self, routine_id: int) -> Optional[Tuple[int, str]]:
        """Get routine by ID."""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_routine_by_id(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting routine {routine_id}: {e}")
            return None

    def create_routine(self, name: str) -> Optional[int]:
        """Create new routine and return routine ID."""
        try:
            with self._get_connection() as conn:
                result = db_utils.create_routine(conn, name)
            return result
        except Exception as e:
            self.logger.error(f"Error creating routine {name}: {e}")
            return None

    def add_exercise_to_routine(self, routine_id: int, exercise_id: int, sets: int, 
                        reps: int, load_lbs: float, order_index: int, notes: str = '') -> bool:
        """Add exercise to routine."""
        try:
            with self._get_connection() as conn:
                result = db_utils.add_exercise_to_routine(conn, routine_id, exercise_id, sets, 
                        reps, load_lbs, order_index, notes)
            return result
        except Exception as e:
            self.logger.error(f"Error adding exercise to routine {routine_id}: {e}")
            return False

    def get_routine_exercises(self, routine_id: int) -> List[Dict]:
        """Get all exercises for a specific routine"""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_routine_exercises(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting exercises for routine {routine_id}: {e}")
            return []

    def delete_routine(self, routine_id: int) -> bool:
        """Delete a routine and all associated exercises"""
        try:
            with self._get_connection() as conn:
                result = db_utils.delete_routine(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error deleting routine {routine_id}: {e}")
            return False

    def save_workout_performance(self, routine_id: int, exercise_id: int, workout_date: str,
                        planned_sets: int, actual_sets: int, planned_reps: int,
                        actual_reps: int, planned_load_lbs: float, actual_load_lbs: float,
                        notes: str = '', completion_status: str = 'completed') -> Optional[int]:
        """Save workout performance data"""
        try:
            with self._get_connection() as conn:
                result = db_utils.save_workout_performance(conn, routine_id, exercise_id, workout_date,
                        planned_sets, actual_sets, planned_reps,
                        actual_reps, planned_load_lbs, actual_load_lbs,
                        notes, completion_status)
            return result
        except Exception as e:
            self.logger.error(f"Error saving workout data for routine {routine_id}, exercise {exercise_id}: {e}")
            return None

    def get_workout_history(self, routine_id: int) -> List[Dict]:
        """Get workout history for a specific routine"""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_workout_history(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting history for routine {routine_id}: {e}")
            return []

    def get_workout_performance_by_date(self, routine_id: int, workout_date: str) -> List[Dict]:
        """Get workout performance for a specific routine and date"""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_workout_performance_by_date(conn, routine_id, workout_date)
            return result
        except Exception as e:
            self.logger.error(f"Error getting history for routine {routine_id} on {workout_date}: {e}")
            return []

    def get_exercise_progress(self, exercise_id: int, limit: int = 10) -> List[Dict]:
        """Get progress history for a specific exercise"""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_exercise_progress(conn, exercise_id, limit)
            return result
        except Exception as e:
            self.logger.error(f"Error getting progress history for exercise {exercise_id}: {e}")
            return []

    def get_recent_workouts(self, limit: int = 10) -> List[Dict]:
        """Get recent workout sessions"""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_recent_workouts(conn, limit)
            return result
        except Exception as e:
            self.logger.error(f"Error getting recent workout data: {e}")
            return []

    def get_workout_stats(self, routine_id: int = None) -> Dict:
        """Get workout statistics"""
        try:
            with self._get_connection() as conn:
                result = db_utils.get_workout_stats(conn, routine_id)
            return result
        except Exception as e:
            self.logger.error(f"Error getting workout stats for routine {routine_id}: {e}")
            return {}

    def initialize_runstrong_database(self) -> bool:
        """Initialize the database with all required tables"""
        try:
            with self._get_connection() as conn:
                result = db_utils.initialize_runstrong_database(conn)
                self.logger.info("Runstrong Database Initialized!")
            return result
        except Exception as e:
            self.logger.error(f"Error Initializing Runstrong Database: {e}")
            return False