# runstrong_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""


from typing import Dict, List, Any, Tuple

from services.base_service import BaseService
from utils import db_utils, format_utils, exception_utils
from config import Config

class RunStrongService(BaseService):
    """Service for RunStrong strength training operations."""

    def get_exercises(self) -> List[Tuple[int, str]]:
        """Get all available exercises."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                return cursor.execute("SELECT * FROM exercises").fetchall()
        except Exception as e:
            self.logger.error(f"Error getting exercises: {e}")
            raise exception_utils.DatabaseError(f"Failed to get exercises: {e}")
        
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
    
    def get_routine_exercises(self, routine_id: int) -> List[Tuple[int, str]]:
        """Get exercises for a specific routine."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.id, e.name
                    FROM routine_exercises re
                    JOIN exercises e ON e.id = re.exercise_id
                    WHERE re.routine_id = ?
                    ORDER BY re.order_index ASC
                """, (routine_id,))
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting routine exercises for {routine_id}: {e}")
            raise exception_utils.DatabaseError(f"Failed to get routine exercises: {e}")
