# services/calendar_service.py

import logging
from typing import Dict, Any, List

from services.base_service import BaseService
import utils.db.running_hub_db_utils as running_hub_db_utils
from utils import exception_utils

class CalendarService(BaseService):
    """Service for handling calendar and planned workout operations."""

    def get_planned_workouts_for_calendar(self, start_date, end_date) -> list[dict]:
        """
        Retrieves planned workouts in a date range and formats them for FullCalendar.
        """
        try:
            with self._get_connection() as conn:
                # Call the new date-filtered database function
                rows = running_hub_db_utils.get_planned_workouts_by_date_range(
                    conn, start_date, end_date
                )
            
            events = []
            for row in rows:
                # Business logic for color-coding events
                if row["success"] == "yes":
                    color = "#0c1559"  # Completed
                elif row["success"] == "no":
                    color = "#ffe066"  # Not completed
                else:
                    color = "#ccc"     # Planned

                events.append({
                    "id": row["id"],
                    "title": row["workout_name"] or row["workout_type"] or "Planned Workout",
                    "start": row["workout_date"],
                    "color": color
                })
            return events
        except Exception as e:
            self.logger.error(f"Error getting planned workouts for calendar: {e}")
            raise exception_utils.ServiceError("Failed to retrieve calendar events.")

    def save_planned_workout(self, workout_data: dict):
        """
        Saves a new or updated planned workout to the database.
        
        Args:
            workout_data: A dictionary containing the details of the workout.
        """
        try:
            # Prepare data for insertion
            workout_data['effort'] = int(workout_data['effort']) if workout_data.get('effort') else None
            workout_data['user_id'] = 1 # Assuming a single user for now

            with self._get_connection() as conn:
                running_hub_db_utils.insert_planned_workout(conn, workout_data)
                conn.commit()
            self.logger.info(f"Successfully saved planned workout for date: {workout_data.get('workout_date')}")
        except Exception as e:
            self.logger.error(f"Error saving planned workout: {e}")
            raise exception_utils.ServiceError("Failed to save workout.")

    def get_planned_workout(self, workout_id: int) -> dict | None:
        """Retrieves a single planned workout."""
        try:
            with self._get_connection() as conn:
                return running_hub_db_utils.get_planned_workout_by_id(conn, workout_id)
        except Exception as e:
            self.logger.error(f"Service error fetching workout {workout_id}: {e}")
            raise exception_utils.ServiceError("Failed to retrieve workout details.")

    def update_planned_workout(self, workout_data: dict) -> None:
        """Updates an existing planned workout."""
        try:
            # Prepare data for update
            workout_data['effort'] = int(workout_data['effort']) if workout_data.get('effort') else None
            
            with self._get_connection() as conn:
                running_hub_db_utils.update_planned_workout(conn, workout_data)
                conn.commit()
            self.logger.info(f"Successfully updated workout ID: {workout_data.get('id')}")
        except Exception as e:
            self.logger.error(f"Service error updating workout {workout_data.get('id')}: {e}")
            raise exception_utils.ServiceError("Failed to update workout.")

    def delete_planned_workout(self, workout_id: int) -> None:
        """Deletes a planned workout."""
        try:
            with self._get_connection() as conn:
                running_hub_db_utils.delete_planned_workout_by_id(conn, workout_id)
                conn.commit()
            self.logger.info(f"Successfully deleted workout ID: {workout_id}")
        except Exception as e:
            self.logger.error(f"Service error deleting workout {workout_id}: {e}")
            raise exception_utils.ServiceError("Failed to delete workout.")
        
    def get_recent_strava_activity_ids(self) -> list[Any]:
        """Retrieves recent Strava IDs"""
        try:
            with self._get_connection() as conn:
                recent_activities = running_hub_db_utils.get_recent_strava_activity_ids(conn)
            self.logger.info(f"Sucesfully pulled recent Strava IDs")
            return recent_activities
        except Exception as e:
            self.logger.error(f"Service error pulling recent Strava IDs: {e}")
            raise exception_utils.ServiceError("Unable to pull recent Strava IDs")