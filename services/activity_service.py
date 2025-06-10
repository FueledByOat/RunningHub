# activity_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""

import sqlite3
from typing import Dict, Any, Optional

from services.base_service import BaseService
from utils import format_utils, exception_utils
from utils.db import db_utils
import utils.db.running_hub_db_utils as running_hub_db_utils


class ActivityService(BaseService):
    """Service for handling activity-related operations."""
    
    def get_latest_activity_id(self) -> Optional[int]:
        """Get the ID of the most recent run activity."""
        try:
            with self._get_connection() as conn:
                return running_hub_db_utils.get_latest_activity_id(conn=conn, activity_types=['Run'])
        except Exception as e:
            self.logger.error(f"Error getting latest activity: {e}")
            return None
        
    def get_formatted_activity_page_details(self, activity_id: int, units: str = 'mi') -> Optional[Dict[str, Any]]:
        """Retreives all activity information with formatted data."""
        try:
            with self._get_connection() as conn:
                activity = running_hub_db_utils.get_activity_details_by_id(conn=conn, activity_id=activity_id)
                return self._format_activity_data(activity, units)
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error getting activity {activity_id}: {e}")
            raise exception_utils.DatabaseError(f"Failed to get activity data: {e}")
    
    def _format_activity_data(self, activity: Dict[str, Any], units: str) -> Dict[str, Any]:
        """Format activity data for display, including unit conversion for miles and kilometers."""
        # Distance conversion
        distance_meters = activity['distance']
        if units == 'mi':
            activity['distance'] = round(distance_meters / 1609, 2)
        else:
            activity['distance'] = round(distance_meters * 0.001, 2)
        
        # Pace calculation
        activity['average_pace'] = format_utils.format_pace(
            activity['distance'], activity['moving_time'], units=units
        )
        
        # Time formatting
        activity['moving_time'] = format_utils.format_time(activity['moving_time'])
        
        # Speed conversion (m/s to mph or kmh)
        speed_multiplier = 2.237 if units == 'mi' else 3.6
        activity['average_speed'] = round(activity['average_speed'] * speed_multiplier, 1)
        activity['max_speed'] = round(activity['max_speed'] * speed_multiplier, 1)
        
        # Heart rate and other metrics
        activity['max_heartrate'] = round(activity['max_heartrate'])
        activity['average_heartrate'] = round(activity['average_heartrate'])
        activity['kilojoules'] = round(activity['kilojoules'])
        
        # Cadence handling (double for running, zero for cycling)
        if activity['type'] == 'Ride':
            activity['average_cadence'] = 0
        else:
            activity['average_cadence'] = int(round(activity['average_cadence'] * 2, 0))
        
        # Date/time formatting
        activity['start_date'], activity['start_time'] = format_utils.format_datetime(
            activity['start_date_local']
        )
        
        return activity