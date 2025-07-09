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
        """Retrieves all activity information with formatted data."""
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

    ## Running Condition Score (RCS) calculations

    def _score_dewpoint(self, dew_c):
        if dew_c < 10:
            return 10
        elif dew_c < 15:
            return 8
        elif dew_c < 18:
            return 6
        elif dew_c < 21:
            return 4
        elif dew_c < 24:
            return 2
        else:
            return 0

    def _score_feelslike(self, temp_c):
        # Ideal range: 5°C to 15°C
        if 5 <= temp_c <= 15:
            return 10
        elif 0 <= temp_c < 5 or 15 < temp_c <= 20:
            return 8
        elif -5 <= temp_c < 0 or 20 < temp_c <= 25:
            return 5
        elif -10 <= temp_c < -5 or 25 < temp_c <= 30:
            return 3
        else:
            return 0

    def _score_wind(self, wind_kph):
        if wind_kph < 10:
            return 10
        elif wind_kph < 20:
            return 6
        elif wind_kph < 30:
            return 3
        else:
            return 0

    def _score_precipitation(self, chance_of_rain, chance_of_snow):
        chance = max(chance_of_rain or 0, chance_of_snow or 0)
        if chance > 80:
            return 0
        elif chance > 60:
            return 3
        elif chance > 40:
            return 5
        elif chance > 20:
            return 8
        else:
            return 10

    def _score_uv(self, uv):
        if uv is None:
            return 10
        elif uv <= 2:
            return 10
        elif uv <= 5:
            return 8
        elif uv <= 7:
            return 5
        elif uv <= 10:
            return 3
        else:
            return 0

    def _score_temp(self, temp_c):
        # Ideal: 5-15°C
        if 5 <= temp_c <= 15:
            return 10
        elif 0 <= temp_c < 5 or 15 < temp_c <= 20:
            return 7
        elif -5 <= temp_c < 0 or 20 < temp_c <= 25:
            return 4
        else:
            return 1

    def calculate_rcs(self, dew_c, feelslike_c, wind_kph, chance_of_rain, chance_of_snow, uv, temp_c):
        """Calculates Running Condition Score (RCS) using the following structure
        
        Component	Importance	Weight	Notes
        Dew Point	High	0.25	Correlates with discomfort; humid air makes sweating ineffective.
        Feels-like Temp	High	0.25	Accounts for both wind chill and heat index.
        Wind Speed	Medium	0.15	Higher winds can impair pace and comfort.
        Precipitation Chance	Medium	0.15	Wet conditions can impact traction and comfort.
        UV Index	Lower	0.10	Strong sun can increase fatigue and sunburn risk.
        Actual Temp	Lower	0.10	Secondary to feels-like, but still impactful.

        Total: 1.00

        All sub-scores will be normalized to a 0–10 scale, where 10 is ideal and 0 is poor.
        
        """
        scores = {
            "dew": self._score_dewpoint(dew_c),
            "feelslike": self._score_feelslike(feelslike_c),
            "wind": self._score_wind(wind_kph),
            "precip": self._score_precipitation(chance_of_rain, chance_of_snow),
            "uv": self._score_uv(uv),
            "temp": self._score_temp(temp_c)
        }

        rcs = (
            scores["dew"] * 0.25 +
            scores["feelslike"] * 0.25 +
            scores["wind"] * 0.15 +
            scores["precip"] * 0.15 +
            scores["uv"] * 0.10 +
            scores["temp"] * 0.10
        )

        return round(rcs, 2)