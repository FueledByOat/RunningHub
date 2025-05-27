# trophy_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from services.base_service import BaseService
from utils import db_utils, format_utils, exception_utils

class TrophyService(BaseService):
    """Service for managing personal records and achievements."""
    
    def get_personal_records(self, units: str = 'mi') -> List[Dict[str, Any]]:
        """Get all personal records for standard race distances."""
        records = []
        
        # Define race distances with their search ranges (in meters)
        race_distances = [
            ('5K', 4500, 5500),
            ('8K', 7600, 8400),
            ('10K', 9500, 10500),
            ('Half Marathon', 20750, 22000),
            ('Marathon', 41800, 43050)
        ]
        
        try:
            with self._get_connection() as conn:
                conn.row_factory = db_utils.dict_factory
                
                # Get records for standard distances
                for distance_name, min_distance, max_distance in race_distances:
                    record = self._get_distance_record(conn, distance_name, min_distance, max_distance, units)
                    if record:
                        records.append(record)
                
                # Get longest run
                longest_run = self._get_longest_run(conn, units)
                if longest_run:
                    records.append(longest_run)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Error getting personal records: {e}")
            raise exception_utils.DatabaseError(f"Failed to get personal records: {e}")
    
    def _get_distance_record(self, conn: sqlite3.Connection, distance_name: str, 
                           min_distance: int, max_distance: int, units: str) -> Optional[Dict[str, Any]]:
        """Get personal record for a specific distance range."""
        try:
            result = conn.execute(
                """SELECT id, name, distance, moving_time, start_date_local
                   FROM activities
                   WHERE type = 'Run' AND distance BETWEEN ? AND ?
                   ORDER BY moving_time ASC
                   LIMIT 1""",
                (min_distance, max_distance)
            ).fetchone()
            
            if not result:
                return None
            
            return self._format_record(result, distance_name, units)
            
        except Exception as e:
            self.logger.error(f"Error getting {distance_name} record: {e}")
            return None
    
    def _get_longest_run(self, conn: sqlite3.Connection, units: str) -> Optional[Dict[str, Any]]:
        """Get longest run record."""
        try:
            result = conn.execute(
                """SELECT id, name, distance, moving_time, start_date_local
                   FROM activities
                   WHERE type = 'Run'
                   ORDER BY distance DESC
                   LIMIT 1"""
            ).fetchone()
            
            if not result:
                return None
            
            distance = (
                round(result['distance'] / 1609, 2) if units == 'mi'
                else round(result['distance'] / 1000, 2)
            )
            
            return self._format_record(result, f"Longest Run: {distance} {units}", units)
            
        except Exception as e:
            self.logger.error(f"Error getting longest run: {e}")
            return None
    
    def _format_record(self, activity: Dict[str, Any], distance_name: str, units: str) -> Dict[str, Any]:
        """Format a personal record for display."""
        try:
            date_str = datetime.strptime(
                activity['start_date_local'].split('T')[0], '%Y-%m-%d'
            ).strftime('%d %b %Y')
            
            time_str = format_utils.format_time(activity['moving_time'])
            
            distance = (
                round(activity['distance'] / 1609, 2) if units == 'mi'
                else round(activity['distance'] / 1000, 2)
            )
            
            pace = format_utils.format_pace(distance, activity['moving_time'], units=units)
            
            return {
                'distance': distance_name,
                'time': time_str,
                'pace': pace,
                'date': date_str
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting record: {e}")
            return {
                'distance': distance_name,
                'time': 'N/A',
                'pace': 'N/A',
                'date': 'N/A'
            }