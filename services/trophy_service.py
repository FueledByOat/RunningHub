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
from utils import format_utils, exception_utils
from utils.db import db_utils
import utils.db.running_hub_db_utils as running_hub_db_utils

class TrophyService(BaseService):
    """Service for managing personal records and achievements."""
    
    def get_personal_records(self, units: str = 'mi') -> List[Dict[str, Any]]:
        """Get all personal records for standard race distances."""
        records = []
        
        # Define race distances with their search ranges (in meters)
        race_distances = [
            ('1 Mile', 1500, 1700),            
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
                    try:
                        record = running_hub_db_utils.get_distance_record(conn, distance_name, min_distance, max_distance, units)
                        if record:
                            records.append(self._format_record(record, distance_name, units))
                    except Exception as e:
                        self.logger.error(f"Error getting {distance_name} record: {e}")
                
                # Get longest run
                try:
                    longest_run = running_hub_db_utils.get_longest_run(conn, units)
                    if longest_run:
                        distance = (
                        round(longest_run['distance'] / 1609, 2) if units == 'mi'
                        else round(longest_run['distance'] / 1000, 2)
                        )
                        records.append(self._format_record(longest_run, f"Longest Run: {distance} {units}", units))
                
                except Exception as e:
                    self.logger.error(f"Error getting longest run: {e}")
                
                return records
                
        except Exception as e:
            self.logger.error(f"Error getting personal records: {e}")
            raise exception_utils.DatabaseError(f"Failed to get personal records: {e}")
        
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