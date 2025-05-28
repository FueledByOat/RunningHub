# statistics_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dateutil.relativedelta import relativedelta

from services.base_service import BaseService
from utils import db_utils, format_utils, exception_utils

class StatisticsService(BaseService):
    """Service for generating running statistics."""
    
    def get_statistics(self, period: str, units: str) -> Dict[str, Any]:
        """Get comprehensive running statistics for a time period."""
        now = datetime.now()
        start_date = self._get_period_start_date(now, period)
        date_range = self._get_date_range_label(period)
        
        try:
            with self._get_connection() as conn:
                conn.row_factory = db_utils.dict_factory
                
                # Get summary statistics
                summary_stats = self._get_summary_statistics(start_date, units)
                
                # Get chart data
                weekly_distances = self._get_weekly_distances(now)
                pace_data = self._get_pace_trends(units)
                
                # Get shoe data and recent activities
                shoe_data = self._get_shoe_usage(start_date, units)
                recent_activities = self._get_recent_activities(units)
                
                return {
                    'period': period,
                    'units': units,
                    'date_range': date_range,
                    'stats': summary_stats,
                    'weekly_distances': json.dumps(weekly_distances),
                    'pace_dates': pace_data['dates'],
                    'pace_values': json.dumps(pace_data['values']),
                    'shoes': shoe_data,
                    'recent_activities': recent_activities,
                    'start_date': start_date,
                    **summary_stats  # Flatten for template compatibility
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            raise exception_utils.DatabaseError(f"Failed to get statistics: {e}")
    
    def _get_period_start_date(self, now: datetime, period: str) -> str:
        """Calculate start date based on period."""
        if period == 'week':
            return (now - timedelta(days=7)).strftime('%Y-%m-%d')
        elif period == 'month':
            return (now - timedelta(days=30)).strftime('%Y-%m-%d')
        elif period == 'year':
            return (now - relativedelta(years=1)).strftime('%Y-%m-%d')
        else:  # 'all'
            return '2000-01-01'
    
    def _get_date_range_label(self, period: str) -> str:
        """Get human-readable date range label."""
        labels = {
            'week': 'Last 7 days',
            'month': 'Last 30 days',
            'year': 'Last 12 months',
            'all': 'All time'
        }
        return labels.get(period, 'All time')
    
    def _get_summary_statistics(self, start_date: str, units: str) -> Dict[str, Any]:
        """Get summary statistics for the period."""
        # Total activities
        try:
            with self._get_connection() as conn:
                total_activities = db_utils.get_total_activities_count(conn, start_date)
        except Exception as e:
            self.logger.error(f"Error getting total activity count statistics: {e}")
        
        # Total elevation
        try:
            with self._get_connection() as conn:
                total_elevation = db_utils.get_total_elevation_count(conn, start_date)
        except Exception as e:
            self.logger.error(f"Error getting total activity elevation statistics: {e}")

        # Total distance
        try:
            with self._get_connection() as conn:
                total_distance_meters = db_utils.get_total_distance_count(conn, start_date)
        except Exception as e:
            self.logger.error(f"Error getting total activity distance statistics: {e}")
        
        total_distance = (
            round(total_distance_meters / 1609, 2) if units == 'mi' 
            else round(total_distance_meters / 1000, 2)
        )
        
        # Total time
        try:
            with self._get_connection() as conn:
                total_seconds = db_utils.get_total_time(conn, start_date)
        except Exception as e:
            self.logger.error(f"Error getting total activity time statistics: {e}")
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        total_time = f"{hours}h {minutes}m"
        
        # Total calories (estimated from kilojoules)
        try:
            with self._get_connection() as conn:
                calories_result = db_utils.get_total_calories(conn, start_date)
        except Exception as e:
            self.logger.error(f"Error getting total activity calorie statistics: {e}")
        
        total_calories = (
            round(calories_result / 4.184) 
            if calories_result else 0
        )
        
        return {
            'total_activities': total_activities,
            'total_distance': total_distance,
            'total_time': total_time,
            'total_calories': total_calories,
            'total_elevation': total_elevation
        }
    
    def _get_weekly_distances(self, now: datetime) -> List[float]:
        """Get distances for the last 7 days."""
        weekly_distances = [0] * 7
        seven_days_ago = now - timedelta(days=7)
        
        try:
            with self._get_connection() as conn:
                activities = db_utils.get_weekly_distances(conn, seven_days_ago)
        except Exception as e:
            self.logger.error(f"Error getting weekly distance statistics: {e}")
        
        for activity in activities:
            try:
                activity_date = datetime.strptime(
                    activity['start_date_local'].split('T')[0], '%Y-%m-%d'
                )
                day_diff = (now - activity_date).days
                if 0 <= day_diff < 7:
                    index = 6 - day_diff  # Index 6 is today, 0 is 6 days ago
                    weekly_distances[index] += round(activity['distance'] / 1000, 2)
            except (ValueError, IndexError):
                continue
        
        return weekly_distances
    
    def _get_pace_trends(self, units: str) -> Dict[str, List]:
        """Get pace trends for the last 10 activities."""        
        try:
            with self._get_connection() as conn:
                activities = db_utils.get_pace_trends(conn)
        except Exception as e:
            self.logger.error(f"Error getting pace trend statistics: {e}")

        dates = []
        values = []
        
        for activity in reversed(activities):  # Chronological order
            try:
                date_str = datetime.strptime(
                    activity['start_date_local'].split('T')[0], '%Y-%m-%d'
                ).strftime('%d %b')
                dates.append(date_str)
                
                # Calculate pace
                distance_km = activity['distance'] / 1000
                if units == 'mi':
                    distance_km = distance_km * 0.621371  # Convert to miles
                
                pace_minutes = (activity['moving_time'] / 60) / distance_km
                values.append(round(pace_minutes, 2))
            except (ValueError, ZeroDivisionError):
                continue
        
        return {'dates': dates, 'values': values}
    
    def _get_shoe_usage(self, start_date: str, units: str) -> List[Dict[str, Any]]:
        """Get shoe usage statistics."""

        try:
            with self._get_connection() as conn:
                shoes = db_utils.get_shoe_usage(conn, start_date)
        except Exception as e:
            self.logger.error(f"Error getting shoe usage statistics: {e}")

        shoe_data = []
        for shoe in shoes:
            if shoe['total_distance'] and shoe['last_used']:
                last_used_date = datetime.strptime(
                    shoe['last_used'].split('T')[0], '%Y-%m-%d'
                ).strftime('%d %b %Y')
                
                distance = (
                    round(shoe['total_distance'] / 1609, 2) if units == 'mi'
                    else round(shoe['total_distance'] / 1000, 2)
                )
                
                shoe_data.append({
                    'name': shoe['gear_id'],
                    'distance': distance,
                    'activities': shoe['activities'],
                    'last_used': last_used_date
                })
        
        return shoe_data
    
    def _get_recent_activities(self, units: str) -> List[Dict[str, Any]]:
        """Get recent activities with formatted data."""

        try:
            with self._get_connection() as conn:
                activities = db_utils.get_recent_activities(conn)
        except Exception as e:
            self.logger.error(f"Error getting recent activity statistics: {e}")

        activities_list = []
        for activity in activities:
            date_str = datetime.strptime(
                activity['start_date_local'].split('T')[0], '%Y-%m-%d'
            ).strftime('%d %b')
            
            distance = (
                round(activity['distance'] / 1609, 2) if units == 'mi'
                else round(activity['distance'] / 1000, 2)
            )
            
            time_str = format_utils.format_time(activity['moving_time'])
            pace = format_utils.format_pace(distance, activity['moving_time'], units=units)
            
            activities_list.append({
                'id': activity['id'],
                'name': activity['name'],
                'date': date_str,
                'distance': distance,
                'time': time_str,
                'pace': pace
            })
        
        return activities_list