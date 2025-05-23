# format_utils.py
"""Utilities for formatting activity data for frontend display."""

import logging
from datetime import datetime
from typing import Tuple, Union

from utils import exception_utils

logger = logging.getLogger(__name__)

def format_pace(distance: float, total_seconds: int, units: str = 'miles') -> str:
    """Format average pace per distance unit.
    
    Args:
        distance: Total distance (miles or kilometers)
        total_seconds: Total time in seconds
        units: Distance units ('miles', 'mi', 'kilometers', 'km')
        
    Returns:
        Formatted pace string (MM:SS per unit)
        
    Raises:
        FormattingError: If inputs are invalid
    """
    # Input validation
    if not isinstance(distance, (int, float)) or not isinstance(total_seconds, (int, float)):
        raise exception_utils.FormattingError("Distance and time must be numeric")
    
    if distance <= 0:
        raise exception_utils.FormattingError("Distance must be positive")
        
    if total_seconds < 0:
        raise exception_utils.FormattingError("Time cannot be negative")
    
    # Calculate seconds per unit
    try:
        if units.lower() in ('miles', 'mi'):
            seconds_per_unit = total_seconds / distance
        elif units.lower() in ('kilometers', 'km'):
            seconds_per_unit = total_seconds / distance
        else:
            raise exception_utils.FormattingError(f"Unsupported units: {units}")
        
        minutes = int(seconds_per_unit // 60)
        seconds = int(seconds_per_unit % 60)
        
        return f"{minutes:02d}:{seconds:02d}"
        
    except ZeroDivisionError:
        raise exception_utils.FormattingError("Cannot calculate pace with zero distance")
    except Exception as e:
        logger.error(f"Unexpected error formatting pace: {e}")
        raise exception_utils.FormattingError(f"Pace formatting failed: {e}") from e


def format_time(total_seconds: Union[int, float]) -> str:
    """Format total time as HH:MM:SS or MM:SS.
    
    Args:
        total_seconds: Total time in seconds
        
    Returns:
        Formatted time string (HH:MM:SS if >= 1 hour, MM:SS otherwise)
        
    Raises:
        FormattingError: If input is invalid
    """
    if not isinstance(total_seconds, (int, float)):
        raise exception_utils.FormattingError("Time must be numeric")
        
    if total_seconds < 0:
        raise exception_utils.FormattingError("Time cannot be negative")
    
    try:
        total_seconds = int(total_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
            
    except Exception as e:
        logger.error(f"Unexpected error formatting time: {e}")
        raise exception_utils.FormattingError(f"Time formatting failed: {e}") from e


def format_datetime(datetime_string: str) -> Tuple[str, str]:
    """Parse ISO datetime string and return formatted date and time.
    
    Args:
        datetime_string: ISO datetime string (e.g., '2025-05-10T12:11:52Z')
        
    Returns:
        Tuple of (formatted_date, formatted_time)
        
    Raises:
        FormattingError: If datetime string cannot be parsed
    """
    if not isinstance(datetime_string, str):
        raise exception_utils.FormattingError("Datetime input must be a string")
    
    if not datetime_string.strip():
        raise exception_utils.FormattingError("Datetime string cannot be empty")
    
    try:
        # Parse ISO format with Z suffix
        dt = datetime.strptime(datetime_string, "%Y-%m-%dT%H:%M:%SZ")
        
        formatted_date = dt.strftime("%Y-%m-%d")
        formatted_time = dt.strftime("%H:%M:%S")
        
        return formatted_date, formatted_time
        
    except ValueError as e:
        logger.error(f"Failed to parse datetime '{datetime_string}': {e}")
        raise exception_utils.FormattingError(f"Invalid datetime format: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error formatting datetime: {e}")
        raise exception_utils.FormattingError(f"Datetime formatting failed: {e}") from e


def format_distance(distance: float, units: str = 'miles', precision: int = 2) -> str:
    """Format distance with specified precision and units.
    
    Args:
        distance: Distance value
        units: Distance units for display
        precision: Decimal places to display
        
    Returns:
        Formatted distance string with units
    """
    if not isinstance(distance, (int, float)):
        raise exception_utils.FormattingError("Distance must be numeric")
        
    if distance < 0:
        raise exception_utils.FormattingError("Distance cannot be negative")
    
    try:
        return f"{distance:.{precision}f} {units}"
    except Exception as e:
        logger.error(f"Error formatting distance: {e}")
        raise exception_utils.FormattingError(f"Distance formatting failed: {e}") from e