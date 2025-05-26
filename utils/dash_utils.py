# dash_utils.py
"""Utilities for preparing and rendering Dash application components.

This module provides functions for:
- Polyline decoding for mapping
- Database operations for activity streams
- Data interpolation for visualization
"""

import json
import logging
import sqlite3
from typing import List, Tuple, Optional, Union

import numpy as np
import polyline

from utils import db_utils
from utils import exception_utils
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Decode a polyline string to list of (lat, lon) coordinates.
    
    Args:
        polyline_str: Encoded polyline string
        
    Returns:
        List of (latitude, longitude) tuples
        
    Raises:
        DataProcessingError: If polyline cannot be decoded
    """
    if not polyline_str:
        return []
        
    try:
        return polyline.decode(polyline_str)
    except Exception as e:
        logger.error(f"Failed to decode polyline: {e}")
        raise exception_utils.DataProcessingError(f"Polyline decode failed: {e}") from e

def get_streams_data(activity_id: int, db_path: str = Config.DB_PATH) -> Tuple[List[float], ...]:
    """Retrieve activity stream data from database.
    
    Args:
        activity_id: Activity identifier
        db_path: Database file path
        
    Returns:
        Tuple of (distance, heartrate, altitude, power, time) lists
        
    Raises:
        DataProcessingError: If data cannot be retrieved or parsed
    """
    query = """
        SELECT distance_data, heartrate_data, altitude_data, watts_data, time_data 
        FROM streams 
        WHERE activity_id = ?
    """
    
    try:
        with db_utils.get_db_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (activity_id,))
            row = cur.fetchone()
            
        if not row:
            logger.warning(f"No stream data found for activity {activity_id}")
            return [], [], [], [], []
            
        # Parse JSON data with error handling
        parsed_data = []
        for i, data in enumerate(row):
            try:
                parsed = json.loads(data) if data else []
                # Replace None values with 0 using numpy for efficiency
                if parsed:
                    cleaned = np.where(np.array(parsed) == None, 0, parsed).tolist()
                else:
                    cleaned = []
                parsed_data.append(cleaned)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse stream data column {i}: {e}")
                parsed_data.append([])
                
        return tuple(parsed_data)
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving streams for activity {activity_id}: {e}")
        raise exception_utils.DataProcessingError(f"Database query failed: {e}") from e


def interpolate_to_common_x(
    x_ref: List[float], 
    y_raw: List[float], 
    x_raw: List[float]
) -> List[Optional[float]]:
    """Interpolate y_raw values to match x_ref domain.
    
    Args:
        x_ref: Reference x values to interpolate to
        y_raw: Y values to interpolate
        x_raw: X values corresponding to y_raw
        
    Returns:
        Interpolated y values aligned with x_ref, None for out-of-bounds
        
    Raises:
        ValueError: If input arrays are insufficient for interpolation
    """
    if len(x_raw) < 2 or len(y_raw) < 2:
        raise ValueError("Need at least 2 points for interpolation")
        
    if len(x_ref) < 1:
        return []
        
    if len(x_raw) != len(y_raw):
        raise ValueError("x_raw and y_raw must have same length")
    
    try:
        # Convert to numpy arrays for efficiency
        x_ref_arr = np.array(x_ref)
        x_raw_arr = np.array(x_raw)
        y_raw_arr = np.array(y_raw)
        
        # Interpolate with NaN for out-of-bounds
        y_interp = np.interp(x_ref_arr, x_raw_arr, y_raw_arr, left=np.nan, right=np.nan)
        
        # Convert NaN to None for consistency
        return [None if np.isnan(y) else float(y) for y in y_interp]
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise exception_utils.DataProcessingError(f"Interpolation error: {e}") from e


def validate_stream_data(data: List[Union[float, int, None]]) -> List[float]:
    """Validate and clean stream data.
    
    Args:
        data: Raw stream data list
        
    Returns:
        Cleaned data with None values replaced by 0
    """
    if not data:
        return []
        
    return [0.0 if x is None else float(x) for x in data]