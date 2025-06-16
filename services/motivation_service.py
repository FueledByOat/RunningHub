# motivation_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from services.base_service import BaseService
from services.coach_g_service import CoachGService
from utils.db import db_utils
from utils.db import language_db_utils

class MotivationService(CoachGService):
    """Service for managing personal records and achievements."""