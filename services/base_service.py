# base_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import uuid 
from typing import Dict, List, Any, Optional, Tuple
from dateutil.relativedelta import relativedelta

from flask import send_from_directory, abort
from werkzeug.utils import secure_filename
import os

from utils import format_utils, exception_utils
from config import Config
from utils.db import db_utils

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Abstract base class for all services."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with error handling."""
        try:
            return db_utils.get_db_connection(self.db_path)
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise exception_utils.DatabaseError(f"Failed to connect to database: {e}")