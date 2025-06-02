# coach_g_service.py
"""
Service layer for running Coach G Language Model.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""
import logging

import json
from typing import Dict, Any

from services.base_service import BaseService
from utils import db_utils, format_utils, exception_utils, language_model_utils

class CoachGService(BaseService):
    """Service for handling database queries."""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.coach_g = language_model_utils.LanguageModel()


    def daily_training_summary(self):
        return self.coach_g.generate_daily_training_summary()

    def general_reply(self, user_query: str):
        return self.coach_g.generate_general_coach_g_reply(user_query)

    def latest_training_metrics(self, sql_query: str, param_input: str = '{}') -> Dict[str, Any]:
        """Execute a database query with parameters."""
        if not sql_query.strip().lower().startswith("select"):
            raise exception_utils.DatabaseError("Only SELECT queries allowed")
        
        try:
            # Parse parameters
            param_input = param_input.strip() or '{}'
            params = json.loads(param_input)
            
            if not isinstance(params, dict):
                raise ValueError("Parameters must be a JSON object")
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query, params)
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return {
                    'columns': columns,
                    'rows': rows,
                    'error': None
                }
                
        except json.JSONDecodeError:
            raise exception_utils.DatabaseError("Invalid JSON format in parameters")
        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            raise exception_utils.DatabaseError(f"Query failed: {e}")
